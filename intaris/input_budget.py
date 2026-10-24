"""Complete-request capacity accounting; discovery is bounded, local and read-only."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from intaris import model_retry

UNKNOWN_CAPACITY_BYTES = 98304
logger = logging.getLogger(__name__)
_sections = ContextVar("input_budget_sections", default=())


@contextmanager
def source_sections(sections):
    """Builder-owned JSON section locations; never derive trust from model/user text."""
    token = _sections.set(tuple(sections))
    try:
        yield
    finally:
        _sections.reset(token)


@contextmanager
def native_source_sections(*, json_index=0):
    """Translate builder-owned state paths to the native Decisions envelope."""
    with source_sections(
        (
            ("input", 0, "content", json_index, "data", *path[1:])
            if path and path[0] == "state"
            else path,
            start,
            end,
        )
        for path, start, end in _sections.get()
    ):
        yield


def located_source(prompt, sources, *, path):
    """Bind the exact canonical source serialization, refusing ambiguous copies."""
    text = json.dumps(sources, ensure_ascii=False)
    if prompt.count(text) != 1:
        return []
    start = prompt.index(text)
    return [(tuple(path), start, start + len(text))]


class ContextInsufficientError(Exception):
    """Essential model evidence does not fit; this is not a missing session or consent."""

    def __init__(self, diagnostics):
        super().__init__(
            "context_insufficient: essential evidence exceeds available model input capacity"
        )
        self.diagnostics = diagnostics


class ModelResponseError(ValueError):
    """Malformed model output, distinct from a missing session at API boundaries."""


def capacity_rejection(body):
    error = body.get("error", body) if isinstance(body, dict) else {}
    code = error.get("code") if isinstance(error, dict) else None
    return code in {
        "context_length_exceeded",
        "input_too_long",
        "max_tokens_exceeded",
        "upstream_context_length_exceeded",
        "upstream_input_too_long",
        "upstream_max_tokens_exceeded",
    }


def serialized(value):
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def redact_scalars(value):
    """Redact complete text before counting/excerpting, not semantic/schema field names."""
    from intaris.redactor import redact

    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: redact_scalars(child) for key, child in value.items()}
    if isinstance(value, list):
        return [redact_scalars(child) for child in value]
    return value


@dataclass(frozen=True)
class Capacity:
    context_window: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    fingerprint: str | None = None
    status: str = "unknown"
    unknown_bytes_limit: int = UNKNOWN_CAPACITY_BYTES

    @classmethod
    def from_model(cls, model):
        if not isinstance(model, dict):
            return cls()
        limits = {
            name: value if type(value := model.get(name)) is int and value > 0 else None
            for name in ("context_window", "max_input_tokens", "max_output_tokens")
        }
        fingerprint = model.get("resolution_fingerprint")
        return cls(
            **limits,
            fingerprint=fingerprint
            if isinstance(fingerprint, str) and len(fingerprint) <= 128
            else None,
            status="known" if any(limits.values()) else "unknown",
        )


class CapacityCache:
    """One transport-instance cache; credentials/tenants cannot share this instance."""

    def __init__(self):
        self._key = None
        self._capacity = Capacity()
        self._expires = 0.0
        self._confirmed_at = 0.0
        self._revision = 0
        self._lock = threading.Lock()

    def invalidate(self):
        with self._lock:
            self._capacity = Capacity()
            self._expires = 0.0
            self._confirmed_at = 0.0
            self._revision += 1

    def resolve(self, key, probe, *, refresh=False, timeout_seconds=0.2):
        """Resolve route-local capacity, optionally bypassing the discovery TTL."""
        with self._lock:
            now = time.monotonic()
            if key != self._key:
                self._key, self._capacity, self._expires = key, Capacity(), 0.0
                self._confirmed_at = 0.0
                self._revision += 1
            stale_deadline = self._confirmed_at + 120
            if self._capacity.status == "stale" and now >= stale_deadline:
                self._capacity, self._expires = Capacity(), 0.0
                self._revision += 1
            if not refresh and now < self._expires:
                return self._capacity
            previous = self._capacity
            revision = self._revision
        timeout = model_retry.retry_timeout(0, timeout_seconds, first_attempt=True)
        if timeout is None:
            return Capacity()
        try:
            capacity = Capacity.from_model(probe(timeout))
            if capacity.context_window is None and capacity.max_input_tokens is None:
                logger.warning(
                    "Model capacity metadata has no input limits "
                    "(timeout_seconds=%.3f)",
                    timeout,
                )
        except Exception as exc:
            import httpx
            from openai import APIConnectionError, APITimeoutError

            status = getattr(exc, "status_code", None)
            if status is None:
                status = getattr(getattr(exc, "response", None), "status_code", None)
            logger.warning(
                "Model capacity metadata probe failed "
                "(error_type=%s, http_status=%s, timeout_seconds=%.3f)",
                type(exc).__name__,
                status,
                timeout,
            )
            transient = isinstance(
                exc,
                (
                    TimeoutError,
                    httpx.TransportError,
                    APIConnectionError,
                    APITimeoutError,
                ),
            ) or status in {429, 500, 502, 503, 504}
            # A contradicted/new route never inherits the previous route's limits.
            capacity = (
                Capacity(
                    previous.context_window,
                    previous.max_input_tokens,
                    previous.max_output_tokens,
                    previous.fingerprint,
                    "stale",
                )
                if transient
                and previous.status in {"known", "stale"}
                and time.monotonic() < stale_deadline
                else Capacity()
            )
        with self._lock:
            now = time.monotonic()
            if capacity.status == "stale" and now >= stale_deadline:
                capacity = Capacity()
            if key == self._key:
                if revision != self._revision:
                    if (
                        self._capacity.status == "stale"
                        and now >= self._confirmed_at + 120
                    ):
                        return Capacity()
                    return self._capacity
                self._capacity = capacity
                self._revision += 1
                self._expires = (
                    min(now + 5, stale_deadline)
                    if capacity.status == "stale"
                    else now + 60
                )
                if capacity.status == "known":
                    self._confirmed_at = now
        return capacity


def measure(payload, capacity=Capacity(), *, jev=False):
    """Estimate the complete envelope with a rough character heuristic, not a tokenizer."""
    raw = serialized(payload)
    characters = len(raw.decode("utf-8"))
    reserve = (
        None if jev else payload.get("max_completion_tokens", payload.get("max_tokens"))
    )
    reserve = reserve if type(reserve) is int and reserve > 0 else None
    # Deliberately rough, not an exact count. Real JSON-heavy evaluator packets
    # measured 2.7-3.6 characters per provider token, so /4 under-counted.
    framing = 64 + 16 * len(payload.get("messages", [])) if not jev else 128
    estimate = math.ceil(characters / 3) + framing
    margin = max(256, math.ceil(estimate * 0.08))
    ceilings = []
    if capacity.max_input_tokens is not None:
        ceilings.append(capacity.max_input_tokens)
    if not jev and capacity.context_window is not None and reserve is not None:
        ceilings.append(max(0, capacity.context_window - reserve))
    ceiling = min(ceilings) if ceilings else None
    overflow = (
        estimate + margin > ceiling
        if ceiling is not None
        else len(raw) > capacity.unknown_bytes_limit
    )
    return {
        "version": "complete-model-input-v1",
        "digest": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "characters": characters,
        "estimator": "rough_characters_div3_plus_framing",
        "estimated_input_tokens": estimate,
        "framing_tokens": framing,
        "margin_tokens": margin,
        "input_ceiling_tokens": ceiling,
        "requested_output_reserve": reserve,
        "hidden_output_overhead": "unknown"
        if jev
        else "included_in_requested_output_cap",
        "capacity_status": capacity.status,
        "capacity_fingerprint": capacity.fingerprint,
        "application_fallback": "UNKNOWN-CAPACITY" if ceiling is None else None,
        "application_fallback_bytes": capacity.unknown_bytes_limit
        if ceiling is None
        else None,
        "overflow": overflow,
    }


def enforce(payload, capacity=Capacity(), *, jev=False):
    """Never silently edit an executable action or manufacture permission reasoning."""
    view = measure(payload, capacity, jev=jev)
    if view["overflow"]:
        raise ContextInsufficientError(view)
    return view


_CONTEXT_ORIGINS = {
    "generated_summary",
    "inherited_context",
    "inherited_assignment",
    "linked_non_authorizing_context",
    "agent_context",
    "tool_context",
}


def contextual_excerpt(text, *, spans=(), keep=512):
    """Excerpt non-authorizing text only; offsets refer to complete redacted input."""
    from intaris.redactor import redact

    text = str(redact(text))
    ranges = [(0, min(keep, len(text))), (max(0, len(text) - keep), len(text))]
    for start, end in spans:
        if 0 <= start < end <= len(text):
            ranges.append((max(0, start - 200), min(len(text), end + 200)))
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        elif end > start:
            merged.append((start, end))
    output, position = [], 0
    for start, end in merged:
        if start > position:
            output.append(
                f"\n[OMITTED original chars {position}:{start}; context only]\n"
            )
        output.append(text[start:end])
        position = end
    if position < len(text):
        output.append(
            f"\n[OMITTED original chars {position}:{len(text)}; context only]\n"
        )
    return "".join(output), [list(pair) for pair in merged]


def middle_excerpt(text: str, limit: int) -> str:
    """Keep both ends of model-view text; never change the audited original."""
    if len(text) <= limit:
        return text
    digest = hashlib.sha256(text.encode()).hexdigest()
    marker = (
        f"\n[OMITTED MIDDLE: {len(text)} original chars; sha256={digest}; "
        "missing text is unknown, not permission or evidence of no restrictions]\n"
    )
    keep = max(1, (limit - len(marker)) // 2)
    return text[:keep] + marker + text[-keep:]


def _bounded_model_view(payload, capacity, *, jev):
    """Last-resort bounded projection of dynamic input, never protocol/schema.

    The caller still owns the complete action, source registry and audit. This
    projection is only for inference; identifiers and dict keys are not rewritten.
    JSON message bodies remain JSON rather than being cut across escape sequences.
    """
    original = copy.deepcopy(payload)
    omitted = []
    identity_collections = {
        "sources",
        "originals",
        "source_metadata",
        "citation_sources",
        "reference_ids",
        "citable",
        "grant_citable",
        "current_request_sources",
        "scope_refs",
        "linked_to",
    }

    def identity_path(path):
        return any(
            part in identity_collections for part in path if isinstance(part, str)
        )

    def mark_incomplete(value):
        if isinstance(value, dict):
            if "ref" in value and any(
                key in value for key in ("origin", "body_availability", "content")
            ):
                value["render_truncated"] = True
                value["body_availability"] = "partial_model_view"
                value["grant_eligible"] = False
            for child in value.values():
                mark_incomplete(child)
        elif isinstance(value, list):
            for child in value:
                mark_incomplete(child)

    def annotate(value):
        mark_incomplete(value)
        if isinstance(value, dict):
            value["model_view_coverage"] = {
                "status": "partial",
                "affirmative_grants_eligible": False,
                "notice": (
                    "Missing text is unknown, not evidence of no restrictions. "
                    "Linked approvals cannot grant partially visible scope."
                ),
            }
        return value

    def bound(value, limit, path, sequence_limit):
        if isinstance(value, str):
            # JSON documents rendered into message strings retain their structure.
            if value[:1] in ("{", "["):
                try:
                    decoded = json.loads(value)
                except ValueError:
                    pass
                else:
                    if isinstance(decoded, (dict, list)):
                        before = len(omitted)
                        projected = bound(decoded, limit, path, sequence_limit)
                        if len(omitted) != before:
                            annotate(projected)
                        return json.dumps(
                            projected,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
            if len(value) > limit:
                omitted.append({"path": list(path), "original_chars": len(value)})
                return middle_excerpt(value, limit)
            return value
        if isinstance(value, dict):
            entries = list(value.items())
            if len(entries) > sequence_limit and not identity_path(path):
                keep = sequence_limit // 2
                omitted.append({"path": list(path), "original_fields": len(entries)})
                marker = "__intaris_omitted_middle_fields__"
                while marker in value:
                    marker += "_"
                entries = (
                    entries[:keep]
                    + [
                        (
                            marker,
                            f"{len(entries) - 2 * keep} fields omitted; unknown evidence",
                        )
                    ]
                    + entries[-keep:]
                )
            return {
                key: bound(child, limit, (*path, key), sequence_limit)
                for key, child in entries
            }
        if isinstance(value, list):
            indexed = list(enumerate(value))
            if len(indexed) > sequence_limit and not identity_path(path):
                keep = sequence_limit // 2
                omitted.append({"path": list(path), "original_items": len(indexed)})
                indexed = (
                    indexed[:keep]
                    + [
                        (
                            keep,
                            f"[OMITTED MIDDLE: {len(value) - 2 * keep} items; unknown evidence]",
                        )
                    ]
                    + indexed[-keep:]
                )
            return [
                bound(child, limit, (*path, index), sequence_limit)
                for index, child in indexed
            ]
        return value

    def longest(value):
        if isinstance(value, str):
            return len(value)
        if isinstance(value, dict):
            return max((longest(child) for child in value.values()), default=0)
        if isinstance(value, list):
            return max((longest(child) for child in value), default=0)
        return 0

    def project(limit, sequence_limit):
        nonlocal omitted
        result = copy.deepcopy(original)
        omitted = []
        if jev:
            # Schema/options and trusted evaluation instructions stay unchanged.
            if "state" in result or "input" in result:
                native = "input" in result
                json_index = (
                    next(
                        index
                        for index, part in enumerate(result["input"][0]["content"])
                        if part["type"] == "input_json"
                    )
                    if native
                    else None
                )
                parent = result["input"][0]["content"][json_index] if native else result
                key = "data" if native else "state"
                prefix = (
                    ("input", 0, "content", json_index, "data")
                    if native
                    else ("state",)
                )
                state = parent[key]
                if isinstance(state, dict):
                    parent[key] = {
                        key: child
                        if key == "evaluation_rules"
                        else bound(child, limit, (*prefix, key), sequence_limit)
                        for key, child in state.items()
                    }
                else:
                    parent[key] = bound(state, limit, prefix, sequence_limit)
                if omitted:
                    annotate(parent[key])
        else:
            for index, message in enumerate(result.get("messages", [])):
                if message.get("role") not in {"system", "developer"}:
                    message["content"] = bound(
                        message.get("content"),
                        limit,
                        ("messages", index, "content"),
                        sequence_limit,
                    )
        return result, measure(result, capacity, jev=jev)

    def accepted(result, view):
        return result, {
            **view,
            "trimmed": bool(omitted),
            "affirmative_grants_eligible": False,
            "source_coverage": "partial",
            "middle_truncations": omitted,
        }

    # Cut only as much as the budget requires: search the largest per-string
    # limit that fits, so one oversized prompt keeps most of its middle instead
    # of collapsing to a fixed excerpt. The fixed ladder stays the fallback when
    # many small items, not one long string, cause the overflow.
    low, high, best = 8192, longest(original), None
    while low <= high:
        middle = (low + high) // 2
        _, view = project(middle, 256)
        if view["overflow"]:
            high = middle - 1
        else:
            best, low = middle, middle + 1
    if best is not None:
        return accepted(*project(best, 256))
    for limit, sequence_limit in (
        (8192, 256),
        (4096, 128),
        (2048, 64),
        (1024, 32),
        (512, 16),
    ):
        result, view = project(limit, sequence_limit)
        if not view["overflow"]:
            return accepted(result, view)
    raise ContextInsufficientError({**view, "middle_truncations": omitted})


def fit(payload, capacity=Capacity(), *, jev=False):
    """Fit source excerpts, then bounded partial views of oversized dynamic input.

    Partial views cannot establish affirmative grants. Full action and audit
    records remain unchanged; schemas and trusted evaluation instructions stay.

    Builders retain source dictionaries, including inside rendered JSON/text sections.
    The adapter recognizes JSON structure, never edits executable strings or guesses
    whether a human statement is relevant from L1/generated labels.
    """
    payload = redact_scalars(payload)
    initial = measure(payload, capacity, jev=jev)
    if not initial["overflow"]:
        return payload, {**initial, "trimmed": False, "omitted_sources": []}
    result = copy.deepcopy(payload)
    documents = []

    rows, items, containers = [], [], []

    def collect(value, source_rows=False):
        if isinstance(value, dict):
            if isinstance(value.get("sources"), list):
                containers.append(value)
            if source_rows and isinstance(value.get("ref"), str) and "origin" in value:
                rows.append(value)
            if isinstance(value.get("source_ref"), str):
                items.append(value)
            for key, child in value.items():
                if key in {
                    "action",
                    "arguments",
                    "args",
                    "args_redacted",
                    "tool_call",
                    "command",
                }:
                    continue
                if isinstance(child, (dict, list)):
                    collect(child, key in {"sources", "originals"})
        elif isinstance(value, list):
            for child in value:
                collect(child, source_rows)

    for path, start, end in _sections.get():
        parent = result
        try:
            for part in path[:-1]:
                parent = parent[part]
            key = path[-1]
            value = parent[key]
            if start is None:
                collect(value)
                continue
            obj = json.loads(value[start:end])
            collect(obj)
            documents.append((parent, key, value, [(start, end, obj)]))
        except (KeyError, IndexError, TypeError, ValueError):
            # A repair/alternate protocol changed framing: no speculative trimming.
            continue
    protected = {
        row["ref"] for row in rows if row.get("origin") not in _CONTEXT_ORIGINS
    }
    # Controller-linked antecedents are protected together with their human source.
    changed = True
    while changed:
        before = set(protected)
        for row in rows:
            links = row.get("scope_refs") or []
            links = list(links) if isinstance(links, list) else []
            links += [row.get("linked_to"), row.get("assessment_reference")]
            links = {ref for ref in links if isinstance(ref, str)}
            if row["ref"] in protected or links & protected:
                protected.add(row["ref"])
                protected.update(links)
        changed = before != protected

    def render():
        for container in containers:
            if any(
                isinstance(row, dict) and row.get("render_truncated")
                for row in container["sources"]
            ):
                container["coverage"] = "partial"
        for parent, key, original, sections in documents:
            text = original
            for start, end, obj in reversed(sections):
                text = (
                    text[:start]
                    + json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                    + text[end:]
                )
            parent[key] = text

    omitted = []
    candidates = sorted(
        (
            row
            for row in rows
            if row["ref"] not in protected and isinstance(row.get("content"), str)
        ),
        key=lambda row: len(row["content"]),
        reverse=True,
    )
    # Equivalent context copies remain traceable rather than silently disappearing.
    seen = {}
    for row in candidates:
        identity = (row["ref"], row["content"])
        if identity in seen:
            row["content"] = f"[DUPLICATE context body: {row['ref']}]"
            row["body_availability"] = "duplicate_model_view"
            row["render_truncated"] = True
            omitted.append({"ref": row["ref"], "reason": "duplicate_context_body"})
        else:
            seen[identity] = True
    render()
    for row in candidates:
        if not measure(result, capacity, jev=jev)["overflow"]:
            break
        original = row["content"]
        spans = [
            tuple(span)
            for span in row.get("context_spans", [])
            if isinstance(span, list)
            and len(span) == 2
            and all(type(position) is int for position in span)
            and 0 <= span[0] < span[1] <= len(original)
        ]
        for item in items:
            quote = item.get("quote")
            if (
                item.get("source_ref") == row["ref"]
                and isinstance(quote, str)
                and quote
            ):
                start, end = item.get("start"), item.get("end")
                if (
                    type(start) is int
                    and type(end) is int
                    and original[start:end] == quote
                ):
                    spans.append((start, end))
                else:
                    start = original.find(quote)
                    while start >= 0:
                        spans.append((start, start + len(quote)))
                        start = original.find(quote, start + len(quote))
        excerpt, retained = contextual_excerpt(original, spans=spans)
        if len(excerpt.encode()) >= len(original.encode()):
            continue
        row.update(
            content=excerpt,
            render_truncated=True,
            body_availability="bounded_excerpt",
            render_original_bytes=len(original.encode()),
            retained_char_ranges=retained,
        )
        omitted.append(
            {
                "ref": row["ref"],
                "retained_char_ranges": retained,
                "original_bytes": len(original.encode()),
            }
        )
        render()
    view = measure(result, capacity, jev=jev)
    view.update(trimmed=True, omitted_sources=omitted)
    if omitted:
        view["source_coverage"] = "partial"
    if view["overflow"]:
        result, bounded = _bounded_model_view(result, capacity, jev=jev)
        bounded["omitted_sources"] = omitted
        return result, bounded
    return result, view
