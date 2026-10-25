"""Audit provenance for the bounded Decisions-to-generative-L1 review path."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from functools import wraps
from typing import Any

from intaris import model_retry
from intaris.redactor import redact


def is_decisions(client: Any) -> bool:
    """Identify the native client without changing Jev or mock-client semantics."""
    return getattr(type(client), "provider", None) == "decisions"


# Native single-decision L1: the three-question Decisions form (aligned, risk,
# decision) answers only calls whose sole open question is the routine intent
# assessment. Anything the three answers cannot ground stays with the factual
# L1: strict scope checks, operator or extracted user restrictions, read-only
# overlays, configured rules, unknown or consequential effects, secrets,
# protected repository/deploy targets and targets outside the working directory.
_NATIVE_PROFILES = frozenset({"balanced", "bounded_autonomy"})
_NATIVE_EFFECTS = frozenset({"read", "write", "execute"})


def native_decision_eligible(
    client: Any,
    state: Any,
    result: Any,
    *,
    profile: str,
    tool: str,
    args: dict,
    working_directory: str | None,
) -> bool:
    """Whether a confident native approve can stand in for the intent assessment."""
    import os

    from intaris.classifier import reads_secret_location
    from intaris.effect_scope import requires_explicit_grant
    from intaris.evaluator import _protected_edit_target

    if not is_decisions(client) or profile not in _NATIVE_PROFILES:
        return False
    if [item.id for item in result.outstanding_obligations] != ["intent.assessment"]:
        return False
    if (
        state.read_only_required
        or state.advisory_clauses
        or state.process_lifecycle
        or (state.workflow_scope or {}).get("declared")
    ):
        return False
    effects = set(state.value("effects", None) or ())
    if not effects or not effects <= _NATIVE_EFFECTS:
        return False
    if state.value("secret_argument") or requires_explicit_grant(state):
        return False
    for source in state.sources.values():
        if source.get("constraint_items") or source.get("interpretation_missing"):
            return False
    if not state.value("paths_complete") or not working_directory:
        return False
    root = os.path.normpath(working_directory)
    paths = state.value("paths", None) or []
    if not os.path.isabs(root) or not all(os.path.isabs(path) for path in paths):
        return False
    if any(
        os.path.commonpath([root, os.path.normpath(path)]) != root
        or _protected_edit_target(path)
        for path in paths
    ):
        return False
    return not reads_secret_location(tool, args, working_directory)


def review_deadline(client: Any, llm: Any = None) -> AbstractContextManager[None]:
    """Keep both L1 stages inside the existing caller deadline, including standalone use."""
    if not is_decisions(client):
        return nullcontext()
    timeout_ms = getattr(llm, "_timeout_ms", None)
    seconds = (
        timeout_ms / 1000 if type(timeout_ms) in (int, float) else client._timeout_s
    )
    return model_retry.model_call_deadline(seconds)


def primary_deadline(client: Any) -> AbstractContextManager[None]:
    """Bound native retries while reserving half the remaining L1 budget for review."""
    remaining = model_retry.remaining_seconds()
    return (
        model_retry.evaluation_deadline(min(client._timeout_s, remaining / 2))
        if is_decisions(client) and remaining is not None
        else nullcontext()
    )


def runtime_deadline(function: Callable[..., Any]) -> Callable[..., Any]:
    """Keep hybrid policy-owned optional review in the same L1 budget."""

    @wraps(function)
    def bounded(runtime: Any, *args: Any, **kwargs: Any) -> Any:
        with review_deadline(runtime.jev, runtime.llm):
            return function(runtime, *args, **kwargs)

    return bounded


def optional_reason(
    client: Any, diagnostics: dict, *, escalated: bool, critical: bool
) -> dict | None:
    """The decision owner supplies the actual escalation, never a risk heuristic."""
    route = diagnostics.get("backend_provenance") or {}
    return (
        {"code": "optional_escalation"}
        if is_decisions(client)
        and client.l1_review_enabled
        and escalated
        and not critical
        and not diagnostics.get("validation_error")
        and route.get("final_backend") == "decisions"
        and len(route.get("attempts", [])) == 1
        else None
    )


def failure(exc: Exception) -> dict:
    """Describe a failure without copying provider messages, URLs or credentials."""
    return {
        "category": getattr(exc, "category", type(exc).__name__),
        "http_status": getattr(exc, "status_code", None),
    }


def temporary_reason(exc: Exception) -> dict:
    """Identify the explicitly supported transient-only fallback."""
    return {"code": "temporary_unavailable", **failure(exc)}


def redact_metadata(metadata: dict) -> dict:
    """Redact metadata values on a copy; structural keys and token counts survive."""

    def visit(value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {name: visit(child, name) for name, child in value.items()}
        if isinstance(value, list):
            return [visit(child) for child in value]
        if isinstance(value, str):
            # This typed classification is not an HTTP Authorization credential.
            if key == "authorization" and value in {
                "explicit",
                "implicit",
                "absent",
                "unclear",
            }:
                return value
            return redact({key: value})[key] if key is not None else redact(value)
        return value

    return visit(metadata)


def attempt(client: Any, diagnostics: dict, view: dict | None = None) -> dict:
    """Record scores and request accounting, never evaluated text or credentials."""
    backend = getattr(type(client), "provider", "llm")
    model = getattr(client, "_model", None)
    actual_view = view if view is not None else getattr(client, "input_view", None)
    return redact_metadata(
        {
            "backend": backend,
            "status": "failed"
            if diagnostics.get("failure") or diagnostics.get("validation_error")
            else "completed",
            "model": model if isinstance(model, str) else None,
            "model_view": actual_view if isinstance(actual_view, dict) else None,
            **(
                {"usage": client.usage}
                if is_decisions(client) and isinstance(client.usage, dict)
                else {}
            ),
            "diagnostics": {
                key: diagnostics[key]
                for key in (
                    "aligned_probability",
                    "alignment_confidence",
                    "risk_confidence",
                    "decision_confidence",
                    "minimum_confidence",
                    "threshold",
                    "classification_uncertainty",
                    "validation_error",
                    "distributions",
                    "thresholds",
                    "confidence_semantics",
                    "validation_errors",
                    "failure",
                )
                if key in diagnostics
            },
        }
    )


def provenance(attempts: list[dict], reason: dict | None = None) -> dict:
    """Describe the L1 route independently of the unchanged evaluation_path."""
    final = next(
        (row for row in reversed(attempts) if row.get("status") != "failed"),
        attempts[-1],
    )
    return redact_metadata(
        {
            "route": "decisions_to_llm" if reason else attempts[-1]["backend"],
            "initial_backend": attempts[0]["backend"],
            "final_backend": final["backend"],
            "final_model": final["model"],
            "fallback_reason": reason,
            "attempts": attempts,
        }
    )


def preserve_restrictions(final_view: dict, initial_view: dict | None) -> dict:
    """A second assessment cannot erase an earlier partial-input safety restriction."""
    view = dict(final_view)
    if isinstance(initial_view, dict):
        if initial_view.get("affirmative_grants_eligible") is False:
            view["affirmative_grants_eligible"] = False
        if initial_view.get("source_coverage") == "partial":
            view["source_coverage"] = "partial"
    return view
