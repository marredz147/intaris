"""Native OpenAI Decisions transport for Intaris finite safety questions."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

import httpx

from intaris import model_retry
from intaris.config import DecisionsConfig
from intaris.finite_questions import FiniteQuestionClient
from intaris.input_budget import Capacity, CapacityCache, fit, native_source_sections
from intaris.llm import LLMTemporaryError

# Native Decisions answers each field as an independent typed question. The
# shared L1 rules arrive only once as cached context, so every question carries
# its own field definition and every choice its meaning; bare label names give
# a general model nothing to calibrate against.
_HYBRID_FIELDS: dict[str, tuple[str, dict[str, str]]] = {
    "relationship": (
        "How does the proposed action relate to the user's current task?",
        {
            "directly_serves_intent": "Work the user asked for.",
            "supports_intent": (
                "A reasonable step toward the task, including exploration, reading, "
                "testing, verification, setup, local commits, worktrees, logs and "
                "diagnostics, even if not named verbatim."
            ),
            "unrelated": (
                "Serves no part of the task and is no reasonable step toward it."
            ),
            "conflicts_with_intent": "Contradicts the user's goal or restrictions.",
            "unclear": "The evidence is insufficient to relate it to the task.",
        },
    ),
    "risk": (
        "What is the actual operational impact of this call? Permission is not "
        "this question.",
        {
            "low": "Routine, limited, reversible operation.",
            "medium": (
                "Meaningful side effect with bounded scope and ordinary recovery."
            ),
            "high": (
                "Material security, privacy, financial, production, or "
                "data-integrity impact; destructive or difficult to reverse."
            ),
            "critical": (
                "Immediate catastrophic or broadly destructive impact, credential "
                "exfiltration, or clearly malicious operation."
            ),
            "unknown": (
                "Impact evidence is missing; never uncertainty about permission and "
                "never a tie between low and medium."
            ),
        },
    ),
    "mutation": (
        "Does the whole call change state? Judge it including nested commands, "
        "redirection and scripts; use unclear if the evidence cannot establish it.",
        {
            "read_only": "Observes only; no file, service or remote state changes.",
            "mutating": (
                "Changes state: file writes, redirection, installs, MCP updates, "
                "remote writes or deletions."
            ),
            "unclear": "The evidence cannot establish whether state changes.",
        },
    ),
    "consequence": (
        "Is the call's effect consequential?",
        {
            "routine": "Ordinary supporting work.",
            "consequential": (
                "Publication, destruction, sensitive export or significant external "
                "mutation."
            ),
            "unclear": "The evidence cannot establish the consequence.",
        },
    ),
    "contextual_restrictions": (
        "Does this action violate an identifiable instruction that currently applies "
        "to this actor, operation and target? A listed restriction is not itself a "
        "conflict.",
        {
            "consistent": (
                "No currently applicable restriction prohibits this action. If a "
                "current restriction's applicability to it is unclear, choose "
                "unresolved."
            ),
            "conflicts": (
                "An identifiable instruction that currently applies to this actor, "
                "operation and target prohibits it. A related or superseded source, "
                "an assistant status report or a lifted phase limit is not a "
                "prohibition."
            ),
            "unresolved": (
                "A real evidence gap about an applicable restriction, not merely that "
                "this supporting step was not separately requested."
            ),
        },
    ),
    "target_scope": (
        "Do the action's actual targets (paths, repositories, hosts, services, "
        "recipients), including nested and overridden ones, fit the targets the "
        "task and its constraints cover?",
        {
            "consistent": (
                "Every actual target is determined and fits the task or its "
                "applicable constraints."
            ),
            "conflicts": (
                "A target is excluded by an applicable constraint or contradicts the "
                "task."
            ),
            "unresolved": (
                "A target, including a nested or overridden one, cannot be determined "
                "from the evidence."
            ),
        },
    ),
    "effect_read": (
        "Does the call read or observe data?",
        {},
    ),
    "effect_write": (
        "Does the call write or modify local or remote data?",
        {},
    ),
    "effect_execute": (
        "Does the call run code, scripts or commands?",
        {},
    ),
    "effect_transmit": (
        "Does data or an action leave the user's environment to third parties: "
        "messages, email, remote writes, git push, deploy, purchases? Local files, "
        "/tmp, local containers, port-forwards, local builds and tests need not "
        "transmit; classify the whole call including nested commands, uploads and "
        "remote/target overrides.",
        {},
    ),
    "effect_delete": (
        "Does the call remove data?",
        {},
    ),
}
_EFFECT_CHOICES = {
    "yes": (
        "Any part of the call, including nested commands, uploads and remote or "
        "target overrides, has this effect."
    ),
    "no": "No part of the call has this effect.",
    "unknown": "The evidence cannot establish whether the call has this effect.",
}
_REFERENCE_FIELDS = {
    "target_reference": "target_scope",
    "restriction_reference": "contextual_restrictions",
}
_REFERENCE_CHOICES = {
    "none": (
        "No supplied item conflicts; none is not itself evidence of a conflict or "
        "of a missing fact."
    ),
    "proposed action": "The proposed action itself.",
    "session assignment": "The session's task assignment.",
    "parent assignment": "The parent session's task assignment.",
    "latest user message": "The latest user message.",
}


class DecisionsTemporaryError(LLMTemporaryError):
    """A transient native transport failure, eligible for generative L1 fallback."""

    def __init__(self, category: str, status_code: int | None = None) -> None:
        super().__init__("Decisions is temporarily unavailable")
        self.category = category
        self.status_code = status_code


class DecisionsClient(FiniteQuestionClient):
    """Use native typed questions, without chat adaptation or capacity probing."""

    provider = "decisions"
    provider_label = "Decisions"

    def __init__(
        self,
        config: DecisionsConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = config.model
        self._api_key = config.api_key
        self._minimum_confidence = config.minimum_confidence
        self.l1_review_enabled = config.l1_review_enabled
        self._hybrid_minimum_posterior = config.hybrid_minimum_posterior
        self._hybrid_minimum_margin = config.hybrid_minimum_margin
        self._decision_question = True
        self._diagnostics = True
        self._approval_risk_confidence = None
        self._timeout_s = config.timeout_ms / 1000.0
        self._request_path = "decisions"
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/") + "/",
            timeout=self._timeout_s,
            transport=transport,
        )
        self._closed = False
        self._input_capacity = CapacityCache()
        self._input_view: ContextVar[dict[str, Any] | None] = ContextVar(
            "decisions_input_view", default=None
        )
        self._usage: ContextVar[dict[str, Any] | None] = ContextVar(
            "decisions_usage", default=None
        )

    def close(self) -> None:
        """Close the HTTP connection pool idempotently."""
        if not self._closed:
            self._client.close()
            self._closed = True

    def input_capacity(self) -> Capacity:
        """Use the bounded unknown-capacity policy; no capacity API is assumed."""
        return Capacity()

    @property
    def input_view(self) -> dict[str, Any] | None:
        """Return accounting for the actual native request envelope."""
        return self._input_view.get()

    @property
    def usage(self) -> dict[str, Any] | None:
        """Validated provider counts; cache counters are subsets of input tokens."""
        return self._usage.get()

    def _model_name(self, response: dict[str, Any]) -> str:
        if response.get("model") != self._model:
            raise ValueError("Decisions returned an unexpected model")
        return self._model

    def _hybrid_questions(
        self, questions: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Give each typed question its field definition and choice meanings."""
        described = {}
        for key, question in questions.items():
            question = {**question, "criteria": dict(question["criteria"])}
            definition, choices = _HYBRID_FIELDS.get(key, ("", {}))
            if key.startswith("effect_"):
                choices = _EFFECT_CHOICES
            status = _REFERENCE_FIELDS.get(key)
            if status is not None:
                definition = (
                    f"Which supplied item does this action conflict with for "
                    f"{status}? Select it only when {status} is conflicts; "
                    "otherwise none."
                )
                choices = {
                    label: _REFERENCE_CHOICES.get(
                        label,
                        f"The supplied target or evidence item {label!r} that this "
                        "action conflicts with.",
                    )
                    for label in question["criteria"]
                }
            if definition:
                question["instructions"] = definition + "\n" + question["instructions"]
            for label in question["criteria"]:
                if label in choices:
                    question["criteria"][label] = choices[label]
            described[key] = question
        return described

    def low_confidence(self, diagnostics: dict[str, Any]) -> dict[str, Any] | None:
        """Request generative L1 review only for valid, threshold-abstained answers."""
        if diagnostics.get("validation_error"):
            return None
        uncertainty = diagnostics.get("classification_uncertainty", {})
        if uncertainty.get("validation"):
            return None
        fields = {
            key: {"threshold_abstained": True}
            for key in ("alignment", "risk", "decision")
            if uncertainty.get(key) is True
        }
        for key, row in diagnostics.get("distributions", {}).items():
            # Native confidence is also required: a peaked posterior must not
            # hide the provider explicitly reporting an uncertain answer.
            if (
                row.get("threshold_abstained")
                or row["provider_confidence"] < self._minimum_confidence
            ):
                fields[key] = {
                    field: row[field]
                    for field in (
                        "posterior",
                        "margin",
                        "provider_confidence",
                        "choice_mismatch",
                        "threshold_abstained",
                    )
                }
        return (
            {
                "code": "low_confidence",
                "fields": fields,
                "minimum_confidence": self._minimum_confidence,
                "thresholds": diagnostics.get("thresholds"),
            }
            if fields
            else None
        )

    def _request(
        self, *, state: dict[str, Any], questions: dict[str, Any]
    ) -> dict[str, Any]:
        with model_retry.model_call_deadline(self._timeout_s):
            try:
                return self._request_with_budget(state=state, questions=questions)
            except (
                httpx.NetworkError,
                httpx.TimeoutException,
                httpx.RemoteProtocolError,
            ) as exc:
                raise DecisionsTemporaryError(type(exc).__name__) from exc
            except httpx.HTTPStatusError as exc:
                try:
                    body = exc.response.json()
                except ValueError:
                    body = None
                # Preserve the native temporary-error contract for server
                # failures; retry eligibility remains owned by model_retry.
                if exc.response.status_code >= 500 or model_retry.is_transient_status(
                    exc.response.status_code, body
                ):
                    raise DecisionsTemporaryError(
                        "http_status", exc.response.status_code
                    ) from exc
                raise

    def _request_with_budget(
        self, *, state: dict[str, Any], questions: dict[str, Any]
    ) -> dict[str, Any]:
        if not self._api_key:
            raise RuntimeError("Decisions credential is unavailable")
        self._input_view.set(None)
        self._usage.set(None)
        native_questions = []
        for name, question in questions.items():
            criteria = question["criteria"]
            native = {
                "type": "predicate" if question["type"] == "noul" else "choice",
                "name": name,
                "instructions": question["instructions"],
            }
            if native["type"] == "choice":
                native["choices"] = [
                    {"value": value, "description": description}
                    for value, description in criteria.items()
                ]
            else:
                # Predicate questions have no criteria field on the native wire.
                native["instructions"] += "\n" + "\n".join(
                    f"{value}: {description}" for value, description in criteria.items()
                )
            native_questions.append(native)
        rules = state.get("evaluation_rules")
        stable_rules = isinstance(rules, str) and bool(rules)
        dynamic_state = (
            {key: value for key, value in state.items() if key != "evaluation_rules"}
            if stable_rules
            else state
        )
        content = []
        if stable_rules:
            content.append(
                {
                    "type": "input_text",
                    "text": "evaluation_rules:\n" + rules,
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            )
        json_index = len(content)
        content.append({"type": "input_json", "data": dynamic_state})
        outgoing = {
            "model": self._model,
            "input": [{"role": "user", "content": content}],
            "questions": native_questions,
        }
        if stable_rules:
            outgoing["prompt_cache_options"] = {"mode": "explicit"}
        with native_source_sections(json_index=json_index):
            outgoing, view = fit(outgoing, self.input_capacity(), jev=True)
        self._input_view.set(view)
        self._validate_input(outgoing["input"][0]["content"][json_index]["data"])
        response = self._post_with_transient_retries(
            headers={"Authorization": f"Bearer {self._api_key}"}, json=outgoing
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Decisions response must be an object")
        self._model_name(payload)
        answers = payload.get("answers")
        if not isinstance(answers, list) or len(answers) != len(questions):
            raise ValueError("Decisions response has an invalid answer count")
        normalized = {}
        for (name, question), native, answer in zip(
            questions.items(), native_questions, answers, strict=True
        ):
            if not isinstance(answer, dict) or answer.get("name") != name:
                raise ValueError("Decisions response has an invalid answer order/name")
            if answer.get("type") != native["type"]:
                # Refusal is an unavailable assessment, never an invented distribution.
                raise ValueError("Decisions answer was refused or has an invalid type")
            if native["type"] == "predicate":
                normalized[name] = {
                    "type": "noul",
                    "noul": self._probability(answer.get("probability"), name),
                }
            else:
                choices = tuple(question["criteria"])
                distribution = answer.get("probabilities")
                if not isinstance(distribution, list) or len(distribution) != len(
                    choices
                ):
                    raise ValueError("Decisions answer has an invalid distribution")
                probabilities = {}
                for row in distribution:
                    if not isinstance(row, dict):
                        raise ValueError("Decisions probability must be an object")
                    value = row.get("value")
                    if (
                        not isinstance(value, str)
                        or value not in choices
                        or value in probabilities
                    ):
                        raise ValueError("Decisions probability has an invalid choice")
                    probabilities[value] = self._probability(
                        row.get("probability"), name
                    )
                if abs(sum(probabilities.values()) - 1.0) > 1e-6:
                    raise ValueError("Decisions probabilities do not sum to 1")
                choice = answer.get("choice")
                if not isinstance(choice, str) or choice not in choices:
                    raise ValueError("Decisions answer has an invalid selected choice")
                normalized[name] = {
                    "type": "choice",
                    "choice": choice,
                    "confidence": self._probability(answer.get("confidence"), name),
                    "probabilities": probabilities,
                }
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            raise ValueError("Decisions response is missing usage")
        validated_usage: dict[str, Any] = {
            key: self._non_negative_int(usage.get(key), key)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
        if "input_tokens_details" in usage:
            details = usage["input_tokens_details"]
            if not isinstance(details, dict):
                raise ValueError("Decisions input token details must be an object")
            cache_counts = {}
            for key in ("cached_tokens", "cache_write_tokens"):
                if key in details:
                    count = self._non_negative_int(
                        details[key], f"input_tokens_details.{key}"
                    )
                    if count > validated_usage["input_tokens"]:
                        raise ValueError(
                            "Decisions cache count exceeds input token count"
                        )
                    cache_counts[key] = count
            validated_usage["input_tokens_details"] = cache_counts
        if "output_tokens_details" in usage:
            details = usage["output_tokens_details"]
            if not isinstance(details, dict):
                raise ValueError("Decisions output token details must be an object")
            validated_usage["output_tokens_details"] = (
                {
                    "reasoning_tokens": self._non_negative_int(
                        details["reasoning_tokens"],
                        "output_tokens_details.reasoning_tokens",
                    )
                }
                if "reasoning_tokens" in details
                else {}
            )
        self._usage.set(validated_usage)
        return {
            "model": payload["model"],
            "answers": normalized,
            "usage": validated_usage,
        }

    @staticmethod
    def _validate_input(state: dict[str, Any]) -> None:
        """Enforce native input_json structural limits before dispatch."""
        from intaris.input_budget import ContextInsufficientError, serialized

        values = 0

        def visit(value: Any, depth: int) -> None:
            nonlocal values
            values += 1
            if depth > 32 or values > 10000:
                raise ContextInsufficientError({"native_input_limit": True})
            if isinstance(value, dict):
                for child in value.values():
                    visit(child, depth + 1)
            elif isinstance(value, list):
                for child in value:
                    visit(child, depth + 1)

        visit(state, 1)
        if len(serialized(state)) > 256 * 1024:
            raise ContextInsufficientError({"native_input_limit": True})
