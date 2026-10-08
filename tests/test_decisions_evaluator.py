"""Native Decisions L1 routing, policy and bounded fallback tests."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import BadRequestError

from intaris.config import DecisionsConfig
from intaris.decision import EvaluationResult
from intaris.decisions import (
    DecisionsClient,
    DecisionsError,
    DecisionsProtocolError,
    DecisionsTemporaryError,
)
from intaris.evaluator import Evaluator
from intaris.llm import LLMClient, LLMTemporaryError


def _setup(*, llm: MagicMock | None = None, jev: MagicMock | None = None):
    audit = MagicMock()
    audit.get_recent.return_value = []
    audit.get_user_decisions.return_value = []
    audit.find_approved_escalation.return_value = None
    sessions = MagicMock()
    sessions.get.return_value = {
        "user_id": "user",
        "session_id": "session",
        "intention": "Edit documentation",
        "status": "active",
        "policy": None,
    }
    decisions = MagicMock()
    decisions._timeout_ms = 3000
    llm = llm if llm is not None else MagicMock()
    llm._model = "legacy-model"
    evaluator = Evaluator(
        llm=llm,
        jev=jev,
        decisions=decisions,
        llm_timeout_ms=4000,
        session_store=sessions,
        audit_store=audit,
    )
    return evaluator, decisions, llm, audit, sessions


def _call(evaluator: Evaluator, **kwargs):
    return evaluator.evaluate(
        user_id="user",
        session_id="session",
        agent_id=None,
        tool="write",
        args={"filePath": "docs/a.md", "content": "text"},
        **kwargs,
    )


def _result(**kwargs):
    return EvaluationResult(
        aligned=kwargs.get("aligned", True),
        risk=kwargs.get("risk", "low"),
        decision=kwargs.get("decision", "approve"),
        reasoning="Native classification",
        metadata={
            "backend": "openai_decisions",
            "requested_model": "native",
            "model": "reported",
        },
    )


def test_native_selection_context_audit_and_fast_paths():
    evaluator, decisions, llm, audit, _ = _setup(jev=MagicMock())
    decisions.evaluate.return_value = _result()
    result = _call(evaluator, context={"evaluation_metadata": {"backend": "injected"}})
    assert result["decision"] == "approve"
    assert result["evaluation_metadata"]["model"] == "reported"
    assert decisions.evaluate.call_args.kwargs["timeout_ms"] == 2000
    assert "Edit documentation" in decisions.evaluate.call_args.kwargs["user_prompt"]
    assert "anti" in decisions.evaluate.call_args.kwargs["system_prompt"].lower()
    assert (
        audit.insert.call_args.kwargs["args_redacted"]["__intaris_context"][
            "evaluation_metadata"
        ]["backend"]
        == "openai_decisions"
    )
    llm.generate.assert_not_called()
    evaluator._jev.evaluate_tool_call.assert_not_called()
    assert (
        evaluator.evaluate(
            user_id="user",
            session_id="session",
            agent_id=None,
            tool="read",
            args={"filePath": "docs/a.md"},
        )["path"]
        == "fast"
    )
    assert (
        evaluator.evaluate(
            user_id="user",
            session_id="session",
            agent_id=None,
            tool="bash",
            args={"command": "rm -rf /"},
        )["path"]
        == "critical"
    )
    assert decisions.evaluate.call_count == 1


@pytest.mark.parametrize(
    "error",
    [
        DecisionsError("auth"),
        DecisionsProtocolError("malformed"),
        ValueError("invalid"),
    ],
)
def test_non_transient_errors_do_not_fallback(error):
    evaluator, decisions, llm, _, _ = _setup()
    decisions.evaluate.side_effect = error
    with pytest.raises(type(error)):
        _call(evaluator)
    llm.generate.assert_not_called()


def test_transient_fallback_uses_llm_not_jev_and_sanitizes_reason():
    jev = MagicMock()
    evaluator, decisions, llm, audit, _ = _setup(jev=jev)
    decisions.evaluate.side_effect = DecisionsTemporaryError("secret https://host/key")
    llm.generate.return_value = json.dumps(
        {"aligned": True, "risk": "low", "reasoning": "ok", "decision": "approve"}
    )
    result = _call(evaluator)
    assert result["decision"] == "approve"
    assert result["evaluation_metadata"]["backend"] == "llm"
    assert result["evaluation_metadata"]["primary_backend"] == "openai_decisions"
    assert result["evaluation_metadata"]["fallback_reason"] == "temporary_unavailable"
    assert "secret" not in json.dumps(audit.insert.call_args.kwargs)
    assert llm.generate.call_count == 1
    assert "deadline" in llm.generate.call_args.kwargs
    jev.evaluate_tool_call.assert_not_called()


@pytest.mark.parametrize(
    "native,expected",
    [
        (_result(decision="escalate"), "escalate"),
        (_result(aligned=False, risk="high", decision="escalate"), "escalate"),
        (_result(aligned=False, risk="low", decision="escalate"), "escalate"),
    ],
)
def test_explicit_escalation_and_uncertainty(native, expected):
    evaluator, decisions, llm, _, _ = _setup()
    decisions.evaluate.return_value = native
    assert _call(evaluator)["decision"] == expected
    llm.generate.assert_not_called()


@pytest.mark.parametrize("status", ["refusal", "low_confidence"])
def test_refusal_and_uncertainty_stay_on_native_path(status):
    evaluator, decisions, llm, _, _ = _setup()
    native = _result(aligned=False, risk="high", decision="escalate")
    native.metadata["status"] = status
    decisions.evaluate.return_value = native
    result = _call(evaluator)
    assert result["decision"] == "escalate"
    assert result["evaluation_metadata"]["status"] == status
    llm.generate.assert_not_called()


@pytest.mark.parametrize("status", ["refusal", "low_confidence"])
@pytest.mark.parametrize("cap", [None, "approve"])
@pytest.mark.parametrize("precedent", [False, True])
def test_real_native_uncertainty_and_refusal_policy(status, cap, precedent):
    payload = {
        "model": "reported",
        "answers": [
            {
                "name": "aligned",
                "type": "refusal" if status == "refusal" else "predicate",
                **({} if status == "refusal" else {"probability": 0.51}),
            },
            {
                "name": "risk",
                "type": "choice",
                "choice": "low",
                "confidence": 0.99,
                "probabilities": [
                    {"value": label, "probability": 0.97 if label == "low" else 0.01}
                    for label in ("low", "medium", "high", "critical")
                ],
            },
            {
                "name": "decision",
                "type": "choice",
                "choice": "approve",
                "confidence": 0.99,
                "probabilities": [
                    {
                        "value": label,
                        "probability": 0.98 if label == "approve" else 0.01,
                    }
                    for label in ("approve", "deny", "escalate")
                ],
            },
        ],
    }
    native = DecisionsClient(
        DecisionsConfig(api_key="test-key", model="native"),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload)
        ),
    )
    evaluator, _, llm, audit, sessions = _setup()
    evaluator._decisions = native
    if cap:
        sessions.get.return_value["policy"] = {"maximum_outcome": cap}
    if precedent:
        audit.get_user_decisions.return_value = [
            {
                "tool": "write",
                "args_redacted": {"filePath": "docs/a.md", "content": "text"},
                "user_decision": "approve",
                "resolved_by": "user",
            }
        ]
    try:
        result = _call(evaluator)
    finally:
        native.close()
    assert result["evaluation_metadata"]["status"] == status
    assert result["evaluation_metadata"]["backend"] == "openai_decisions"
    assert result["decision"] == ("approve" if cap else "escalate")
    if cap:
        assert result["raw_decision"] == "escalate"
        assert result["effective_decision"] == "approve"
        assert result["outcome_override"] == "session_policy.maximum_outcome"
        assert audit.insert.call_args.kwargs["raw_decision"] == "escalate"
    llm.generate.assert_not_called()


@pytest.mark.parametrize("cap", [None, "approve"])
def test_confirmed_critical_escalation_still_raw_denial(cap):
    evaluator, decisions, llm, audit, sessions = _setup()
    decisions.evaluate.return_value = _result(risk="critical", decision="escalate")
    if cap:
        sessions.get.return_value["policy"] = {"maximum_outcome": cap}
    result = _call(evaluator)
    assert audit.insert.call_args.kwargs["raw_decision"] == "deny"
    assert result["decision"] == ("approve" if cap else "deny")
    if cap:
        assert result["raw_decision"] == "deny"
        assert result["effective_decision"] == "approve"
        assert result["outcome_override"] == "session_policy.maximum_outcome"
    llm.generate.assert_not_called()


def test_late_native_result_not_used_or_audited():
    evaluator, decisions, llm, audit, _ = _setup()
    evaluator._llm_timeout_ms = 10

    def late(**kwargs):
        time.sleep(0.02)
        return _result()

    decisions.evaluate.side_effect = late
    with pytest.raises(DecisionsTemporaryError):
        _call(evaluator)
    assert decisions.evaluate.call_args.kwargs["timeout_ms"] == 5
    llm.generate.assert_not_called()
    audit.insert.assert_not_called()
    assert evaluator._approved_paths == {}


def test_late_legacy_fallback_not_audited():
    evaluator, decisions, _, audit, _ = _setup()
    evaluator._llm_timeout_ms = 30
    decisions.evaluate.side_effect = DecisionsTemporaryError("offline")
    legacy = object.__new__(LLMClient)
    legacy._model = "legacy"
    legacy._temperature = 0.1
    legacy._reasoning_effort = None
    legacy._supports_structured = False
    legacy._param_fixes = {}
    legacy._transient_retries = 0
    legacy._client = MagicMock()
    legacy._client.with_options.return_value.chat.completions.create.side_effect = (
        lambda **kw: (
            time.sleep(0.04)
            or SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "aligned": True,
                                    "risk": "low",
                                    "reasoning": "ok",
                                    "decision": "approve",
                                }
                            )
                        )
                    )
                ]
            )
        )
    )
    evaluator._llm = legacy
    with pytest.raises(LLMTemporaryError):
        _call(evaluator)
    audit.insert.assert_not_called()
    assert evaluator._approved_paths == {}


def test_human_precedent_and_policy_cap_preserved():
    evaluator, decisions, _, audit, sessions = _setup()
    decisions.evaluate.return_value = _result(aligned=False, decision="escalate")
    audit.get_user_decisions.return_value = [
        {
            "tool": "write",
            "args_redacted": {"filePath": "docs/a.md", "content": "text"},
            "user_decision": "approve",
            "resolved_by": "user",
            "user_note": "Approved",
        }
    ]
    assert _call(evaluator)["decision"] == "approve"
    audit.get_user_decisions.return_value = []
    sessions.get.return_value["policy"] = {"maximum_outcome": "approve"}
    result = _call(evaluator)
    assert result["raw_decision"] == "escalate"
    assert result["decision"] == "approve"
    assert result["outcome_override"] == "session_policy.maximum_outcome"


def test_no_fallback_when_no_llm():
    evaluator, decisions, _, _, _ = _setup()
    evaluator._llm = None
    decisions.evaluate.side_effect = DecisionsTemporaryError("unavailable")
    with pytest.raises(DecisionsTemporaryError):
        _call(evaluator)
    assert decisions.evaluate.call_args.kwargs["timeout_ms"] == 3000


def test_deadline_disables_sdk_retries_and_rejects_late_response():
    llm = object.__new__(LLMClient)
    llm._model = "legacy"
    llm._temperature = 0.1
    llm._reasoning_effort = None
    llm._supports_structured = False
    llm._param_fixes = {}
    llm._transient_retries = 3
    llm._client = MagicMock()
    bounded = llm._client.with_options.return_value
    bounded.chat.completions.create.side_effect = lambda **kw: (
        time.sleep(0.02)
        or SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
        )
    )
    with pytest.raises(LLMTemporaryError, match="timed out"):
        llm.generate(
            [{"role": "user", "content": "test"}], deadline=time.monotonic() + 0.01
        )
    llm._client.with_options.assert_called_once_with(max_retries=0)
    assert bounded.chat.completions.create.call_count == 1
    assert bounded.chat.completions.create.call_args.kwargs["timeout"] <= 0.01


def test_deadline_prevents_second_structured_output_attempt():
    llm = object.__new__(LLMClient)
    llm._model = "legacy"
    llm._temperature = 0.1
    llm._reasoning_effort = None
    llm._supports_structured = None
    llm._param_fixes = {}
    llm._transient_retries = 0
    llm._client = MagicMock()
    bounded = llm._client.with_options.return_value

    def late_error(**kw):
        time.sleep(0.02)
        raise LLMTemporaryError("temporarily unavailable")

    bounded.chat.completions.create.side_effect = late_error
    with pytest.raises(LLMTemporaryError):
        llm.generate(
            [{"role": "user", "content": "test"}],
            json_schema={"name": "x"},
            deadline=time.monotonic() + 0.01,
        )
    assert bounded.chat.completions.create.call_count == 1


def _bounded_llm():
    llm = object.__new__(LLMClient)
    llm._model = "legacy"
    llm._temperature = 0.1
    llm._reasoning_effort = None
    llm._supports_structured = None
    llm._param_fixes = {}
    llm._transient_retries = 3
    llm._client = MagicMock()
    return llm, llm._client.with_options.return_value.chat.completions.create


def _bad_request(param, *, code="unsupported_parameter"):
    return BadRequestError(
        "unsupported",
        response=httpx.Response(
            400, request=httpx.Request("POST", "https://example.invalid")
        ),
        body={"param": param, "code": code},
    )


@pytest.mark.parametrize("param", ["max_tokens", "temperature"])
def test_bounded_parameter_adaptation_keeps_structured_mode(param):
    llm, create = _bounded_llm()
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            time.sleep(0.01)
            raise _bad_request(param)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
        )

    create.side_effect = request
    assert (
        llm.generate(
            [{"role": "user", "content": "test"}],
            json_schema={"name": "x"},
            deadline=time.monotonic() + 1,
        )
        == "{}"
    )
    assert len(calls) == 2
    assert calls[1]["timeout"] < calls[0]["timeout"]
    assert all(call["response_format"]["type"] == "json_schema" for call in calls)
    assert (
        ("max_completion_tokens" in calls[1])
        if param == "max_tokens"
        else ("temperature" not in calls[1])
    )
    assert llm._supports_structured is True
    assert llm._client.with_options.call_count == 2
    llm._client.with_options.assert_any_call(max_retries=0)


def test_bounded_adaptation_exhausted_before_second_attempt():
    llm, create = _bounded_llm()

    def late_error(**kwargs):
        time.sleep(0.02)
        raise _bad_request("max_tokens")

    create.side_effect = late_error
    with pytest.raises(LLMTemporaryError, match="timed out"):
        llm.generate(
            [{"role": "user", "content": "test"}],
            json_schema={"name": "x"},
            deadline=time.monotonic() + 0.01,
        )
    assert create.call_count == 1
    assert llm._supports_structured is None
    assert llm._param_fixes == {}


@pytest.mark.parametrize("expired", [False, True])
def test_bounded_structured_output_bad_request_falls_back_only_with_time(expired):
    llm, create = _bounded_llm()
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            if expired:
                time.sleep(0.02)
            raise _bad_request("response_format", code="not_supported")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
        )

    create.side_effect = request
    if expired:
        with pytest.raises(LLMTemporaryError, match="timed out"):
            llm.generate(
                [{"role": "user", "content": "test"}],
                json_schema={"name": "x"},
                deadline=time.monotonic() + 0.01,
            )
        assert len(calls) == 1
        assert llm._supports_structured is None
    else:
        assert (
            llm.generate(
                [{"role": "user", "content": "test"}],
                json_schema={"name": "x"},
                deadline=time.monotonic() + 1,
            )
            == "{}"
        )
        assert len(calls) == 2
        assert calls[1]["response_format"]["type"] == "json_object"
        assert calls[1]["timeout"] < calls[0]["timeout"]
        assert llm._supports_structured is False
    assert llm._client.with_options.call_count == len(calls)
