"""Infrastructure failures remain blocked without becoming approval prompts."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from intaris.audit import AuditStore
from intaris.config import JudgeConfig
from intaris.judge import JudgeReviewer
from intaris.model_retry import evaluation_deadline
from tests.test_actionable_review import reply
from tests.test_hybrid_api import api as review_api
from tests.test_policy_evidence_runtime import response as assessment_response

api = review_api


@pytest.mark.parametrize("transient", [True, False])
def test_initial_assessment_failure_never_requests_human_permission(
    api, monkeypatch, transient
):
    import intaris.api.evaluate as endpoint
    import intaris.server as server

    client, version, _ = api
    if version != "hybrid_v1":
        pytest.skip("Hybrid review infrastructure contract")
    headers = {
        "Authorization": "Bearer hybrid-test-key",
        "X-User-Id": "owner",
        "X-Agent-Id": "test-reviewer",
    }
    assert (
        client.post(
            "/api/v1/intention",
            headers=headers,
            json={
                "session_id": "initial-outage",
                "intention": "Inspect the local parser.",
            },
        ).status_code
        == 200
    )
    server._get_evaluator()._llm.generate.side_effect = (
        httpx.ReadTimeout("Assessment transport timed out")
        if transient
        else ValueError("Malformed assessment")
    )
    emit = AsyncMock()
    monkeypatch.setattr(endpoint, "_emit_unresolved_escalation", emit)
    result = client.post(
        "/api/v1/evaluate",
        headers=headers,
        json={
            "session_id": "initial-outage",
            "tool": "write",
            "args": {"file_path": "/repo/parser.py"},
        },
    )
    assert result.status_code == 503, result.text
    assert result.json()["detail"]["code"] == "assessment_failed"
    assert result.json()["detail"]["retryable"] is transient
    emit.assert_not_awaited()


@pytest.mark.parametrize(
    "failure", ["assessment_failed", "authorization", "timeout", None]
)
def test_failed_review_is_retryable_not_human_permission(api, monkeypatch, failure):
    import intaris.api.evaluate as endpoint
    import intaris.server as server

    client, version, _ = api
    if version != "hybrid_v1":
        pytest.skip("Hybrid review infrastructure contract")
    headers = {
        "Authorization": "Bearer hybrid-test-key",
        "X-User-Id": "owner",
        "X-Agent-Id": "test-reviewer",
    }
    server._get_evaluator()._llm.generate.side_effect = lambda *args, **kw: (
        assessment_response(kw, high=True)
    )
    posted = client.post(
        "/api/v1/intention",
        headers=headers,
        json={"session_id": "review", "intention": "Implement the parser."},
    )
    assert posted.status_code == 200, posted.text
    stage = {"stage": "judge", "response_origin": "model"}
    if failure == "assessment_failed":
        stage["response_origin"] = failure
    if failure == "authorization":
        stage["authorization_check"] = {
            "outcome": "unavailable",
            "error": "LLMTemporaryError",
        }
    metadata = {"evaluation_policy_version": "hybrid_v1", "stages": [stage]}
    reviewer = Mock(is_enabled=True)
    reviewer.review_for_evaluate = AsyncMock(
        return_value=SimpleNamespace(
            decision="escalate",
            reasoning="Review did not establish permission.",
            risk="high",
            record={"evaluation_metadata": metadata},
            notification_event_type=(
                "judge_error" if failure == "timeout" else "judge_deferral"
            ),
        )
    )
    emit = AsyncMock()
    monkeypatch.setattr(endpoint, "_emit_unresolved_escalation", emit)
    for route in client.app.routes:
        if getattr(route, "path", None) == "/api/v1":
            route.app.state.judge_reviewer = reviewer
    result = client.post(
        "/api/v1/evaluate",
        headers=headers,
        json={
            "session_id": "review",
            "tool": "write",
            "args": {"file_path": "/repo/parser.py"},
        },
    )
    if failure:
        assert result.status_code == 503, result.text
        assert result.json()["detail"]["code"] == "review_unavailable"
        emit.assert_not_awaited()
    else:
        assert result.status_code == 200, result.text
        assert result.json()["decision"] == "escalate"
        emit.assert_awaited_once()


@pytest.mark.asyncio
async def test_hybrid_review_encloses_multiple_model_stages_but_honors_request_deadline():
    record = {
        "call_id": "review-timeout",
        "decision": "escalate",
        "risk": "high",
        "evaluation_metadata": {"evaluation_policy_version": "hybrid_v1"},
    }
    reviewer = JudgeReviewer(
        llm=SimpleNamespace(_timeout_ms=20),
        config=JudgeConfig(mode="auto"),
        audit_store=Mock(
            get_by_call_id=Mock(return_value=record),
            is_obsolete_escalation=Mock(return_value=False),
        ),
        session_store=Mock(),
        evaluator=Mock(),
    )

    async def sequential(**kwargs):
        for _ in range(4):
            await asyncio.sleep(0.025)

    reviewer._do_review = AsyncMock(side_effect=sequential)
    with evaluation_deadline(0.2):
        outcome = await reviewer.review_for_evaluate(
            call_id="review-timeout", user_id="owner", session_id="review"
        )
    assert outcome.notification_event_type != "judge_error"
    with evaluation_deadline(0.01):
        outcome = await reviewer.review_for_evaluate(
            call_id="review-timeout", user_id="owner", session_id="review"
        )
    assert outcome.notification_event_type == "judge_error"


def test_real_unattended_reviewer_keeps_authorization_outage_unresolved(
    api, monkeypatch
):
    import intaris.api.evaluate as endpoint
    import intaris.judge_simulator as simulator
    import intaris.server as server

    client, version, _ = api
    if version != "hybrid_v1":
        pytest.skip("Hybrid review infrastructure contract")
    headers = {
        "Authorization": "Bearer hybrid-test-key",
        "X-User-Id": "owner",
        "X-Agent-Id": "test-reviewer",
    }
    evaluator = server._get_evaluator()
    evaluator._llm.generate.side_effect = lambda *args, **kw: assessment_response(
        kw, high=True
    )
    assert (
        client.post(
            "/api/v1/intention",
            headers=headers,
            json={
                "session_id": "unattended",
                "intention": "Implement the local parser.",
            },
        ).status_code
        == 200
    )
    audit = AuditStore(server._get_db())
    reviewer = JudgeReviewer(
        llm=Mock(_timeout_ms=4000),
        config=JudgeConfig(mode="auto"),
        audit_store=audit,
        session_store=evaluator._sessions,
        evaluator=evaluator,
    )
    monkeypatch.setattr(
        simulator,
        "classify",
        lambda *args, **kwargs: (
            simulator.Answer.model_validate(
                reply(cites=[], user_would="ask", basis="review_unavailable")
            ),
            {"response_origin": "assessment_failed"},
        ),
    )
    emit = AsyncMock()
    monkeypatch.setattr(endpoint, "_emit_unresolved_escalation", emit)
    for route in client.app.routes:
        if getattr(route, "path", None) == "/api/v1":
            route.app.state.judge_reviewer = reviewer
    result = client.post(
        "/api/v1/evaluate",
        headers=headers,
        json={
            "session_id": "unattended",
            "tool": "write",
            "args": {"file_path": "/repo/parser.py"},
        },
    )
    assert result.status_code == 503, result.text
    assert result.json()["detail"]["code"] == "review_unavailable"
    emit.assert_not_awaited()
    with server._get_db().cursor() as cur:
        cur.execute(
            "SELECT call_id FROM audit_log WHERE session_id = ? "
            "AND record_type = 'tool_call'",
            ("unattended",),
        )
        call_id = cur.fetchone()["call_id"]
    record = audit.get_by_call_id(call_id, user_id="owner")
    assert record["decision"] == "escalate"
    assert record["resolved_by"] is None
    assert record["user_decision"] is None


@pytest.mark.parametrize(
    "first_vote", ["ask", "approve", "refuse", "unverified_refuse", "unverified_ask"]
)
def test_real_balanced_review_distinguishes_task_support_from_human_choice(
    api, monkeypatch, first_vote
):
    import intaris.api.evaluate as endpoint
    import intaris.server as server

    client, version, _ = api
    if version != "hybrid_v1":
        pytest.skip("Hybrid review contract")
    headers = {
        "Authorization": "Bearer hybrid-test-key",
        "X-User-Id": "owner",
        "X-Agent-Id": "test-reviewer",
    }
    task = "Implement and test the parser, preserving generated test evidence."
    if first_vote == "refuse":
        task = "Implement the parser. Do not write test-evidence.json."
    assert (
        client.post(
            "/api/v1/intention",
            headers=headers,
            json={
                "session_id": "routine-review",
                "intention": task,
                "original_assignment": task,
            },
        ).status_code
        == 200
    )
    evaluator = server._get_evaluator()
    evaluator._llm.generate.side_effect = lambda *args, **kw: assessment_response(
        kw, high=True
    )

    llm = Mock(_timeout_ms=4000)
    unverified = first_vote.startswith("unverified_")
    llm.generate.return_value = json.dumps(
        reply(
            user_would=first_vote.removeprefix("unverified_"),
            basis="irreversible_unclear"
            if first_vote == "ask"
            else "restriction_conflict"
            if first_vote == "refuse" or unverified
            else "routine_within_task",
            risk="low",
            question="Which recipient should receive this?"
            if first_vote in {"ask", "unverified_ask"}
            else "",
            explanation="A recipient choice remains."
            if first_vote == "ask"
            else "The user prohibited this file."
            if first_vote == "refuse"
            else "This is routine task-related evidence handling.",
            cites=[1] if first_vote == "refuse" or unverified else [],
            restriction_quote=(
                "Do not write test-evidence.json."
                if first_vote == "refuse"
                else "A previous guard blocked this action."
                if unverified
                else ""
            ),
        )
    )
    audit = AuditStore(server._get_db())
    reviewer = JudgeReviewer(
        llm=llm,
        config=JudgeConfig(mode="auto" if unverified else "advisory"),
        audit_store=audit,
        session_store=evaluator._sessions,
        evaluator=evaluator,
    )

    emit = AsyncMock()
    monkeypatch.setattr(endpoint, "_emit_unresolved_escalation", emit)
    for route in client.app.routes:
        if getattr(route, "path", None) == "/api/v1":
            route.app.state.judge_reviewer = reviewer
    result = client.post(
        "/api/v1/evaluate",
        headers=headers,
        json={
            "session_id": "routine-review",
            "tool": "write",
            "args": {"file_path": "/repo/test-evidence.json"},
        },
    )
    assert result.status_code == 200, result.text
    assert (
        result.json()["decision"]
        == {
            "approve": "approve",
            "ask": "escalate",
            "refuse": "deny",
            "unverified_refuse": "escalate",
            "unverified_ask": "escalate",
        }[first_vote]
    )
    # Citable original instructions retain the restriction option even if its
    # quote repair fails; validation failure is not a permission judgment.
    assert llm.generate.call_count == (2 if unverified else 1)
    if first_vote == "approve":
        emit.assert_not_awaited()
        record = audit.get_by_call_id(result.json()["call_id"], user_id="owner")
        metadata = record["evaluation_metadata"]
        assert metadata["policy_evidence"]["review"]["user_would"] == first_vote
        assert "authorization_check" not in metadata["stages"][-1]
    elif first_vote == "ask" or unverified:
        emit.assert_awaited_once()
        if unverified:
            record = audit.get_by_call_id(result.json()["call_id"], user_id="owner")
            assert record["user_decision"] is None
            assert record["resolved_by"] is None
            assert result.json()["risk"] == record["risk"]
            assert "review-validation failure" in result.json()["reasoning"]
            assert "withheld permission" in result.json()["reasoning"]
            assert (
                "Permission for this action has not been verified"
                not in result.json()["reasoning"]
            )
            stage = record["evaluation_metadata"]["stages"][-1]
            assert stage["classification"]["basis"] == "review_unavailable"
    else:
        emit.assert_not_awaited()
