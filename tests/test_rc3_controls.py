"""RC3 static read routing, advisory context and reversible operator routing."""

from unittest.mock import Mock

import pytest

from intaris.classifier import observe_hybrid, verified_read_only_bash
from intaris.policy_activation import HybridError
from tests.test_hybrid_races import race_store as race_store
from tests.test_workflow_scope import evaluate, install
from tests.test_workflow_scope import workflow as workflow


@pytest.mark.parametrize("profile", ["balanced", "bounded_autonomy"])
@pytest.mark.parametrize(
    "tool",
    [
        "get_current_datetime",
        "skill_load",
        "skill",
        "mcp_office365-examplecorp__list-chats",
    ],
)
def test_unattended_reads_ignore_unrelated_advisory(race_store, profile, tool):
    sessions, _, llm, evaluator, sid = race_store
    sessions.update_session(
        sid,
        user_id="race-owner",
        policy={
            "profile": profile,
            "interaction_mode": "none",
            "deny_policies": [
                "Assistant must not send email without consent.",
                "Assistant must not make commitments.",
            ],
        },
    )
    llm.generate.side_effect = AssertionError("No model for static read")
    result = evaluator.evaluate(
        user_id="race-owner",
        session_id=sid,
        agent_id=None,
        tool=tool,
        args={},
        judge_unattended=True,
    )
    assert result["decision"] == "approve"
    assert (
        result["evaluation_metadata"]["policy_result"]["permission_basis"]
        == "read_only_static"
    )
    assert not result["evaluation_metadata"]["judge_required"]
    llm.generate.assert_not_called()


def test_git_expansions_and_option_named_paths():
    assert verified_read_only_bash(
        {
            "command": "git diff --no-ext-diff --no-textconv -- private/*",
        }
    )
    facts = observe_hybrid(
        "bash",
        {"command": "git diff --no-ext-diff --no-textconv -- --stat"},
        working_directory="/repo",
        policy={"deny_paths": ["/repo/--stat"]},
    )
    assert facts["configured_deny"]


@pytest.mark.parametrize("idle", [False, True])
def test_force_legacy_bypasses_hybrid_gates_reversibly(workflow, monkeypatch, idle):
    sessions, audit, llm, evaluator, sid, _ = workflow
    install(workflow)
    sessions.set_alignment_state(sid, user_id="race-owner", state="pending")
    if idle:
        sessions.update_status(sid, "idle", user_id="race-owner")
    original = sessions.get(sid, user_id="race-owner")
    evaluator._alignment_barrier = Mock()
    evaluator._alignment_barrier.is_misaligned.side_effect = AssertionError(
        "No barrier"
    )
    evaluator._task_context = Mock()
    evaluator._task_context.capture.side_effect = AssertionError("No extraction")
    llm.generate.side_effect = AssertionError("Read must be static")
    monkeypatch.setenv("EVALUATION_POLICY_FORCE", "legacy")
    result = evaluate(
        workflow,
        evaluation_policy_version="hybrid_v1",
        expected_policy_revision=-1,
        expected_policy_hash="old-policy",
    )
    assert result["decision"] == "approve"
    metadata = result.get("evaluation_metadata") or {}
    assert metadata.get("evaluation_policy_version") != "hybrid_v1"
    saved = sessions.get(sid, user_id="race-owner")
    assert saved["policy"] == original["policy"]
    assert saved["state_revision"] == original["state_revision"] + int(idle)
    assert saved["status"] == "active"
    assert saved["alignment_state"] == "pending"
    evaluator._alignment_barrier.is_misaligned.assert_not_called()
    assert audit.get_by_call_id(result["call_id"], user_id="race-owner")
    monkeypatch.delenv("EVALUATION_POLICY_FORCE")
    evaluator._alignment_barrier = None
    with pytest.raises(HybridError) as error:
        evaluate(workflow, expected_policy_revision=-1)
    assert error.value.code == "stale_policy_revision"


def test_force_legacy_receipt_and_ceiling(race_store, monkeypatch):
    sessions, audit, llm, evaluator, sid = race_store
    llm.generate.return_value = (
        '{"aligned":true,"risk":"low","decision":"approve","reasoning":"test"}'
    )
    monkeypatch.setenv("EVALUATION_POLICY_FORCE", "legacy")
    request = dict(
        user_id="race-owner",
        session_id=sid,
        agent_id=None,
        tool="read",
        args={"file_path": "/repo/README.md"},
        minimum_outcome="escalate",
    )
    pending = evaluator.evaluate(**request)
    assert pending["decision"] == "escalate"
    audit.resolve_escalation(pending["call_id"], "approve", user_id="race-owner")
    result = evaluator.evaluate(**request, approval_call_id=pending["call_id"])
    assert result["decision"] == "approve"
    blocked = evaluator.evaluate(
        user_id="race-owner",
        session_id=sid,
        agent_id=None,
        tool="write",
        args={"file_path": "/repo/new"},
        context={"chat_mode": "plan", "read_only_required": True},
    )
    assert blocked["decision"] == "deny"
    with pytest.raises(HybridError) as error:
        audit.resolve_escalation(blocked["call_id"], "approve", user_id="race-owner")
    assert error.value.code == "read_only_ceiling"


@pytest.mark.parametrize("profile", ["balanced", "bounded_autonomy"])
@pytest.mark.parametrize(
    "clause",
    [
        "Do not access files without approval",
        "Do not read customer records",
        "Never read /repo/README.md",
    ],
)
def test_advisory_clauses_never_route_static_reads(race_store, profile, clause):
    sessions, _, llm, evaluator, sid = race_store
    sessions.update_session(
        sid,
        user_id="race-owner",
        policy={
            "profile": profile,
            "deny_policies": [clause],
        },
    )
    llm.generate.side_effect = AssertionError("No model for static read")
    result = evaluator.evaluate(
        user_id="race-owner",
        session_id=sid,
        agent_id=None,
        tool="read",
        args={"file_path": "/repo/README.md"},
    )
    llm.generate.assert_not_called()
    assert result["decision"] == "approve"
    assert not result["evaluation_metadata"]["judge_required"]
    assert result["evaluation_metadata"]["outstanding_obligations"] == []


def test_colon_named_git_operand_denied_through_evaluator(race_store):
    sessions, _, _, evaluator, sid = race_store
    sessions.update_session(
        sid,
        user_id="race-owner",
        policy={
            "deny_paths": ["/repo/reports:annual.txt"],
        },
    )
    args = {"command": "git diff --no-ext-diff --no-textconv -- reports:annual.txt"}
    facts = observe_hybrid(
        "bash",
        args,
        policy={"deny_paths": ["/repo/reports:annual.txt"]},
        working_directory="/repo",
    )
    assert "/repo/reports:annual.txt" in facts["paths"]
    assert facts["configured_deny"]
    result = evaluator.evaluate(
        user_id="race-owner",
        session_id=sid,
        agent_id=None,
        tool="bash",
        args=args,
        context={"working_directory": "/repo"},
    )
    assert result["decision"] == "deny"
