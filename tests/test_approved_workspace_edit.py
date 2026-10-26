"""Approved workspace edits: follow-up edits in an approved write scope skip
the evaluator model until new user evidence or a policy change."""

import json
import os
import uuid
from unittest.mock import Mock

import pytest

from intaris.audit import AuditStore
from intaris.config import DBConfig
from intaris.db import Database
from intaris.evaluator import Evaluator
from intaris.session import SessionStore
from tests.test_policy_evidence_runtime import response

WD = "/repo"


def _answer(outcome):
    def generate(*args, **kwargs):
        properties = kwargs["json_schema"]["schema"]["properties"]
        if "aligned" in properties:  # legacy safety evaluation
            return json.dumps(
                {
                    "approve": {"aligned": True, "risk": "low", "decision": "approve"},
                    "escalate": {
                        "aligned": True,
                        "risk": "high",
                        "decision": "escalate",
                    },
                    "deny": {"aligned": False, "risk": "high", "decision": "deny"},
                }[outcome]
                | {"reasoning": "fixture"}
            )
        result = json.loads(response(kwargs, high=outcome != "approve"))
        references = properties.get("restriction_reference", {}).get("enum", [])
        if "latest_user_message" in references:
            # Re-admission must assess the new user source, not merely repeat
            # the earlier session-intention citation after its scope was reset.
            result["restriction_reference"] = "latest_user_message"
        return json.dumps(result)

    return generate


@pytest.fixture(
    params=[
        ("hybrid_v1", "sqlite"),
        ("legacy", "sqlite"),
        ("hybrid_v1", "postgresql"),
        ("legacy", "postgresql"),
    ],
    ids=["hybrid-sqlite", "legacy-sqlite", "hybrid-pg", "legacy-pg"],
)
def env(request, tmp_path, monkeypatch):
    version, backend = request.param
    monkeypatch.setenv("EVALUATION_POLICY_VERSION", version)
    url = os.environ.get("TEST_HYBRID_POSTGRES_URL", "")
    if backend == "postgresql" and not url:
        pytest.skip("Isolated PostgreSQL URL not configured")
    db = Database(
        DBConfig(backend=backend, path=str(tmp_path / "state.db"), database_url=url)
    )
    sessions, audit = SessionStore(db), AuditStore(db)
    llm = Mock()
    llm.generate.side_effect = _answer("approve")
    evaluator = Evaluator(llm=llm, session_store=sessions, audit_store=audit)
    # Tenant per test: escalation retries are user-scoped and the PostgreSQL
    # database is shared across tests.
    session_id = str(uuid.uuid4())
    sessions.create(
        user_id=session_id,
        session_id=session_id,
        intention="Implement the local parser",
        details={"working_directory": WD},
    )
    yield sessions, audit, llm, evaluator, session_id, version
    db.close()


def edit(evaluator, session_id, path, *, tool="edit", args=None, **options):
    return evaluator.evaluate(
        user_id=session_id,
        session_id=session_id,
        agent_id=None,
        tool=tool,
        args=args or {"file_path": path, "old_string": "a", "new_string": "b"},
        **options,
    )


def assert_model(llm, result, calls):
    assert llm.generate.call_count == calls
    assert not result["reasoning"].startswith("approved_workspace_edit")


def assert_skipped(llm, result, calls, scope=WD):
    assert llm.generate.call_count == calls
    assert result["decision"] == "approve"
    assert result["path"] == "fast"
    assert result["reasoning"].startswith("approved_workspace_edit")
    assert scope in result["reasoning"]
    assert result["evaluation_metadata"]["approved_workspace_scopes"] == [scope]


def test_follow_up_edit_skips_model_and_is_audited(env):
    _, audit, llm, evaluator, sid, version = env
    first = edit(evaluator, sid, f"{WD}/parser.py")
    assert first["decision"] == "approve"
    assert_model(llm, first, 1)

    second = edit(evaluator, sid, "pkg/lexer.py", tool="write")
    assert_skipped(llm, second, 1)
    record = audit.get_by_call_id(second["call_id"], user_id=sid)
    assert record["decision"] == "approve"
    assert record["reasoning"].startswith("approved_workspace_edit")
    assert record["evaluation_metadata"]["approved_workspace_scopes"] == [WD]
    if version == "hybrid_v1":
        policy_result = record["evaluation_metadata"]["policy_result"]
        assert policy_result["permission_basis"] == "approved_workspace_edit"
        assert not record["evaluation_metadata"]["stages"]
    else:
        assert (
            record["evaluation_metadata"]["permission_basis"]
            == "approved_workspace_edit"
        )


def user_message(audit, sid, text):
    audit.insert(
        call_id=str(uuid.uuid4()),
        user_id=sid,
        session_id=sid,
        agent_id=None,
        tool=None,
        args_redacted=None,
        classification=None,
        evaluation_path="reasoning",
        decision="approve",
        risk=None,
        reasoning=None,
        latency_ms=0,
        record_type="reasoning",
        content=f"User message: {text}",
    )


def test_new_user_message_resets_scope(env):
    _, audit, llm, evaluator, sid, version = env
    edit(evaluator, sid, f"{WD}/a.py")
    assert_skipped(llm, edit(evaluator, sid, f"{WD}/b.py"), 1)
    user_message(audit, sid, "now refactor the lexer")
    renewed = edit(evaluator, sid, f"{WD}/c.py")
    assert_model(llm, renewed, 2)
    if version == "hybrid_v1":
        assert renewed["decision"] == "approve"
    following = edit(evaluator, sid, f"{WD}/d.py")
    if version == "hybrid_v1":
        # A fresh assessment for the new revision may reopen the cache.
        assert_skipped(llm, following, 2)
        assert following["decision"] == "approve"
    else:
        assert_skipped(llm, following, 2)


def test_policy_change_resets_scope(env):
    sessions, _, llm, evaluator, sid, _ = env
    edit(evaluator, sid, f"{WD}/a.py")
    assert_skipped(llm, edit(evaluator, sid, f"{WD}/b.py"), 1)
    sessions.update_session(
        sid,
        user_id=sid,
        policy={"profile": "balanced", "deny_tools": ["forbidden_external_publish"]},
    )
    assert_model(llm, edit(evaluator, sid, f"{WD}/c.py"), 2)


def test_edit_outside_scope_goes_to_model(env):
    _, _, llm, evaluator, sid, _ = env
    edit(evaluator, sid, f"{WD}/a.py")
    assert_model(llm, edit(evaluator, sid, "/srv/other/x.py"), 2)
    # An approved outside edit opens only its parent directory.
    assert_skipped(llm, edit(evaluator, sid, "/srv/other/y.py"), 2, "/srv/other")
    assert_model(llm, edit(evaluator, sid, "/srv/z.py"), 3)


@pytest.mark.parametrize(
    "path",
    [
        f"{WD}/.env",
        f"{WD}/.git/config",
        f"{WD}/.github/workflows/ci.yml",
        f"{WD}/.gitlab-ci.yml",
        f"{WD}/Jenkinsfile",
        f"{WD}/deploy/k8s/app.yaml",
    ],
)
def test_protected_targets_go_to_model(env, path):
    _, _, llm, evaluator, sid, _ = env
    edit(evaluator, sid, f"{WD}/a.py")
    assert_model(llm, edit(evaluator, sid, path), 2)


def test_strict_profile_always_uses_model(env):
    sessions, _, llm, evaluator, sid, _ = env
    sessions.update_session(sid, user_id=sid, policy={"profile": "strict"})
    for index, name in enumerate(("a.py", "b.py", "c.py"), start=1):
        assert_model(llm, edit(evaluator, sid, f"{WD}/{name}"), index)


def test_explicit_read_only_ceiling_is_never_skipped(env):
    _, _, llm, evaluator, sid, _ = env
    edit(evaluator, sid, f"{WD}/a.py")
    assert_skipped(llm, edit(evaluator, sid, f"{WD}/b.py"), 1)
    planned = edit(
        evaluator,
        sid,
        f"{WD}/c.py",
        context={"chat_mode": "plan", "read_only_required": True},
    )
    assert planned["decision"] != "approve"
    assert not planned["reasoning"].startswith("approved_workspace_edit")


def test_minimum_outcome_floor_is_never_skipped(env):
    _, _, llm, evaluator, sid, _ = env
    edit(evaluator, sid, f"{WD}/a.py")
    floored = edit(evaluator, sid, f"{WD}/b.py", minimum_outcome="escalate")
    assert floored["decision"] != "approve"
    assert not floored["reasoning"].startswith("approved_workspace_edit")


def test_apply_patch_with_one_target_outside_scope_goes_to_model(env):
    _, _, llm, evaluator, sid, _ = env
    edit(evaluator, sid, f"{WD}/a.py")
    inside = (
        "*** Begin Patch\n*** Update File: b.py\n@@\n-a\n+b\n"
        "*** Add File: pkg/c.py\n+c\n*** End Patch\n"
    )
    assert_skipped(
        llm,
        edit(evaluator, sid, None, tool="apply_patch", args={"patchText": inside}),
        1,
    )
    mixed = (
        "*** Begin Patch\n*** Update File: b.py\n@@\n-a\n+b\n"
        "*** Update File: /etc/hosts\n@@\n-a\n+b\n*** End Patch\n"
    )
    result = edit(evaluator, sid, None, tool="apply_patch", args={"patchText": mixed})
    assert_model(llm, result, 2)


def test_opencode_patch_tool_shares_apply_patch_scopes(env):
    _, _, llm, evaluator, sid, _ = env
    edit(evaluator, sid, f"{WD}/a.py")
    inside = "*** Begin Patch\n*** Update File: b.py\n@@\n-a\n+b\n*** End Patch\n"
    assert_skipped(
        llm, edit(evaluator, sid, None, tool="patch", args={"patchText": inside}), 1
    )
    outside = (
        "*** Begin Patch\n*** Update File: /etc/hosts\n@@\n-a\n+b\n*** End Patch\n"
    )
    result = edit(evaluator, sid, None, tool="patch", args={"patchText": outside})
    assert_model(llm, result, 2)
    deleting = "*** Begin Patch\n*** Delete File: b.py\n*** End Patch\n"
    result = edit(evaluator, sid, None, tool="patch", args={"patchText": deleting})
    assert_model(llm, result, 3)


@pytest.mark.parametrize(
    "patch",
    [
        "*** Begin Patch\n*** Delete File: b.py\n*** End Patch\n",
        "--- a/b.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-a\n",
    ],
)
def test_deleting_patch_goes_to_model(env, patch):
    _, _, llm, evaluator, sid, _ = env
    edit(evaluator, sid, f"{WD}/a.py")
    result = edit(evaluator, sid, None, tool="apply_patch", args={"patchText": patch})
    assert_model(llm, result, 2)


def test_concurrent_user_message_rejects_cached_approval(env, monkeypatch):
    from intaris.policy_activation import HybridError

    _, audit, llm, evaluator, sid, _ = env
    edit(evaluator, sid, f"{WD}/a.py")
    original = evaluator._approved_workspace_scopes

    def match_then_message(**kwargs):
        scopes = original(**kwargs)
        user_message(audit, sid, "stop editing")
        return scopes

    monkeypatch.setattr(evaluator, "_approved_workspace_scopes", match_then_message)
    with pytest.raises(HybridError) as error:
        edit(evaluator, sid, f"{WD}/b.py")
    assert error.value.code == "stale_admission"
    assert llm.generate.call_count == 1


def test_bash_is_not_covered(env):
    _, _, llm, evaluator, sid, _ = env
    edit(evaluator, sid, f"{WD}/a.py")
    result = edit(
        evaluator, sid, None, tool="bash", args={"command": f"touch {WD}/b.py"}
    )
    assert_model(llm, result, 2)


@pytest.mark.parametrize("outcome", ["escalate", "deny"])
def test_escalated_or_denied_first_edit_records_no_scope(env, outcome):
    _, _, llm, evaluator, sid, version = env
    if outcome == "deny" and version == "hybrid_v1":
        pytest.skip("The hybrid L1 assessment does not deny a routine edit")
    llm.generate.side_effect = _answer(outcome)
    first = edit(evaluator, sid, f"{WD}/a.py")
    assert first["decision"] == outcome
    llm.generate.side_effect = _answer("approve")
    assert_model(llm, edit(evaluator, sid, f"{WD}/b.py"), 2)


def test_human_approval_opens_scope_and_human_denial_wins(env):
    _, audit, llm, evaluator, sid, version = env
    llm.generate.side_effect = _answer("escalate")
    first = edit(evaluator, sid, f"{WD}/a.py")
    assert first["decision"] == "escalate"
    record = audit.resolve_escalation(
        call_id=first["call_id"],
        user_decision="approve",
        user_id=sid,
        resolved_by="user",
    )
    evaluator.learn_from_approved_escalation(record)
    llm.generate.side_effect = _answer("approve")
    assert_skipped(llm, edit(evaluator, sid, f"{WD}/b.py"), 1)

    # A human denial of a target keeps that target with the model, even
    # after a later approval reopened the scope for the new revision.
    user_message(audit, sid, "also touch the plan")
    llm.generate.side_effect = _answer("escalate")
    denied = edit(evaluator, sid, f"{WD}/secret_plan.py")
    audit.resolve_escalation(
        call_id=denied["call_id"],
        user_decision="deny",
        user_id=sid,
        resolved_by="user",
    )
    llm.generate.side_effect = _answer("approve")
    renewed = edit(evaluator, sid, f"{WD}/c.py")
    if version == "hybrid_v1":
        assert renewed["decision"] == "approve"
    following = edit(evaluator, sid, f"{WD}/d.py")
    if version == "hybrid_v1":
        assert_skipped(llm, following, 3)
        assert following["decision"] == "approve"
    else:
        assert_skipped(llm, following, 3)
    retry = edit(
        evaluator,
        sid,
        None,
        args={
            "file_path": f"{WD}/secret_plan.py",
            "old_string": "x",
            "new_string": "y",
        },
    )
    assert_model(llm, retry, 4)


def test_human_denial_keeps_call_scoped_working_directory(env):
    _, audit, llm, evaluator, sid, _ = env
    call_dir = {"working_directory": "/srv/app"}
    llm.generate.side_effect = _answer("escalate")
    denied = edit(evaluator, sid, "plan.py", context=call_dir)
    audit.resolve_escalation(
        call_id=denied["call_id"],
        user_decision="deny",
        user_id=sid,
        resolved_by="user",
    )
    llm.generate.side_effect = _answer("approve")
    edit(evaluator, sid, "a.py", context=call_dir)
    assert_skipped(llm, edit(evaluator, sid, "b.py", context=call_dir), 2, "/srv/app")
    retry = edit(
        evaluator,
        sid,
        None,
        args={"file_path": "/srv/app/plan.py", "old_string": "x", "new_string": "y"},
        context=call_dir,
    )
    assert_model(llm, retry, 3)


def test_denied_deletion_patch_keeps_deleted_path_with_model(env):
    _, audit, llm, evaluator, sid, _ = env
    llm.generate.side_effect = _answer("escalate")
    denied = edit(
        evaluator,
        sid,
        None,
        tool="apply_patch",
        args={"patchText": "*** Begin Patch\n*** Delete File: b.py\n*** End Patch\n"},
    )
    assert denied["decision"] in {"escalate", "deny"}
    audit.resolve_escalation(
        call_id=denied["call_id"],
        user_decision="deny",
        user_id=sid,
        resolved_by="user",
    )
    llm.generate.side_effect = _answer("approve")
    edit(evaluator, sid, f"{WD}/a.py")
    assert_skipped(llm, edit(evaluator, sid, f"{WD}/c.py"), 2)
    assert_model(llm, edit(evaluator, sid, f"{WD}/b.py"), 3)


def _escalate_and_approve(audit, llm, evaluator, sid):
    llm.generate.side_effect = _answer("escalate")
    first = edit(evaluator, sid, f"{WD}/a.py")
    record = audit.resolve_escalation(
        call_id=first["call_id"],
        user_decision="approve",
        user_id=sid,
        resolved_by="user",
    )
    llm.generate.side_effect = _answer("approve")
    return record


def test_approval_learned_after_new_user_evidence_opens_no_scope(env):
    _, audit, llm, evaluator, sid, _ = env
    record = _escalate_and_approve(audit, llm, evaluator, sid)
    user_message(audit, sid, "actually, stop and explain first")
    evaluator.learn_from_approved_escalation(record)
    assert_model(llm, edit(evaluator, sid, f"{WD}/b.py"), 2)


def test_legacy_approval_without_resolution_revision_opens_no_scope(env):
    _, audit, llm, evaluator, sid, version = env
    if version == "hybrid_v1":
        pytest.skip("Hybrid receipts carry their revision in evaluation metadata")
    record = _escalate_and_approve(audit, llm, evaluator, sid)
    record.pop("resolution_revision")
    evaluator.learn_from_approved_escalation(record)
    assert_model(llm, edit(evaluator, sid, f"{WD}/b.py"), 2)


def test_resolve_with_side_effects_carries_resolution_revision(env):
    import asyncio

    from intaris.judge import resolve_with_side_effects

    _, audit, llm, evaluator, sid, _ = env
    llm.generate.side_effect = _answer("escalate")
    first = edit(evaluator, sid, f"{WD}/a.py")
    asyncio.run(
        resolve_with_side_effects(
            call_id=first["call_id"],
            user_id=sid,
            user_decision="approve",
            user_note=None,
            resolved_by="user",
            audit_store=audit,
            evaluator=evaluator,
        )
    )
    llm.generate.side_effect = _answer("approve")
    assert_skipped(llm, edit(evaluator, sid, f"{WD}/b.py"), 1)
