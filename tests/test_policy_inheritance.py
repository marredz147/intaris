"""Delegation is not an independent policy or approval boundary."""

import asyncio
import json
from unittest.mock import Mock

import pytest

from intaris.audit import AuditStore
from intaris.config import DBConfig
from intaris.db import Database
from intaris.evaluator import Evaluator
from intaris.policy_activation import HybridError
from intaris.policy_inheritance import validate_parent_chain
from intaris.session import SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(Database(DBConfig(path=str(tmp_path / "inherit.db"))))


def parent(store, **policy):
    return store.create(
        user_id="owner",
        session_id="parent",
        intention="Research offers",
        policy={
            "evaluation_policy_version": "hybrid_v1",
            "profile": "balanced",
            **policy,
        },
    )


def test_child_inherits_mode_restrictions_and_bounds_but_cannot_add_grants(store):
    parent(
        store,
        deny_tools=["parent-forbidden"],
        allow_tools=["read"],
        read_only_required=True,
        minimum_outcome="escalate",
        maximum_outcome="escalate",
        deny_policies=["Production changes require a human checkpoint."],
    )
    child = store.create(
        user_id="owner",
        session_id="child",
        parent_session_id="parent",
        intention="Research",
        policy={
            "evaluation_policy_version": "legacy",
            "allow_tools": ["bash"],
            "deny_tools": ["child-forbidden"],
            "maximum_outcome": "approve",
            "read_only_required": False,
            "interaction_mode": "none",
        },
    )
    policy = child["policy"]
    assert policy["evaluation_policy_version"] == "hybrid_v1"
    assert policy["profile"] == "balanced"
    assert policy["allow_tools"] == []
    assert set(policy["deny_tools"]) == {"parent-forbidden", "child-forbidden"}
    assert policy["read_only_required"] is True
    assert policy["minimum_outcome"] == "escalate"
    assert policy["maximum_outcome"] == "escalate"
    assert policy["deny_policies"] == ["Production changes require a human checkpoint."]
    assert policy["interaction_mode"] == "none"
    assert json.loads(child["policy_source"])["evaluation_policy_version"] == "legacy"


def test_parent_changes_refresh_descendants_without_fossilizing_inherited_clauses(
    store,
):
    parent(store, deny_tools=["old-parent-denial"])
    store.create(
        user_id="owner",
        session_id="child",
        parent_session_id="parent",
        intention="Research",
        policy={"deny_tools": ["child-denial"]},
    )
    store.create(
        user_id="owner",
        session_id="grandchild",
        parent_session_id="child",
        intention="Summarize",
    )
    before = store.get("grandchild", user_id="owner")
    store.update_session(
        "parent",
        user_id="owner",
        policy={
            "evaluation_policy_version": "hybrid_v1",
            "profile": "strict",
            "deny_tools": ["new-parent-denial"],
        },
    )
    after = store.refresh_inherited_policy("grandchild", user_id="owner")
    assert after["policy"]["profile"] == "strict"
    assert set(after["policy"]["deny_tools"]) == {"new-parent-denial", "child-denial"}
    assert after["state_revision"] > before["state_revision"]
    revision = after["state_revision"]
    assert (
        store.refresh_inherited_policy("grandchild", user_id="owner")["state_revision"]
        == revision
    )


def test_parent_change_between_activation_and_admission_is_fenced(store):
    parent(store)
    child = store.create(
        user_id="owner",
        session_id="child",
        parent_session_id="parent",
        intention="Research",
    )
    store.update_session(
        "parent",
        user_id="owner",
        policy={
            "evaluation_policy_version": "hybrid_v1",
            "profile": "strict",
        },
    )
    with store._db.cursor() as cur:
        with pytest.raises(HybridError, match="Parent authorization changed"):
            validate_parent_chain(cur, child)


def test_child_minimum_cannot_be_bypassed_by_omitting_request_floor(store):
    parent(store, minimum_outcome="deny")
    store.create(
        user_id="owner",
        session_id="child",
        parent_session_id="parent",
        intention="Research",
    )
    store.set_alignment_state("child", user_id="owner", state="ready")
    llm = Mock()
    result = Evaluator(
        llm=llm, session_store=store, audit_store=AuditStore(store._db)
    ).evaluate(
        user_id="owner",
        session_id="child",
        agent_id=None,
        tool="read",
        args={"file_path": "/repo/file"},
    )
    assert result["decision"] == "deny"
    llm.generate.assert_not_called()


def test_child_cannot_invent_allowance_and_cannot_read_other_tenant_parent(store):
    parent(store)
    with pytest.raises(HybridError, match="Child prose allowances"):
        store.create(
            user_id="owner",
            session_id="child",
            parent_session_id="parent",
            intention="Research",
            policy={"allow_policies": ["Allow arbitrary uploads."]},
        )
    with pytest.raises(ValueError):
        store.create(
            user_id="other-tenant",
            session_id="other-child",
            parent_session_id="parent",
            intention="Research",
        )


def test_explicit_legacy_parent_remains_legacy_even_if_child_requests_hybrid(store):
    store.create(
        user_id="owner",
        session_id="parent",
        intention="Legacy work",
        policy={"evaluation_policy_version": "legacy", "deny_tools": ["forbidden"]},
    )
    child = store.create(
        user_id="owner",
        session_id="child",
        parent_session_id="parent",
        intention="Support",
        policy={"evaluation_policy_version": "hybrid_v1"},
    )
    assert child["policy"]["evaluation_policy_version"] == "legacy"
    assert child["policy"]["deny_tools"] == ["forbidden"]


def test_nested_activation_refreshes_changed_ancestor_before_child_has_evaluated(store):
    parent(store)
    store.create(
        user_id="owner",
        session_id="child",
        parent_session_id="parent",
        intention="Research",
    )
    policy = {"evaluation_policy_version": "hybrid_v1", "profile": "strict"}
    store.update_session("parent", user_id="owner", policy=policy)
    assert store.get("child", user_id="owner")["policy"]["profile"] == "balanced"
    grandchild = store.create(
        user_id="owner",
        session_id="grandchild",
        parent_session_id="child",
        intention="Research",
        policy={"evaluation_policy_version": "hybrid_v1", "profile": "balanced"},
    )
    assert grandchild["policy"]["profile"] == "strict"
    assert store.get("child", user_id="owner")["policy"]["profile"] == "strict"
    updated = store.update_session(
        "grandchild",
        user_id="owner",
        policy={"evaluation_policy_version": "legacy"},
    )
    assert updated["policy"]["profile"] == "strict"


def raced_parent_reads(store, monkeypatch, suspended_reads):
    """Make the given parent reads (1-based) observe a stale suspension."""
    parent(store, evaluation_policy_version="legacy")
    child = store.create(
        user_id="owner",
        session_id="child",
        parent_session_id="parent",
        intention="Research",
    )
    original_get = store.get
    reads = []

    def raced_get(session_id, *, user_id):
        value = original_get(session_id, user_id=user_id)
        if session_id == "parent":
            reads.append(1)
            if suspended_reads(len(reads)):
                return {**value, "status": "suspended"}
        return value

    monkeypatch.setattr(store, "get", raced_get)
    monkeypatch.setattr(store, "refresh_inherited_policy", lambda *a, **kw: child)
    model = Mock()
    evaluator = Evaluator(
        llm=model, session_store=store, audit_store=AuditStore(store._db)
    )
    return evaluator, model, reads, original_get


def test_parent_resume_between_lifecycle_reads_is_reevaluated_not_swallowed(
    store, monkeypatch
):
    evaluator, model, reads, original_get = raced_parent_reads(
        store, monkeypatch, lambda read: read == 1
    )
    result = evaluator.evaluate(
        user_id="owner",
        session_id="child",
        agent_id=None,
        tool="read",
        args={"path": "/repo/readme.md"},
    )
    # The re-run saw the active parent: a real evaluation, not a lifecycle denial.
    assert result["decision"] == "approve"
    assert len(reads) > 2
    assert original_get("child", user_id="owner")["status"] == "active"


def test_persistent_parent_conflict_still_fails_closed(store, monkeypatch):
    evaluator, model, reads, original_get = raced_parent_reads(
        store, monkeypatch, lambda read: read % 2 == 1
    )
    with pytest.raises(HybridError) as error:
        evaluator.evaluate(
            user_id="owner",
            session_id="child",
            agent_id=None,
            tool="read",
            args={"path": "/repo/readme.md"},
        )
    assert error.value.code == "policy_state_changed"
    assert error.value.retryable
    assert len(reads) == 6  # Three bounded attempts, two parent reads each.
    model.generate.assert_not_called()
    assert original_get("child", user_id="owner")["status"] == "active"


def test_pinned_revision_keeps_parent_conflict_fence(store, monkeypatch):
    evaluator, model, reads, original_get = raced_parent_reads(
        store, monkeypatch, lambda read: read == 1
    )
    with pytest.raises(HybridError) as error:
        evaluator.evaluate(
            user_id="owner",
            session_id="child",
            agent_id=None,
            tool="read",
            args={"path": "/repo/readme.md"},
            expected_policy_revision=original_get("child", user_id="owner")[
                "policy_revision"
            ],
        )
    assert error.value.code == "policy_state_changed"
    assert len(reads) == 2


@pytest.mark.parametrize("update", ["intention", "user_evidence", "suspension"])
def test_ancestor_authorization_changes_invalidate_descendant_before_admission(
    store, update
):
    parent(store)
    child = store.create(
        user_id="owner",
        session_id="child",
        parent_session_id="parent",
        intention="Research",
    )
    if update == "intention":
        store.update_session(
            "parent", user_id="owner", intention="Stop research; read-only review"
        )
    elif update == "user_evidence":
        with store._db.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET latest_user_evidence = ?, state_revision = state_revision + 1 "
                "WHERE user_id = ? AND session_id = ?",
                (
                    json.dumps({"content": "Do not publish anything."}),
                    "owner",
                    "parent",
                ),
            )
    else:
        store.update_status("parent", "suspended", user_id="owner")
    with store._db.cursor() as cur:
        with pytest.raises(HybridError):
            validate_parent_chain(cur, child)
    if update == "suspension":
        with pytest.raises(HybridError, match="suspended"):
            store.refresh_inherited_policy("child", user_id="owner")
    else:
        refreshed = store.refresh_inherited_policy("child", user_id="owner")
        assert refreshed["state_revision"] > child["state_revision"]
        with store._db.cursor() as cur:
            validate_parent_chain(cur, refreshed)


@pytest.mark.parametrize(
    "parent_choice,child_choice,expected",
    [
        ("defer", "defer", "defer"),
        ("defer", "deny", "deny"),
        ("deny", "defer", "deny"),
    ],
)
def test_judge_defer_vocabulary_is_not_confused_with_tool_escalation(
    store, parent_choice, child_choice, expected
):
    parent(store, judge={"on_uncertain": parent_choice})
    child = store.create(
        user_id="owner",
        session_id="child",
        parent_session_id="parent",
        intention="Research",
        policy={
            "judge": {"on_uncertain": child_choice, "allowed_decisions": ["defer"]}
        },
    )
    assert child["policy"]["judge"]["on_uncertain"] == expected
    assert child["policy"]["judge"]["allowed_decisions"] == ["defer"]


def test_parent_change_before_child_admission_is_reevaluated(store, monkeypatch):
    parent(store)
    store.create(
        user_id="owner",
        session_id="child",
        parent_session_id="parent",
        intention="Research",
    )
    with store._db.cursor() as cur:
        cur.execute("UPDATE sessions SET alignment_state = 'ready'")
    audit = AuditStore(store._db)
    original_insert = audit.insert
    admissions = []

    def insert_after_parent_change(**kwargs):
        if kwargs.get("admission_snapshot") is not None:
            admissions.append(1)
            if len(admissions) == 1:
                # A busy parent records new user evidence mid-evaluation.
                store.update_session(
                    "parent", user_id="owner", intention="Research offers today"
                )
        return original_insert(**kwargs)

    monkeypatch.setattr(audit, "insert", insert_after_parent_change)
    evaluator = Evaluator(llm=Mock(), session_store=store, audit_store=audit)
    result = evaluator.evaluate(
        user_id="owner",
        session_id="child",
        agent_id=None,
        tool="read",
        args={"path": "/repo/readme.md"},
    )
    assert result["decision"] == "approve"
    assert len(admissions) == 2
    # The admitted child is bound to the parent's current authorization.
    child = store.get("child", user_id="owner")
    with store._db.cursor() as cur:
        validate_parent_chain(cur, child)


def test_async_evaluation_reevaluates_parent_race(store, monkeypatch):
    evaluator, _, reads, _ = raced_parent_reads(
        store, monkeypatch, lambda read: read == 1
    )
    result = asyncio.run(
        evaluator.evaluate_async(
            user_id="owner",
            session_id="child",
            agent_id=None,
            tool="read",
            args={"path": "/repo/readme.md"},
        )
    )
    assert result["decision"] == "approve"
    assert len(reads) > 2


@pytest.mark.parametrize(
    "error",
    [
        HybridError("stale_admission", "Session changed during evaluation", 409),
        HybridError("policy_state_changed", "Invalid policy ancestry", 409),
    ],
)
def test_session_local_or_permanent_conflicts_are_not_rerun(store, error):
    calls = []

    def flow(**kwargs):
        calls.append(kwargs)
        raise error
        yield  # pragma: no cover

    evaluator = Evaluator(
        llm=Mock(), session_store=store, audit_store=AuditStore(store._db)
    )
    evaluator._evaluation_flow = flow
    with pytest.raises(HybridError):
        evaluator.evaluate(
            user_id="owner", session_id="child", agent_id=None, tool="read", args={}
        )
    assert len(calls) == 1
