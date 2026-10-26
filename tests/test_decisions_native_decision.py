"""Native three-question Decisions answer inside the hybrid policy (offline HTTP)."""

import json
from unittest.mock import MagicMock

import pytest

from intaris.audit import AuditStore
from intaris.config import DBConfig
from intaris.db import Database
from intaris.evaluator import Evaluator
from intaris.evaluator_backend import EvaluatorBackend
from intaris.policy_activation import HybridError
from intaris.session import SessionStore
from tests.test_decisions import client_for

NATIVE_QUESTIONS = ["aligned", "risk", "decision"]


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("HYBRID_CONTEXT_MODE", "mixed")
    opened = []

    def evaluate(
        tool, args, *, labels=None, mutate=None, policy=None, working_directory="/repo"
    ):
        client, requests = client_for(labels=labels, mutate=mutate)
        backend = EvaluatorBackend("decisions", decisions_factory=lambda: client)
        db = Database(DBConfig(backend="sqlite", path=str(tmp_path / "native.db")))
        opened.append((backend, db))
        sessions, audit = SessionStore(db), AuditStore(db)
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("generative review requested")
        sessions.create(
            user_id="owner",
            session_id="work",
            intention="Update the usage guide in the repository",
            details={"working_directory": working_directory},
            policy={"evaluation_policy_version": "hybrid_v1", **(policy or {})},
        )
        evaluator = Evaluator(
            llm=llm,
            evaluator_backend=backend,
            session_store=sessions,
            audit_store=audit,
        )
        try:
            result = evaluator.evaluate(
                user_id="owner",
                session_id="work",
                agent_id=None,
                tool=tool,
                args=args,
            )
        except HybridError as exc:
            result = exc
        asked = [
            [q["name"] for q in json.loads(r.content)["questions"]] for r in requests
        ]
        return result, asked, llm

    yield evaluate
    for backend, db in opened:
        backend.close()
        db.close()


def stage(result):
    return result["evaluation_metadata"]["stages"][0]


def test_confident_native_approve_decides_without_generative_review(run):
    result, asked, llm = run("write", {"file_path": "/repo/usage.md", "content": "x"})
    assert asked == [NATIVE_QUESTIONS]
    llm.generate.assert_not_called()
    assert result["decision"] == "approve"
    l1 = stage(result)
    assert l1["native_decision"] == {
        "approved": True,
        "aligned": True,
        "risk": "low",
        "decision": "approve",
    }
    assert l1["backend_provenance"]["route"] == "decisions"
    # Fields the three answers cannot ground stay abstentions for the policy.
    assert l1["classification"]["relationship"] == "supports_intent"
    assert l1["classification"]["target_scope"] == "unresolved"
    assert l1["classification"]["contextual_restrictions"] == "unresolved"


@pytest.mark.parametrize(
    "labels,mutate",
    [
        ({"decision": "escalate"}, None),
        ({"decision": "deny"}, None),
        ({"risk": "high"}, None),
        (None, lambda p: p["answers"][0].update(probability=0.5)),
        (None, lambda p: p["answers"][2].update(confidence=0.69)),
        (None, lambda p: p["answers"][0].update(type="refusal")),
    ],
)
def test_any_other_native_answer_requires_the_factual_review(run, labels, mutate):
    result, asked, llm = run(
        "write",
        {"file_path": "/repo/usage.md", "content": "x"},
        labels=labels,
        mutate=mutate,
    )
    assert asked == [NATIVE_QUESTIONS]
    llm.generate.assert_called()
    # The mandatory review failed in this harness: never an approval.
    assert isinstance(result, HybridError)
    assert result.code == "assessment_failed"


@pytest.mark.parametrize(
    "tool,args,policy",
    [
        ("write", {"file_path": "/etc/hosts", "content": "x"}, None),
        ("write", {"file_path": "/repo/../etc/hosts", "content": "x"}, None),
        (
            "write",
            {"file_path": "/repo/usage.md", "content": "x"},
            {"profile": "strict"},
        ),
        ("bash", {"command": "git push origin main"}, None),
        ("write", {"file_path": "/repo/.git/config", "content": "x"}, None),
        (
            "write",
            {"file_path": "/repo/.github/workflows/ci.yml", "content": "x"},
            None,
        ),
        ("read", {"file_path": "/repo/.env"}, None),
        ("custom_tool", {"file_path": "/repo/usage.md"}, None),
    ],
)
def test_ungroundable_calls_keep_the_factual_l1(run, tool, args, policy):
    result, asked, _ = run(tool, args, policy=policy)
    assert asked
    assert all(questions != NATIVE_QUESTIONS for questions in asked)


@pytest.mark.parametrize(
    "policy",
    [{"deny_tools": ["write"]}, {"read_only_required": True}],
)
def test_native_approve_cannot_override_hard_policy(run, policy):
    result, asked, _ = run(
        "write", {"file_path": "/repo/usage.md", "content": "x"}, policy=policy
    )
    assert asked != [NATIVE_QUESTIONS]
    assert isinstance(result, HybridError) or result["decision"] != "approve"


def test_relative_working_directory_is_not_eligible(run):
    _, asked, _ = run(
        "write",
        {"file_path": "/repo/usage.md", "content": "x"},
        working_directory="repo",
    )
    assert asked != [NATIVE_QUESTIONS]


def test_bounded_autonomy_profile_uses_native_decision(run):
    result, asked, llm = run(
        "write",
        {"file_path": "/repo/usage.md", "content": "x"},
        policy={"profile": "bounded_autonomy"},
    )
    assert asked == [NATIVE_QUESTIONS]
    llm.generate.assert_not_called()
    assert result["decision"] == "approve"
