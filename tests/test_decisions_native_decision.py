"""Native three-question Decisions second opinion inside the hybrid policy (offline HTTP)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from intaris import l1_review
from intaris.audit import AuditStore
from intaris.config import DBConfig, DecisionsConfig
from intaris.db import Database
from intaris.decisions import DecisionsClient
from intaris.evaluator import Evaluator
from intaris.evaluator_backend import EvaluatorBackend
from intaris.policy_activation import HybridError
from intaris.session import SessionStore
from tests.test_decisions import answer_request

NATIVE_QUESTIONS = ["aligned", "risk"]
# Factual best guesses for a routine local edit; relationship is unconfident.
ROUTINE = {
    "relationship": "supports_intent",
    "risk": "low",
    "mutation": "mutating",
    "effect_read": "no",
    "effect_write": "yes",
    "effect_execute": "no",
    "effect_transmit": "no",
    "effect_delete": "no",
    "contextual_restrictions": "consistent",
    "target_scope": "consistent",
}


def unconfident(payload, name):
    for answer in payload["answers"]:
        if answer["name"] == name:
            rows = answer["probabilities"]
            for row in rows:
                row["probability"] = (
                    0.4 if row["value"] == answer["choice"] else 0.6 / (len(rows) - 1)
                )
            answer["confidence"] = 0.3


def client(
    factual=None,
    native=None,
    *,
    unsure=("relationship",),
    native_mutate=None,
    native_status=200,
):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        names = [q["name"] for q in body["questions"]]
        requests.append(names)
        if "aligned" in names and native_status != 200:
            return httpx.Response(native_status, json={"error": {"message": "busy"}})
        if "aligned" in names:
            payload = answer_request(body, native)
            if native_mutate:
                native_mutate(payload)
            return httpx.Response(200, json=payload)
        payload = answer_request(body, {**ROUTINE, **(factual or {})})
        for name in unsure:
            unconfident(payload, name)
        return httpx.Response(200, json=payload)

    return (
        DecisionsClient(
            DecisionsConfig(api_key="native-test-key"),
            transport=httpx.MockTransport(handler),
        ),
        requests,
    )


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("HYBRID_CONTEXT_MODE", "mixed")
    opened = []

    def evaluate(
        tool,
        args,
        *,
        factual=None,
        native=None,
        unsure=("relationship",),
        native_mutate=None,
        native_status=200,
        policy=None,
        working_directory="/repo",
    ):
        decisions, requests = client(
            factual,
            native,
            unsure=unsure,
            native_mutate=native_mutate,
            native_status=native_status,
        )
        backend = EvaluatorBackend("decisions", decisions_factory=lambda: decisions)
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
        return result, requests, llm

    yield evaluate
    for backend, db in opened:
        backend.close()
        db.close()


def stage(result):
    return result["evaluation_metadata"]["stages"][0]


def factual_only(requests):
    return bool(requests) and all("aligned" not in names for names in requests)


WRITE = ("write", {"file_path": "/repo/usage.md", "content": "x"})
BASH = ("bash", {"command": "npm run build && npm test"})


@pytest.mark.parametrize("tool,args", [WRITE, BASH])
def test_unconfident_facts_get_a_native_decision_instead_of_the_llm(run, tool, args):
    result, requests, llm = run(tool, args)
    assert requests[-1] == NATIVE_QUESTIONS and len(requests) == 2
    llm.generate.assert_not_called()
    assert result["decision"] == "approve"
    l1 = stage(result)
    assert l1["backend_provenance"]["route"] == "decisions"
    assert [row["backend"] for row in l1["backend_provenance"]["attempts"]] == [
        "decisions",
        "decisions",
    ]
    assert l1["native_decision"] == {
        "approved": True,
        "aligned": True,
        "risk": "low",
        "decision": "approve",
    }
    assert l1["classification"]["relationship"] == "supports_intent"
    assert l1["classification"]["risk"] == "low"
    # Confident factual answers survive; the native answer settles only intent.
    assert l1["classification"]["target_scope"] == "consistent"


def test_confident_facts_need_no_second_opinion(run):
    result, requests, llm = run(*WRITE, unsure=())
    assert factual_only(requests) and len(requests) == 1
    llm.generate.assert_not_called()
    assert result["decision"] == "approve"


@pytest.mark.parametrize(
    "native,native_mutate",
    [
        ({"risk": "high"}, None),
        ({"risk": "critical"}, None),
        (None, lambda p: p["answers"][0].update(probability=0.2)),
        (None, lambda p: p["answers"][0].update(probability=0.5)),
        (None, lambda p: p["answers"][1].update(confidence=0.69)),
        (None, lambda p: p["answers"][0].update(type="refusal")),
    ],
)
def test_any_other_native_answer_keeps_the_generative_review(
    run, native, native_mutate
):
    result, requests, llm = run(*BASH, native=native, native_mutate=native_mutate)
    assert requests[-1] == NATIVE_QUESTIONS
    llm.generate.assert_called()
    # The generative review failed in this harness: never an approval.
    assert isinstance(result, HybridError)
    assert result.code == "assessment_failed"


@pytest.mark.parametrize(
    "factual",
    [
        {"effect_transmit": "yes"},
        {"effect_delete": "yes"},
        {"relationship": "unrelated"},
        {"relationship": "conflicts_with_intent"},
        {"risk": "high"},
        {"target_scope": "conflicts"},
        {"contextual_restrictions": "conflicts"},
    ],
)
def test_consequential_best_guesses_skip_the_native_decision(run, factual):
    result, requests, llm = run(*BASH, factual=factual, unsure=tuple(factual))
    assert factual_only(requests)
    llm.generate.assert_called()
    assert isinstance(result, HybridError) or result["decision"] != "approve"


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
        ("write", {"file_path": "/repo/.git/config", "content": "x"}, None),
        (
            "write",
            {"file_path": "/repo/.github/workflows/ci.yml", "content": "x"},
            None,
        ),
        ("bash", {"command": "git push origin main"}, None),
        ("mcp__chat__send_message", {"text": "Build finished"}, None),
    ],
)
def test_ungroundable_calls_never_ask_the_native_questions(run, tool, args, policy):
    _, requests, _ = run(tool, args, policy=policy)
    assert all("aligned" not in names for names in requests)


@pytest.mark.parametrize(
    "policy",
    [{"deny_tools": ["write"]}, {"read_only_required": True}],
)
def test_native_approve_cannot_override_hard_policy(run, policy):
    result, requests, _ = run(*WRITE, policy=policy)
    assert all("aligned" not in names for names in requests)
    assert isinstance(result, HybridError) or result["decision"] != "approve"


def test_relative_working_directory_is_not_eligible(run):
    _, requests, _ = run(*WRITE, working_directory="repo")
    assert all("aligned" not in names for names in requests)


def test_bounded_autonomy_profile_uses_native_decision(run):
    result, requests, llm = run(*BASH, policy={"profile": "bounded_autonomy"})
    assert requests[-1] == NATIVE_QUESTIONS
    llm.generate.assert_not_called()
    assert result["decision"] == "approve"


def _eligible(sources, values=None, **kwargs):
    state = SimpleNamespace(
        read_only_required=False,
        advisory_clauses=[],
        process_lifecycle=None,
        workflow_scope=None,
        sources=sources,
        value=lambda key, default=None: {"paths": ["/repo/a.md"], **(values or {})}.get(
            key, default
        ),
    )
    result = SimpleNamespace(
        outstanding_obligations=[SimpleNamespace(id="intent.assessment")]
    )
    decisions = DecisionsClient(DecisionsConfig(api_key="native-test-key"))
    try:
        return l1_review.native_decision_eligible(
            decisions,
            state,
            result,
            profile="balanced",
            tool="write",
            args={"file_path": "/repo/a.md"},
            working_directory="/repo",
            **kwargs,
        )
    finally:
        decisions.close()


def test_user_restrictions_and_failed_extraction_keep_the_factual_review():
    user = {"origin": "client_user_claim", "text": "Fix the docs"}
    # Mixed context has no extractor; the balanced policy accepts that.
    assert _eligible({"latest_user_message": {**user, "interpretation_missing": True}})
    assert not _eligible(
        {"latest_user_message": {**user, "constraint_items": [{"kind": "restriction"}]}}
    )
    # A configured extractor that failed is relabelled raw by the relaxed view.
    assert not _eligible({"latest_user_message": user}, interpretation_unavailable=True)


@pytest.mark.parametrize(
    "effects,eligible",
    [
        (None, True),
        (["unknown"], True),
        (["write", "execute"], True),
        (["transmit"], False),
    ],
)
def test_statically_known_effects_must_stay_local(effects, eligible):
    assert _eligible({}, {"effects": effects} if effects else {}) is eligible


def test_unconfident_restrictions_keep_the_generative_review(run):
    result, requests, llm = run(
        *BASH, unsure=("relationship", "contextual_restrictions")
    )
    assert factual_only(requests)
    llm.generate.assert_called()
    assert isinstance(result, HybridError)


def test_native_outage_falls_back_to_the_generative_review(run):
    result, requests, llm = run(*BASH, native_status=503)
    assert any(names == NATIVE_QUESTIONS for names in requests)
    llm.generate.assert_called()
    assert isinstance(result, HybridError) and result.code == "assessment_failed"


@pytest.mark.parametrize(
    "view",
    [{"source_coverage": "partial"}, {"affirmative_grants_eligible": False}],
)
def test_trimmed_input_keeps_the_generative_review(view):
    routine = {"distributions": {"relationship": {"choice": "supports_intent"}}}
    assert l1_review.native_answers_eligible(routine)
    assert not l1_review.native_answers_eligible({**routine, "model_view": view})


def native_answers(aligned, risk=None, confidence=1.0):
    risk = risk or {"low": 1.0}

    def mutate(payload):
        for answer in payload["answers"]:
            if answer["name"] == "aligned":
                answer["probability"] = aligned
            if answer["name"] == "risk":
                answer["choice"] = max(risk, key=risk.get)
                answer["confidence"] = confidence
                answer["probabilities"] = [
                    {"value": row["value"], "probability": risk.get(row["value"], 0.0)}
                    for row in answer["probabilities"]
                ]

    return mutate


def evaluate_native(mutate, minimum=0.7, operation=None):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        payload = answer_request(body)
        mutate(payload)
        return httpx.Response(200, json=payload)

    decisions = DecisionsClient(
        DecisionsConfig(api_key="native-test-key", minimum_confidence=minimum),
        transport=httpx.MockTransport(handler),
    )
    try:
        return (
            decisions.evaluate_native(
                state={"task_context": "Edit /repo/usage.md"},
                evaluation_rules="Apply the Intaris rubric.",
                operation=operation,
            ),
            requests,
        )
    finally:
        decisions.close()


def test_aligned_and_risk_decide_by_chosen_answer_probability():
    # Production shape: aligned 0.79 fails a margin rule (0.58) but passes
    # the pre-hybrid probability rule at 0.7; no separate decision question.
    result, requests = evaluate_native(
        native_answers(0.79, {"low": 0.75, "medium": 0.25}, confidence=0.8)
    )
    assert "classification_uncertainty" not in result.metadata
    assert (result.aligned, result.risk, result.decision) == (True, "low", "approve")
    assert result.metadata["alignment_confidence"] == pytest.approx(0.79)
    assert result.metadata["risk_confidence"] == pytest.approx(0.75)
    assert [q["name"] for q in requests[0]["questions"]] == ["aligned", "risk"]


@pytest.mark.parametrize(
    "aligned,risk,decision",
    [
        (0.9, {"medium": 1.0}, "approve"),
        (0.9, {"high": 1.0}, "escalate"),
        (0.9, {"critical": 1.0}, "deny"),
        (0.1, {"low": 1.0}, "escalate"),
    ],
)
def test_decision_follows_alignment_and_risk(aligned, risk, decision):
    result, _ = evaluate_native(native_answers(aligned, risk))
    assert result.decision == decision


@pytest.mark.parametrize(
    "aligned,risk,confidence,minimum,certain",
    [
        (0.65, {"low": 1.0}, 1.0, 0.7, False),
        (0.65, {"low": 1.0}, 1.0, 0.6, True),
        (0.95, {"low": 0.65, "medium": 0.35}, 1.0, 0.7, False),
        (0.95, {"low": 0.9, "medium": 0.1}, 0.6, 0.7, False),
        (0.95, {"low": 0.9, "medium": 0.1}, 0.6, 0.6, True),
    ],
)
def test_native_threshold_follows_minimum_confidence(
    aligned, risk, confidence, minimum, certain
):
    result, _ = evaluate_native(
        native_answers(aligned, risk, confidence), minimum=minimum
    )
    assert ("classification_uncertainty" not in result.metadata) is certain


def test_invalid_native_response_is_never_certain():
    result, _ = evaluate_native(lambda payload: payload["answers"].pop())
    assert result.metadata["validation_error"]
    assert result.decision == "escalate"


@pytest.mark.parametrize(
    "tool,kind,phrase",
    [
        ("read", "read", "reads or searches"),
        ("patch", "edit", "edits files"),
        ("bash", "command", "runs a command"),
        ("subagent", "delegate", "delegates work"),
        ("custom_tool", None, None),
    ],
)
def test_questions_carry_operation_guidance(tool, kind, phrase):
    from intaris.decisions import native_operation

    assert native_operation(tool) == kind
    _, requests = evaluate_native(native_answers(0.9), operation=kind)
    aligned = requests[0]["questions"][0]["instructions"]
    if phrase:
        assert phrase in aligned
    else:
        assert "This call" not in aligned


def test_production_flow_sends_operation_guidance(run, monkeypatch):
    seen = []
    original = DecisionsClient.evaluate_native

    def spy(self, **kwargs):
        seen.append(kwargs.get("operation"))
        return original(self, **kwargs)

    monkeypatch.setattr(DecisionsClient, "evaluate_native", spy)
    result, requests, _ = run(*BASH)
    assert requests[-1] == NATIVE_QUESTIONS
    assert seen == ["command"]
    assert result["decision"] == "approve"
