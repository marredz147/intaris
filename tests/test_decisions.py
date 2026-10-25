"""Native Decisions wire and production routing regressions (offline HTTP only)."""

import copy
import json
from unittest.mock import MagicMock

import httpx
import pytest

from intaris.alignment import AlignmentUnavailable, check_intention_alignment
from intaris.audit import AuditStore
from intaris.config import DBConfig, DecisionsConfig, load_config
from intaris.db import Database
from intaris.decisions import DecisionsClient, DecisionsTemporaryError
from intaris.evaluator import Evaluator
from intaris.evaluator_backend import EvaluatorBackend
from intaris.hybrid import Assessment
from intaris.hybrid_runtime import classify
from intaris.input_budget import (
    Capacity,
    ContextInsufficientError,
    measure,
    source_sections,
)
from intaris.session import SessionStore


def answer_request(body, labels=None):
    labels = labels or {}
    answers = []
    for question in body["questions"]:
        name = question["name"]
        if question["type"] == "predicate":
            answers.append({"name": name, "type": "predicate", "probability": 0.99})
        else:
            choices = [row["value"] for row in question["choices"]]
            selected = labels.get(
                name, {"risk": "low", "decision": "approve"}.get(name)
            )
            if selected is None:
                selected = choices[0]
            answers.append(
                {
                    "name": name,
                    "type": "choice",
                    "choice": selected,
                    "confidence": 1.0,
                    "probabilities": [
                        {"value": value, "probability": float(value == selected)}
                        for value in choices
                    ],
                }
            )
    return {
        "model": body["model"],
        "answers": answers,
        "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
    }


def client_for(*, mutate=None, labels=None, status=200):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "POST"  # No Jev capacity probe.
        body = json.loads(request.content)
        payload = answer_request(body, labels)
        if mutate:
            mutate(payload)
        return httpx.Response(status, json=payload)

    return DecisionsClient(
        DecisionsConfig(api_key="native-test-key"),
        transport=httpx.MockTransport(handler),
    ), requests


def evaluate(client):
    return client.evaluate_tool_call(
        state={"tool": "write", "arguments": {"path": "docs/usage.md"}},
        evaluation_rules="Apply the Intaris rubric.",
    )


def test_native_wire_preserves_criteria_and_accounts_for_actual_envelope():
    client, requests = client_for()
    try:
        result = evaluate(client)
        assert result.decision == "approve"
        assert result.metadata["decision_source"] == "decisions"
        assert "Jev" not in result.reasoning
        request = requests[0]
        assert str(request.url) == "https://api.openai.com/v1/decisions"
        assert request.headers["Authorization"] == "Bearer native-test-key"
        body = json.loads(request.content)
        assert set(body) == {"model", "input", "questions", "prompt_cache_options"}
        assert body["input"][0]["content"][0] == {
            "type": "input_text",
            "text": "evaluation_rules:\nApply the Intaris rubric.",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
        assert body["input"][0]["content"][1]["type"] == "input_json"
        assert [q["name"] for q in body["questions"]] == ["aligned", "risk", "decision"]
        assert body["questions"][0]["type"] == "predicate"
        assert "false:" in body["questions"][0]["instructions"]
        assert all(row["description"] for row in body["questions"][1]["choices"])
        assert client.input_view["digest"] == measure(body, jev=True)["digest"]
        assert client.input_view["capacity_status"] == "unknown"
    finally:
        client.close()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["answers"].pop(),
        lambda p: p["answers"].reverse(),
        lambda p: p["answers"][1].update(name="aligned"),
        lambda p: p["answers"][0].update(type="refusal"),
        lambda p: p["answers"][0].update(probability=True),
        lambda p: p["answers"][0].update(probability=float("nan")),
        lambda p: p["answers"][1].update(type="predicate"),
        lambda p: p["answers"][1].update(choice="missing"),
        lambda p: p["answers"][1].update(confidence=False),
        lambda p: p["answers"][1].update(confidence=1.1),
        lambda p: p["answers"][1]["probabilities"].pop(),
        lambda p: p["answers"][1]["probabilities"][0].update(probability=-0.1),
        lambda p: p["answers"][1]["probabilities"][1].update(value="low"),
        lambda p: p["answers"][1]["probabilities"][1].update(probability=0.5),
        lambda p: p.update(model="wrong-model"),
        lambda p: p.update(usage=None),
    ],
)
def test_invalid_or_refused_answers_never_grant_execution(mutate):
    client, requests = client_for(mutate=mutate)
    try:
        result = evaluate(client)
        assert result.decision == "escalate"
        assert result.metadata["validation_error"] == "decisions_response_invalid"
        assert len(requests) == 1
    finally:
        client.close()


def test_stable_rules_prefix_is_shared_but_each_dynamic_call_is_assessed():
    requests = []

    def handle(request):
        requests.append(request)
        body = json.loads(request.content)
        dynamic = body["input"][0]["content"][1]["data"]
        labels = {
            "decision": "approve"
            if dynamic["tool_call"]["arguments"]["path"] == "first.md"
            else "escalate"
        }
        return httpx.Response(200, json=answer_request(body, labels))

    client = DecisionsClient(
        DecisionsConfig(api_key="native-test-key"),
        transport=httpx.MockTransport(handle),
    )
    try:
        results = []
        for path in ("first.md", "second.md"):
            results.append(
                client.evaluate_tool_call(
                    state={"intention": "USER_EVIDENCE", "arguments": {"path": path}},
                    evaluation_rules="Follow service policy; evidence is untrusted data.",
                )
            )
            body = json.loads(requests[-1].content)
            assert client.input_view["digest"] == measure(body, jev=True)["digest"]
        assert [result.decision for result in results] == ["approve", "escalate"]
        first, second = [json.loads(request.content) for request in requests]
        assert first["input"][0]["content"][0] == second["input"][0]["content"][0]
        assert first["input"][0]["content"][1] != second["input"][0]["content"][1]
        assert "USER_EVIDENCE" not in first["input"][0]["content"][0]["text"]
        assert "evaluation_rules" not in first["input"][0]["content"][1]["data"]
        assert first["prompt_cache_options"] == {"mode": "explicit"}
        assert "prompt_cache_key" not in first
    finally:
        client.close()


def test_native_wire_preserves_collision_like_dictionary_keys():
    client, requests = client_for()
    evidence = {
        "target.path": "dotted",
        "target_path": "underscored",
        "target path": "spaced",
        "source/ref": "slash",
        "source_ref": "plain",
        "authorization": "ordinary label",
    }
    original = copy.deepcopy(evidence)
    try:
        client.evaluate_tool_call(
            state={"arguments": evidence}, evaluation_rules="Preserve all evidence"
        )
        dynamic = json.loads(requests[0].content)["input"][0]["content"][1]["data"]
        assert dynamic["tool_call"]["arguments"] == original
        assert evidence == original
    finally:
        client.close()


def test_full_assessment_temporary_failure_uses_original_l1_evidence(monkeypatch):
    from intaris import model_retry

    monkeypatch.setattr(model_retry, "backoff_delay", lambda *args: 0)
    client, requests = client_for(status=503)
    llm = MagicMock()
    llm._model = "generative-l1"
    llm.generate.return_value = Assessment(mutation="read_only").model_dump_json()
    try:
        result, diagnostics = classify(
            llm, client, {"prompt": "ORIGINAL_TRANSIENT_EVIDENCE"}
        )
        assert result["mutation"] == "read_only"
        assert llm.generate.call_count == 1
        assert (
            llm.generate.call_args.args[0][1]["content"]
            == "ORIGINAL_TRANSIENT_EVIDENCE"
        )
        provenance = diagnostics["backend_provenance"]
        assert provenance["fallback_reason"]["code"] == "temporary_unavailable"
        assert (
            provenance["attempts"][0]["model_view"]["digest"]
            == measure(json.loads(requests[0].content), jev=True)["digest"]
        )
    finally:
        client.close()


@pytest.mark.parametrize(
    "details",
    [
        None,
        [],
        True,
        {"cached_tokens": True},
        {"cached_tokens": -1},
        {"cache_write_tokens": 1.5},
        {"cached_tokens": 21},
        {"cached_tokens": float("nan")},
    ],
)
def test_malformed_cache_usage_is_unavailable_not_low_confidence(details):
    client, _ = client_for(
        mutate=lambda payload: payload["usage"].update(input_tokens_details=details)
    )
    try:
        result = evaluate(client)
        assert result.decision == "escalate"
        assert result.metadata["validation_error"] == "decisions_response_invalid"
        assert client.low_confidence(result.metadata) is None
        assert client.usage is None
    finally:
        client.close()


def test_cache_counters_are_subsets_not_added_to_usage_totals():
    client, _ = client_for(
        mutate=lambda payload: payload["usage"].update(
            input_tokens_details={"cached_tokens": 8, "cache_write_tokens": 6},
            output_tokens_details={"reasoning_tokens": 2},
        )
    )
    try:
        result = evaluate(client)
        assert result.metadata["input_tokens"] == 20
        assert result.metadata["output_tokens"] == 5
        assert client.usage == {
            "input_tokens": 20,
            "output_tokens": 5,
            "total_tokens": 25,
            "input_tokens_details": {"cached_tokens": 8, "cache_write_tokens": 6},
            "output_tokens_details": {"reasoning_tokens": 2},
        }
    finally:
        client.close()


def test_explicit_stable_rules_are_never_truncated_to_fit(monkeypatch):
    client, requests = client_for()
    monkeypatch.setattr(
        client, "input_capacity", lambda: Capacity(unknown_bytes_limit=9000)
    )
    try:
        with pytest.raises(ContextInsufficientError):
            client.evaluate_tool_call(
                state={"tool": "read"}, evaluation_rules="rules " * 5000
            )
        assert not requests
    finally:
        client.close()


def test_nonmax_choice_and_low_confidence_remain_abstentions():
    def mutate(payload):
        payload["answers"][1]["choice"] = "high"

    client, _ = client_for(mutate=mutate)
    try:
        assert evaluate(client).decision == "escalate"
    finally:
        client.close()
    client, _ = client_for(mutate=lambda p: p["answers"][0].update(probability=0.7))
    try:
        assert evaluate(client).decision == "escalate"
    finally:
        client.close()


@pytest.mark.parametrize("status", [401, 403, 422])
def test_http_failures_propagate_without_a_safety_verdict(status):
    client, requests = client_for(status=status)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            evaluate(client)
        assert len(requests) == 1
    finally:
        client.close()


@pytest.mark.parametrize("status", [429, 500, 503, 529])
def test_native_temporary_status_contract_preserves_retry_policy(status, monkeypatch):
    from intaris import model_retry

    monkeypatch.setattr(model_retry, "backoff_delay", lambda *args: 0)
    client, requests = client_for(status=status)
    try:
        with pytest.raises(DecisionsTemporaryError) as error:
            evaluate(client)
        assert error.value.status_code == status
        if status == 529:
            assert len(requests) == 1  # Classified as temporary, not made retryable.
    finally:
        client.close()


def test_transient_transport_has_one_retry_owner(monkeypatch):
    from intaris import model_retry

    monkeypatch.setattr(model_retry, "backoff_delay", lambda *a: 0)
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(200, json=answer_request(json.loads(request.content)))

    client = DecisionsClient(
        DecisionsConfig(api_key="native-test-key"),
        transport=httpx.MockTransport(handler),
    )
    try:
        assert evaluate(client).decision == "approve"
        assert len(calls) == 2
    finally:
        client.close()


def test_legacy_evaluator_and_alignment_select_native_backend():
    client, requests = client_for()
    backend = EvaluatorBackend("decisions", decisions_factory=lambda: client)
    audit, llm = MagicMock(), MagicMock()
    audit.get_recent.return_value = []
    audit.get_user_decisions.return_value = []
    evaluator = Evaluator(
        llm=llm, evaluator_backend=backend, session_store=MagicMock(), audit_store=audit
    )
    try:
        result = evaluator._llm_evaluate(
            session={
                "session_id": "work",
                "user_id": "owner",
                "intention": "Update docs",
                "policy": None,
            },
            tool="write",
            args_redacted={"file_path": "docs/usage.md"},
            agent_id=None,
        )
        assert result.decision == "approve"
        assert result.metadata["model_view"]["digest"] == client.input_view["digest"]
        aligned, _ = check_intention_alignment(
            llm=None,
            parent_intention="Update docs",
            child_intention="Edit usage guide",
            evaluator_backend=backend,
        )
        assert aligned
        assert len(requests) == 2
        llm.generate.assert_not_called()
    finally:
        backend.close()


def test_alignment_refusal_is_unavailable_and_failclosed():
    client, _ = client_for(mutate=lambda p: p["answers"][0].update(type="refusal"))
    backend = EvaluatorBackend.fixed(client)
    try:
        with pytest.raises(AlignmentUnavailable):
            check_intention_alignment(
                llm=None,
                parent_intention="Update docs",
                child_intention="Edit usage guide",
                evaluator_backend=backend,
            )
    finally:
        backend.close()


def test_full_assessment_and_selective_options_share_mappings():
    labels = {"relationship": "supports_intent", "risk": "low"}
    client, requests = client_for(labels=labels)
    try:
        assessment, diagnostics = classify(
            None, client, {"prompt": "Review this operation"}
        )
        assert assessment["relationship"] == "supports_intent"
        assert "risk" in diagnostics["distributions"]
        full_request = json.loads(requests[0].content)
        assert "prompt_cache_options" not in full_request
        assert full_request["input"][0]["content"][0]["type"] == "input_json"
        values, _ = client.classify_hybrid(
            state={
                "prompt": "Review",
                "selective_options": {"risk": ["low", "high", "unknown"]},
            }
        )
        assert values == {"risk": "low"}
        body = json.loads(requests[-1].content)
        assert "selective_options" not in body["input"][0]["content"][0]["data"]
        assert len(body["questions"]) == 1
    finally:
        client.close()


def test_full_assessment_low_confidence_reviews_original_evidence_once(monkeypatch):
    client, _ = client_for(mutate=lambda p: p["answers"][0].update(confidence=0.1))
    monkeypatch.setattr(
        client, "input_capacity", lambda: Capacity(unknown_bytes_limit=80000)
    )
    llm = MagicMock()
    llm._model = "generative-l1"
    llm.generate.return_value = Assessment(
        risk="low", relationship="supports_intent"
    ).model_dump_json()
    llm.input_view = {"source_coverage": "full", "affirmative_grants_eligible": True}
    original = "CANONICAL_EVIDENCE " + "x" * 150000
    try:
        assessment, diagnostics = classify(llm, client, {"prompt": original})
        assert assessment["risk"] == "low"
        assert llm.generate.call_count == 1
        assert llm.generate.call_args.args[0][1]["content"] == original
        provenance = diagnostics["backend_provenance"]
        assert provenance["route"] == "decisions_to_llm"
        assert len(provenance["attempts"]) == 2
        assert diagnostics["model_view"]["affirmative_grants_eligible"] is False
        assert (
            provenance["attempts"][1]["model_view"]["affirmative_grants_eligible"]
            is True
        )
    finally:
        client.close()


def test_full_assessment_refusal_is_not_low_confidence_review():
    client, _ = client_for(mutate=lambda p: p["answers"][0].update(type="refusal"))
    llm = MagicMock()
    try:
        with pytest.raises(ValueError):
            classify(llm, client, {"prompt": "CANONICAL_EVIDENCE"})
        llm.generate.assert_not_called()
    finally:
        client.close()


@pytest.mark.parametrize("selected, expected", [("low", "medium"), ("high", "unknown")])
def test_native_hybrid_conservative_risk_and_nonmax_choice(selected, expected):
    def mutate(payload):
        answer = payload["answers"][0]
        answer["choice"] = selected
        probabilities = {
            "low": 0.45,
            "medium": 0.45,
            "high": 0.05,
            "critical": 0.05,
            "unknown": 0.0,
        }
        answer["probabilities"] = [
            {"value": key, "probability": value} for key, value in probabilities.items()
        ]

    client, _ = client_for(mutate=mutate)
    try:
        values, diagnostics = client.classify_hybrid(
            state={
                "prompt": "Assess risk",
                "selective_options": {
                    "risk": ["low", "medium", "high", "critical", "unknown"]
                },
            }
        )
        assert values["risk"] == expected
        assert diagnostics["distributions"]["risk"]["threshold_abstained"]
        assert diagnostics["distributions"]["risk"]["choice_mismatch"] == (
            selected == "high"
        )
    finally:
        client.close()


def test_native_question_instructions_cannot_be_trimmed_out_of_budget():
    client, requests = client_for()
    try:
        with pytest.raises(ContextInsufficientError):
            client._request(
                state={"prompt": "short"},
                questions={
                    "risk": {
                        "type": "choice",
                        "instructions": "x" * 110000,
                        "criteria": {"low": "routine", "high": "hazard"},
                    }
                },
            )
        assert requests == []
    finally:
        client.close()


def test_native_budget_translates_source_paths_and_preserves_human_restrictions(
    monkeypatch,
):
    client, requests = client_for()
    monkeypatch.setattr(
        client, "input_capacity", lambda: Capacity(unknown_bytes_limit=9000)
    )
    sources = {
        "sources": [
            {
                "ref": "human",
                "origin": "recorded_user_event",
                "content": "Keep data private",
            },
            {"ref": "context", "origin": "inherited_context", "content": "x" * 20000},
        ]
    }
    original = copy.deepcopy(sources)
    try:
        with source_sections(
            [(("state", "tool_call", "context", "raw_task_sources"), None, None)]
        ):
            client.evaluate_tool_call(
                state={"context": {"raw_task_sources": sources}},
                evaluation_rules="Bounded task",
            )
        body = json.loads(requests[0].content)
        data = body["input"][0]["content"][1]["data"]
        retained = data["tool_call"]["context"]["raw_task_sources"]
        assert retained["sources"][0]["content"] == "Keep data private"
        assert retained["sources"][1]["render_truncated"]
        assert client.input_view["digest"] == measure(body, jev=True)["digest"]
        assert client.input_view["source_coverage"] == "partial"
        assert sources == original
    finally:
        client.close()


def test_native_partial_dynamic_input_cannot_establish_affirmative_grants(monkeypatch):
    client, requests = client_for()
    monkeypatch.setattr(
        client, "input_capacity", lambda: Capacity(unknown_bytes_limit=9000)
    )
    try:
        client.classify_hybrid(
            state={
                "prompt": "x" * 30000,
                "selective_options": {"risk": ["low", "high", "unknown"]},
            }
        )
        body = json.loads(requests[0].content)
        assert client.input_view["affirmative_grants_eligible"] is False
        assert client.input_view["source_coverage"] == "partial"
        assert client.input_view["digest"] == measure(body, jev=True)["digest"]
    finally:
        client.close()


def test_native_structural_input_limit_prevents_dispatch():
    client, requests = client_for()
    value = "leaf"
    for _ in range(33):
        value = {"nested": value}
    try:
        with pytest.raises(ContextInsufficientError):
            client.check_alignment(
                parent_intention=value, child_intention="docs", evaluation_rules="rules"
            )
        assert requests == []
    finally:
        client.close()


def test_explicit_config_does_not_override_auto_or_shared_credentials(monkeypatch):
    monkeypatch.delenv("COGNIS_INFERENCE_URL", raising=False)
    monkeypatch.delenv("COGNIS_INFERENCE_TOKEN_FILE", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "chat-test-key")
    monkeypatch.setenv("DECISIONS_API_KEY", "native-test-key")
    monkeypatch.delenv("DECISIONS_L1_REVIEW_ENABLED", raising=False)
    monkeypatch.setenv("EVALUATOR_BACKEND", "llm")
    assert load_config().decisions.api_key == "native-test-key"
    monkeypatch.setenv("EVALUATOR_BACKEND", "decisions")
    config = load_config()
    assert config.decisions.l1_review_enabled is False
    monkeypatch.setenv("DECISIONS_L1_REVIEW_ENABLED", "true")
    assert load_config().decisions.l1_review_enabled is True
    assert config.decisions.model == "gpt-6-luna"
    assert not config.jev.enabled
    monkeypatch.setenv("DECISIONS_MODEL", "compatible-future-model")
    assert load_config().decisions.model == "compatible-future-model"
    monkeypatch.delenv("DECISIONS_API_KEY")
    with pytest.raises(ValueError, match="DECISIONS_API_KEY"):
        load_config()


@pytest.mark.parametrize("version", ["legacy", "hybrid_v1"])
@pytest.mark.parametrize("refuse", [False, True])
def test_production_evaluation_persists_native_assessment_or_failure(
    tmp_path, monkeypatch, version, refuse
):
    monkeypatch.setenv("HYBRID_CONTEXT_MODE", "mixed")
    labels = {
        "relationship": "supports_intent",
        "operation": "write",
        "risk": "low",
        "authorization": "explicit",
        "authorization_source": "session_intention",
        "authorization_scope": "within_scope",
        "environment": "development",
        "side_effect_scope": "bounded",
        "reversibility": "easy",
        "data_sensitivity": "ordinary",
        "exposure": "none",
        "mutation": "mutating",
        "consequence": "routine",
        "contextual_restrictions": "consistent",
        "target_scope": "consistent",
    }
    client, requests = client_for(
        labels=labels,
        mutate=(lambda p: p["answers"][0].update(type="refusal")) if refuse else None,
    )
    backend = EvaluatorBackend("decisions", decisions_factory=lambda: client)
    db = Database(DBConfig(backend="sqlite", path=str(tmp_path / "native.db")))
    sessions, audit = SessionStore(db), AuditStore(db)
    llm = MagicMock()
    llm.generate.side_effect = AssertionError("Unexpected generative assessment")
    sessions.create(
        user_id="owner",
        session_id="work",
        intention="Update the usage guide",
        policy={"evaluation_policy_version": version},
    )
    evaluator = Evaluator(
        llm=llm, evaluator_backend=backend, session_store=sessions, audit_store=audit
    )
    try:
        result = evaluator.evaluate(
            user_id="owner",
            session_id="work",
            agent_id=None,
            tool="write",
            args={
                "file_path": "/repo/usage.md",
                "content": "bounded update",
                "password": "synthetic-private-password",
            },
        )
        assert requests
        if refuse:
            assert result["decision"] != "approve"
            assert "decisions_response_invalid" in json.dumps(result)
        else:
            assert "decisions_response_invalid" not in json.dumps(result)
        saved = audit.get_by_call_id(result["call_id"], user_id="owner")
        assert saved["decision"] == result["decision"]
        if not refuse:
            assert "gpt-6-luna" in json.dumps(saved)
        assert "native-test-key" not in json.dumps(saved)
        assert "synthetic-private-password" not in json.dumps(saved)
        assert all(
            "synthetic-private-password" not in request.content.decode()
            for request in requests
        )
    finally:
        backend.close()
        db.close()


@pytest.mark.parametrize(
    "path", ["static_mutation", "opaque_overlay", "alignment_overlay"]
)
def test_production_overlays_persist_native_review_and_restricted_views(tmp_path, path):
    client, requests = client_for(
        labels={"mutation": "read_only"},
        mutate=lambda payload: payload["answers"][0].update(
            confidence=0.1 if payload["answers"][0]["type"] == "choice" else 1.0,
            **(
                {"probability": 0.7}
                if payload["answers"][0]["type"] == "predicate"
                else {}
            ),
        ),
    )
    backend = EvaluatorBackend.fixed(client)
    db = Database(DBConfig(backend="sqlite", path=str(tmp_path / "overlay.db")))
    sessions, audit = SessionStore(db), AuditStore(db)
    llm = MagicMock()
    model_metadata = "sk-SyntheticMetadataSecret1234567890"
    llm._model = model_metadata
    llm.input_view = {"source_coverage": "full", "affirmative_grants_eligible": True}

    def chat_answer(messages, *, json_schema, **kwargs):
        if json_schema["name"] == "safety_evaluation":
            return json.dumps(
                {
                    "aligned": True,
                    "risk": "low",
                    "decision": "approve",
                    "reasoning": "Bounded task",
                }
            )
        return Assessment(
            mutation="read_only", risk="low", relationship="supports_intent"
        ).model_dump_json()

    llm.generate.side_effect = chat_answer
    session_id = "work"
    if path == "alignment_overlay":
        sessions.create(
            user_id="owner",
            session_id="parent",
            intention="Inspect local files",
            policy={"evaluation_policy_version": "legacy"},
        )
        session_id = "child"
    sessions.create(
        user_id="owner",
        session_id=session_id,
        intention="Inspect local files",
        policy={"evaluation_policy_version": "legacy"},
        **({"parent_session_id": "parent"} if path == "alignment_overlay" else {}),
    )
    evaluator = Evaluator(
        llm=llm, evaluator_backend=backend, session_store=sessions, audit_store=audit
    )
    if path == "alignment_overlay":
        evaluator._alignment_barrier = MagicMock()
        evaluator._alignment_barrier.is_misaligned.return_value = (
            "Child requires independent approval"
        )
    try:
        result = evaluator.evaluate(
            user_id="owner",
            session_id=session_id,
            agent_id=None,
            tool="write" if path == "static_mutation" else "service:inspect",
            args={"file_path": "/repo/notes.md", "content": "bounded"}
            if path == "static_mutation"
            else {},
            context={
                "chat_mode": "plan",
                "read_only_required": True,
                "supplementary_context": "x" * 150000,
            },
        )
        saved = audit.get_by_call_id(result["call_id"], user_id="owner")
        metadata = saved["evaluation_metadata"]
        assert metadata["backend_provenance"]["route"] == "decisions_to_llm"
        assert model_metadata not in json.dumps(saved)
        assert "[REDACTED:api_key]" in json.dumps(metadata["backend_provenance"])
        assert llm._model == model_metadata
        assert metadata["model_view"]["affirmative_grants_eligible"] is False
        assert metadata["model_view"]["source_coverage"] == "partial"
        assert (
            metadata["backend_provenance"]["attempts"][0]["model_view"]["digest"]
            == measure(json.loads(requests[0].content), jev=True)["digest"]
        )
        assert saved["decision"] == "deny"
        if path != "static_mutation":
            secondary = next(
                stage
                for stage in metadata["stages"]
                if stage["stage"].startswith("read_only")
            )
            assert secondary["backend_provenance"]["route"] == "decisions_to_llm"
            assert secondary["model_view"]["affirmative_grants_eligible"] is False
            assert metadata["mutation"] == "unclear"
        if path == "opaque_overlay":
            # Top-level identity remains the primary assessment, not the
            # independently budgeted secondary read-only proof.
            assert len(requests) == 2
            assert (
                metadata["stages"][0]["backend_provenance"]["attempts"][0][
                    "model_view"
                ]["digest"]
                == measure(json.loads(requests[1].content), jev=True)["digest"]
            )
        assert sessions.get(session_id, user_id="owner")["total_calls"] == 1
    finally:
        backend.close()
        db.close()


@pytest.mark.parametrize("version", ["legacy", "hybrid_v1"])
def test_provider_metadata_is_redacted_on_persist_without_key_or_reference_loss(
    tmp_path, version
):
    secret = "sk-SyntheticMetadataSecret1234567890"
    extra = {
        "provider_note": secret,
        "api_key": "synthetic-metadata-key-not-pattern",
        "model": secret,
        "authorization": "explicit",
        "source_ref": "source.1",
        "grant_reference": "grant.1",
        "policy_hash": "a" * 64,
    }
    original = copy.deepcopy(extra)
    client, _ = client_for()
    captured = []
    if version == "legacy":
        native = client.evaluate_tool_call

        def evaluate_native(**kwargs):
            result = native(**kwargs)
            result.metadata.update(extra)
            captured.append((result.metadata, copy.deepcopy(result.metadata)))
            return result

        client.evaluate_tool_call = evaluate_native
    else:
        native = client.classify_hybrid

        def classify_native(**kwargs):
            result, diagnostics = native(**kwargs)
            diagnostics.update(extra)
            captured.append((diagnostics, copy.deepcopy(diagnostics)))
            return result, diagnostics

        client.classify_hybrid = classify_native
    backend = EvaluatorBackend.fixed(client)
    db = Database(DBConfig(backend="sqlite", path=str(tmp_path / "metadata.db")))
    sessions, audit = SessionStore(db), AuditStore(db)
    sessions.create(
        user_id="owner",
        session_id="work",
        intention="Update local documentation",
        policy={"evaluation_policy_version": version},
    )
    evaluator = Evaluator(
        llm=MagicMock(),
        evaluator_backend=backend,
        session_store=sessions,
        audit_store=audit,
    )
    args = {
        "file_path": "/repo/notes.md",
        "content": "bounded change",
        "password": "synthetic-password",
    }
    args_original = copy.deepcopy(args)
    try:
        result = evaluator.evaluate(
            user_id="owner", session_id="work", agent_id=None, tool="write", args=args
        )
        saved = audit.get_by_call_id(result["call_id"], user_id="owner")
        assert secret not in json.dumps(saved)
        assert "synthetic-password" not in json.dumps(saved)
        metadata = (
            saved["evaluation_metadata"]
            if version == "legacy"
            else saved["evaluation_metadata"]["stages"][0]
        )
        assert metadata["provider_note"] == "[REDACTED:api_key]"
        assert metadata["api_key"] == "[REDACTED:credential]"
        for key in ("authorization", "source_ref", "grant_reference", "policy_hash"):
            assert metadata[key] == extra[key]
        assert all(value == snapshot for value, snapshot in captured)
        assert extra == original and args == args_original
    finally:
        backend.close()
        db.close()


def test_native_hybrid_questions_describe_fields_and_every_choice():
    options = {
        "relationship": ["directly_serves_intent", "supports_intent", "unclear"],
        "target_scope": ["consistent", "conflicts", "unresolved"],
        "target_reference": ["none", "proposed action", "target 1"],
        "restriction_reference": ["none", "session assignment"],
        "effect_transmit": ["yes", "no", "unknown"],
        "risk": ["low", "medium", "high", "critical", "unknown"],
    }
    client, requests = client_for()
    try:
        client.classify_hybrid(state={"prompt": "Review", "selective_options": options})
        questions = {q["name"]: q for q in json.loads(requests[0].content)["questions"]}
        assert set(questions) == set(options)
        for name, labels in options.items():
            choices = questions[name]["choices"]
            assert [row["value"] for row in choices] == labels
            for row in choices:
                assert row["description"] != row["value"].replace("_", " ")
            # The shared factual-label contract follows the field definition.
            assert questions[name]["instructions"].endswith(
                "Return a factual label, never a permission verdict."
            )
        assert (
            "only when target_scope is conflicts"
            in (questions["target_reference"]["instructions"])
        )
        assert (
            "'target 1'" in questions["target_reference"]["choices"][2]["description"]
        )
        assert (
            "uploads and remote/target overrides"
            in (questions["effect_transmit"]["instructions"])
        )
    finally:
        client.close()


def test_base_hybrid_questions_hook_keeps_provider_wording():
    from intaris.finite_questions import FiniteQuestionClient
    from intaris.jev import JevClient

    assert JevClient._hybrid_questions is FiniteQuestionClient._hybrid_questions
    questions = {"risk": {"type": "choice", "instructions": "x", "criteria": {}}}
    assert FiniteQuestionClient._hybrid_questions(None, questions) is questions
