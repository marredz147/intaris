"""Actual app/barrier/client wiring; HTTP provider responses are entirely offline."""

import copy
import json
import uuid
from contextlib import ExitStack, closing, contextmanager

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.test_api import _reset_server_globals, env_no_auth, tmp_db  # noqa: F401

CANARY = "FULL_ORIGINAL_TRANSPORT_CANARY"


def assert_strict_wire_objects(node):
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node.get("properties", {}))
        for child in node.values():
            assert_strict_wire_objects(child)
    elif isinstance(node, list):
        for child in node:
            assert_strict_wire_objects(child)


def test_extraction_wire_changes_only_object_closure_not_internal_defaults():
    from intaris.task_context import Extraction, Item, extraction_wire_schema

    original = Extraction.model_json_schema()
    wire = extraction_wire_schema()

    def without_closure(value):
        if isinstance(value, dict):
            return {
                key: without_closure(child)
                for key, child in value.items()
                if key not in {"required", "additionalProperties"}
            }
        if isinstance(value, list):
            return [without_closure(child) for child in value]
        return value

    assert without_closure(wire) == without_closure(original)
    assert "scope_refs" not in original["$defs"]["Item"]["required"]
    assert_strict_wire_objects(wire)
    assert (
        Item(
            kind="request",
            horizon="short",
            explicitness="explicit",
            source_ref="session_intention",
            start=0,
            end=1,
        ).scope_refs
        == []
    )
    assert Extraction.model_json_schema() == original


VALUES = {
    "relationship": "supports_intent",
    "operation": "write",
    "risk": "high",
    # Implicit keeps the Judge path under test; an explicit user grant would
    # approve before the Judge (covered in tests/test_rc5c_fixes.py).
    "authorization": "implicit",
    "authorization_source": "latest_user_message",
    "authorization_scope": "within_scope",
    "environment": "development",
    "side_effect_scope": "bounded",
    "reversibility": "easy",
    "data_sensitivity": "ordinary",
    "exposure": "none",
    "mutation": "mutating",
    "effects": ["write"],
    "consequence": "consequential",
    "contextual_restrictions": "consistent",
    "restriction_reference": "latest_user_message",
    "target_scope": "consistent",
    "target_reference": "action",
}


@contextmanager
def _isolated_postgres(url):
    """Each app owns its queue; another test's worker must not claim its tasks."""
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn

    name = "source_service_" + uuid.uuid4().hex
    with closing(psycopg2.connect(url)) as connection, connection.cursor() as admin:
        connection.autocommit = True
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            yield make_dsn(url, dbname=name)
        finally:
            admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))


@pytest.fixture
def service_databases():
    with ExitStack() as stack:
        yield lambda url: stack.enter_context(_isolated_postgres(url))


@pytest.fixture(params=["llm", "jev", "decisions"])
def service(request, env_no_auth, monkeypatch, tmp_path, service_databases):  # noqa: F811
    import intaris.server as server

    options = (
        request.param
        if isinstance(request.param, tuple)
        else (request.param, "source_context")
    )
    backend, mode = options[:2]
    generation = options[2] if len(options) >= 3 else "spans"
    # Existing renderer assertions characterize JSON; projection acceptance opts in.
    projection = options[4] if len(options) >= 5 else "json"
    if len(options) >= 4 and options[3] == "postgresql":
        import os

        url = os.environ.get("TEST_HYBRID_POSTGRES_URL")
        if not url:
            pytest.skip("TEST_HYBRID_POSTGRES_URL is not configured")
        monkeypatch.setenv("DB_BACKEND", "postgresql")
        monkeypatch.setenv("DATABASE_URL", service_databases(url))
    for key, value in {
        "HYBRID_CONTEXT_MODE": mode,
        "L1_INPUT_PROJECTION": projection,
        "SOURCE_CONTEXT_GENERATION": generation,
        "EVALUATOR_BACKEND": backend,
        "LLM_MODEL": options[5] if len(options) >= 6 else "openai/gpt-oss-20b",
        "LLM_BASE_URL": "https://source-context-provider.test/v1",
        "ANALYSIS_LLM_API_KEY": "offline-test-token",
        "ANALYSIS_LLM_BASE_URL": "https://source-context-provider.test/v1",
        "ANALYSIS_LLM_MODEL": "openai/gpt-oss-120b",
        "JUDGE_LLM_API_KEY": "offline-test-token",
        "JUDGE_LLM_BASE_URL": "https://source-context-provider.test/v1",
        "JUDGE_LLM_MODEL": "openai/gpt-oss-120b",
        "JUDGE_MODE": "advisory",
        "JEV_API_KEY": "offline-test-token",
        "JEV_BASE_URL": "https://source-context-jev.test",
        "JEV_MODEL": "jev-1.13.0",
        "DECISIONS_API_KEY": "offline-decisions-token",
        "DECISIONS_BASE_URL": "https://source-context-decisions.test/v1",
        "EVENT_STORE_ENABLED": "true",
        "EVENT_STORE_BACKEND": "filesystem",
        "EVENT_STORE_PATH": str(tmp_path / "events"),
    }.items():
        monkeypatch.setenv(key, value)
    wire = []
    hooks = {}

    def respond(transport, req):
        assert req.url.host in {
            "source-context-provider.test",
            "source-context-jev.test",
            "source-context-decisions.test",
        }, req.url
        if req.method == "GET":
            return httpx.Response(
                200,
                json={
                    "id": req.url.path.rsplit("/", 1)[-1],
                    "context_window": 262144,
                    "max_input_tokens": 240000,
                    "max_output_tokens": 16000,
                },
            )
        body = json.loads(req.content)
        if "questions" in body:
            native = isinstance(body["questions"], list)
            wire.append(("l1-decisions" if native else "l1-jev", body))
            answers = {}
            questions = (
                ((question["name"], question) for question in body["questions"])
                if native
                else body["questions"].items()
            )
            for field, question in questions:
                if native and question["type"] == "predicate":
                    answers[field] = {"type": "predicate", "probability": 0.99}
                    continue
                choices = (
                    [row["value"] for row in question["choices"]]
                    if native
                    else question["criteria"]
                )
                value = VALUES.get(field, "approve" if field == "decision" else None)
                value = (
                    {
                        "latest_user_message": "latest user message",
                        "session_intention": "session assignment",
                        "action": "proposed action",
                    }.get(value, value)
                    if value not in choices
                    else value
                )
                if field.startswith("effect_"):
                    value = "yes" if field == "effect_write" else "no"
                if field.endswith("_reference") or field == "authorization_source":
                    if value not in choices:
                        value = "none"
                assert value in choices, (field, value, choices)
                answers[field] = {
                    "type": "choice",
                    "choice": value,
                    "confidence": 1,
                    "probabilities": {
                        option: float(option == value) for option in choices
                    },
                }
            if native:
                assert req.url.path == "/v1/decisions"
                assert req.headers["Authorization"] == "Bearer offline-decisions-token"
                answers = [
                    {
                        **answer,
                        "name": field,
                        **(
                            {
                                "probabilities": [
                                    {"value": value, "probability": probability}
                                    for value, probability in answer[
                                        "probabilities"
                                    ].items()
                                ]
                            }
                            if "probabilities" in answer
                            else {}
                        ),
                    }
                    for field, answer in answers.items()
                ]
                if hooks.get("decisions_answers"):
                    hooks["decisions_answers"](answers)
                if hooks.get("decisions_timeout"):
                    raise httpx.ReadTimeout("offline timeout", request=req)
            return httpx.Response(
                hooks.get("decisions_status", 200) if native else 200,
                json={
                    "model": body["model"],
                    "answers": answers,
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                        **(
                            {"input_tokens_details": hooks["decisions_usage_details"]}
                            if native and "decisions_usage_details" in hooks
                            else {}
                        ),
                    },
                },
            )
        schema = (body.get("response_format") or {}).get("json_schema", {})
        name = schema.get("name")
        if name == "task_context_extraction":
            # Actual serialized HTTP request from LLMClient/OpenAI SDK, not an
            # assumption about the raw Pydantic schema or a mocked generate().
            assert body["response_format"]["type"] == "json_schema"
            assert schema["strict"] is True
            assert_strict_wire_objects(schema["schema"])
            if generation == "spans":
                item = schema["schema"]["$defs"]["Item"]
                assert "scope_refs" in item["required"]
            else:
                item = schema["schema"]["$defs"][
                    "InterpretedItem" if generation == "coherent" else "QuoteItem"
                ]
                assert "scope_refs" not in item["properties"]
                assert "start" not in item["properties"]
            wire.append(("extract", body))
            if hooks.get("before_extract"):
                hooks["before_extract"](body)
            sources = json.loads(body["messages"][1]["content"])["sources"]
            source = next(
                row for row in sources if row["origin"] == "client_user_claim"
            )
            answer = {
                "items": [
                    {
                        "kind": "request",
                        "horizon": "short",
                        "explicitness": "explicit",
                        "source_ref": source["ref"],
                        "start": 0,
                        "end": min(24, len(source["content"])),
                        "scope_refs": source["scope_refs"],
                    }
                ]
            }
            if generation in {"quotes", "coherent"}:
                answer["items"][0] = {
                    "kind": "context",
                    "horizon": "short",
                    "explicitness": "explicit",
                    "source_ref": source["ref"],
                    "quote": source["content"],
                }
            if generation == "coherent":
                # Mechanical echo only, never an ideal-goal oracle.
                answer["items"][0].update(
                    temporal_status="uncertain",
                    polarity="unknown",
                    reference_resolution="missing",
                    relation="uncertain",
                    related_source_refs=[],
                )
                answer["current_request"] = {"status": "unknown", "item_indices": []}
            if hooks.get("extract_answer"):
                answer = hooks["extract_answer"](sources)
        elif name == "user_simulator":
            wire.append(("judge", body))
            payload = json.loads(body["messages"][1]["content"])
            refs = [int(row.split("]", 1)[0][1:]) for row in payload["originals"]]
            answer = {
                "user_would": "approve" if refs else "ask",
                "risk": "low",
                "confidence": "high",
                "basis": "explicit_request",
                "cites": refs[:1],
                "question": "" if refs else "Confirm this action?",
                "action_summary": "Perform the fixture's proposed bounded action.",
                "explanation": "The captured user source supports this fixture action.",
            }
        elif name == "authorization_support":
            wire.append(("authorization", body))
            payload = json.loads(body["messages"][1]["content"])
            sources = [
                row
                for row in payload["sources"]
                if row["index"] in payload["grant_citable"]
            ]
            answer = {
                "outcome": "supported" if sources else "unresolved",
                "strength": "routine" if sources else "none",
                "support": [
                    {"source": row["index"], "quote": row["text"]}
                    for row in sources[:1]
                ],
                "explanation": "The fixture action supports its recorded task.",
                "decision_status": "none" if sources else "unresolved",
                "missing_user_decision": "",
            }
        elif name in {"hybrid_selective_assessment", "hybrid_obligation_review"}:
            judge = name == "hybrid_obligation_review"
            wire.append(("judge" if judge else "l1-gpt", body))
            properties = schema["schema"]["properties"]
            fields = properties["classification"]["properties"] if judge else properties
            labels = {
                field: "" if field == "restriction_quote" else VALUES[field]
                for field in fields
            }
            for field, value in list(labels.items()):
                if (
                    field
                    in {
                        "authorization_source",
                        "restriction_reference",
                        "target_reference",
                    }
                    and value not in fields[field]["enum"]
                ):
                    labels[field] = {
                        "latest_user_message": "latest user message",
                        "session_intention": "session assignment",
                        "action": "proposed action",
                    }[value]
                    if labels[field] not in fields[field]["enum"]:
                        labels[field] = "none"
            answer = labels
            if not judge and hooks.get("l1_answer"):
                answer = hooks["l1_answer"](labels)
            if judge:
                answer = {
                    "classification": labels,
                    "obligations": {
                        identifier: {
                            "status": "consistent",
                            "reference": "latest_user_message",
                        }
                        for identifier in properties["obligations"]["properties"]
                    },
                }
        elif name == "safety_evaluation":
            wire.append(("l1-gpt", body))
            answer = {
                "aligned": True,
                "risk": "low",
                "reasoning": "The original evidence supports this bounded task.",
                "decision": "approve",
            }
            if hooks.get("l1_answer"):
                answer = hooks["l1_answer"](answer)
        elif name == "session_intention":
            wire.append(("intention", body))
            assert body["response_format"]["type"] == "json_schema"
            assert schema["strict"] is True
            assert_strict_wire_objects(schema["schema"])
            sources = json.loads(body["messages"][1]["content"].splitlines()[1])[
                "sources"
            ]
            assert body["max_tokens"] == min(
                8192,
                max(4096, 2048 + (sum(len(row["text"]) for row in sources) + 3) // 4),
            )
            source = next(row for row in reversed(sources) if row["text"])
            answer = {
                "title": "Local work",
                "intention": "Perform the requested local repository work.",
                "items": [
                    {
                        "kind": "approval" if source["text"] == "yes" else "request",
                        "quote": source["text"],
                        "msg": source["msg"],
                        "status": "current",
                        "confirmation": {
                            "form": "single_permission",
                            "actions": [source["preceding_assistant"]],
                        }
                        if source.get("preceding_assistant")
                        and len(source["preceding_assistant"]) <= 240
                        else None,
                    }
                ],
                "current": [] if source["text"] == "yes" else [0],
            }
            if hooks.get("intention_answer"):
                answer = hooks["intention_answer"](sources)
        elif not name:
            wire.append(("intention", body))
            answer = {
                "title": "Local work",
                "intention": "Perform the requested local repository work.",
            }
        else:
            raise AssertionError(f"Unexpected provider route: {name}")
        return httpx.Response(
            hooks.get("l1_status", 200)
            if name in {"safety_evaluation", "hybrid_selective_assessment"}
            else 200,
            json={
                "id": str(uuid.uuid4()),
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": json.dumps(answer)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    # Intercept below BOTH SDKs. No LLMClient/JevClient/barrier/evaluator mocks.
    def checked(transport, req):
        try:
            return respond(transport, req)
        except Exception as exc:
            wire.append(("mock-error", {"error": repr(exc)}))
            raise

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", checked)
    app = server.create_app()
    app.state.offline_provider_hooks = hooks
    with TestClient(
        app, headers={"X-User-Id": "default", "X-Agent-Id": "offline-agent"}
    ) as client:
        yield client, app, wire


def create(client, *, profile="balanced"):
    sid = str(uuid.uuid4())
    response = client.post(
        "/api/v1/intention",
        json={
            "session_id": sid,
            "intention": "Inspect the local repository",
            "policy": {"evaluation_policy_version": "hybrid_v1", "profile": profile},
        },
    )
    assert response.status_code == 200, response.text
    return sid


def record(client, sid, *events):
    response = client.post(f"/api/v1/session/{sid}/events", json=list(events))
    assert response.status_code == 200, response.text


def reason(client, sid):
    response = client.post(
        "/api/v1/reasoning", json={"session_id": sid, "from_events": True}
    )
    assert response.status_code == 200, response.text


def evaluate(client, sid, **extra):
    response = client.post(
        "/api/v1/evaluate",
        json={
            "session_id": sid,
            "tool": "write",
            "args": {"file_path": "/repo/parser.py", "content": "bounded patch"},
            **extra,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("service", [("decisions", "mixed")], indirect=True)
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("temporary", [429, 503, 529, "timeout"])
def test_temporary_native_errors_review_original_evidence_at_l1(
    service, legacy, temporary, monkeypatch
):
    from intaris import model_retry

    client, app, wire = service
    monkeypatch.setattr(model_retry, "backoff_delay", lambda *args: 0)
    hooks = app.state.offline_provider_hooks
    hooks["decisions_timeout" if temporary == "timeout" else "decisions_status"] = (
        True if temporary == "timeout" else temporary
    )
    hooks["l1_answer"] = lambda labels: {
        **labels,
        "risk": "low",
        **({"consequence": "routine"} if "consequence" in labels else {}),
    }
    sid = str(uuid.uuid4())
    response = client.post(
        "/api/v1/intention",
        json={
            "session_id": sid,
            "intention": "CANONICAL_TRANSIENT_EVIDENCE",
            "policy": {
                "evaluation_policy_version": "legacy" if legacy else "hybrid_v1"
            },
        },
    )
    assert response.status_code == 200
    result = evaluate(client, sid)
    assert result["decision"] == "approve", result
    provenance = result["evaluation_metadata"]["backend_provenance"]
    assert provenance["fallback_reason"]["code"] == "temporary_unavailable"
    assert [row["status"] for row in provenance["attempts"]] == ["failed", "completed"]
    assert provenance["final_backend"] == "llm"
    native = next(body for stage, body in wire if stage == "l1-decisions")
    chat = [body for stage, body in wire if stage == "l1-gpt"]
    assert len(chat) == 1 and "CANONICAL_TRANSIENT_EVIDENCE" in json.dumps(chat[0])
    from intaris.input_budget import measure

    assert (
        provenance["attempts"][0]["model_view"]["digest"] == measure(native)["digest"]
    )
    saved = client.get(f"/api/v1/audit/{result['call_id']}").json()
    assert saved["evaluation_metadata"]["backend_provenance"] == provenance
    assert client.get(f"/api/v1/session/{sid}").json()["total_calls"] == 1


@pytest.mark.parametrize("service", [("decisions", "mixed")], indirect=True)
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_optional_native_escalation_review_is_policy_owned(service, legacy, enabled):
    from intaris import server

    client, app, wire = service
    server._get_evaluator_backend().select().l1_review_enabled = enabled
    server._get_config().decisions.l1_review_enabled = enabled
    hooks = app.state.offline_provider_hooks

    def escalate(answers):
        for row in answers:
            if row["name"] == "decision":
                row["choice"] = "escalate"
                for probability in row["probabilities"]:
                    probability["probability"] = float(
                        probability["value"] == "escalate"
                    )

    hooks["decisions_answers"] = escalate
    hooks["l1_answer"] = lambda labels: {
        **labels,
        "risk": "low",
        **({"consequence": "routine"} if "consequence" in labels else {}),
    }
    sid = str(uuid.uuid4())
    response = client.post(
        "/api/v1/intention",
        json={
            "session_id": sid,
            "intention": "Inspect the local repository",
            "policy": {
                "evaluation_policy_version": "legacy" if legacy else "hybrid_v1"
            },
        },
    )
    assert response.status_code == 200
    result = evaluate(client, sid)
    provenance = result["evaluation_metadata"]["backend_provenance"]
    assert len([body for stage, body in wire if stage == "l1-gpt"]) == int(enabled)
    if enabled:
        assert result["decision"] == "approve"
        assert provenance["fallback_reason"]["code"] == "optional_escalation"
    else:
        assert provenance["route"] == "decisions"
    saved = client.get(f"/api/v1/audit/{result['call_id']}").json()
    assert saved["evaluation_metadata"]["backend_provenance"] == provenance
    assert (
        client.get("/api/v1/config").json()["evaluator"]["l1_review_enabled"] == enabled
    )


@pytest.mark.parametrize("service", [("decisions", "mixed")], indirect=True)
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "case", ["low_confidence", "critical", "static_deny", "review_failure"]
)
def test_optional_review_never_duplicates_low_review_or_relaxes_critical(
    service, legacy, case
):
    from intaris import server

    client, app, wire = service
    server._get_evaluator_backend().select().l1_review_enabled = True
    hooks = app.state.offline_provider_hooks

    def native_answers(answers):
        if case == "low_confidence":
            low_native_confidence(answers)
        for answer in answers:
            if answer["name"] == "decision" or (
                case == "critical" and answer["name"] == "risk"
            ):
                label = "critical" if answer["name"] == "risk" else "escalate"
                answer["choice"] = label
                for probability in answer["probabilities"]:
                    probability["probability"] = float(probability["value"] == label)

    hooks["decisions_answers"] = native_answers
    if case == "review_failure":
        hooks["l1_status"] = 401
    else:
        hooks["l1_answer"] = lambda labels: {
            **labels,
            "risk": "low",
            **({"consequence": "routine"} if "consequence" in labels else {}),
        }
    sid = str(uuid.uuid4())
    policy = {
        "evaluation_policy_version": "legacy" if legacy else "hybrid_v1",
        **({"deny_tools": ["write"]} if case == "static_deny" else {}),
    }
    assert (
        client.post(
            "/api/v1/intention",
            json={
                "session_id": sid,
                "intention": "Inspect local repository",
                "policy": policy,
            },
        ).status_code
        == 200
    )
    result = evaluate(client, sid)
    reviews = [body for stage, body in wire if stage == "l1-gpt"]
    if case in {"critical", "static_deny"}:
        if case == "static_deny" or legacy:
            assert result["decision"] == "deny", result
        else:
            # Upstream independently reviews model-predicted critical risk;
            # this toggle must not add a generative L1 correction of its own.
            assert (
                result["evaluation_metadata"]["backend_provenance"]["route"]
                == "decisions"
            )
        assert not reviews
    else:
        assert len(reviews) == 1
        provenance = result["evaluation_metadata"]["backend_provenance"]
        assert provenance["fallback_reason"]["code"] == (
            "low_confidence" if case == "low_confidence" else "optional_escalation"
        )
        if case == "review_failure":
            if legacy:
                assert result["decision"] != "approve"
            assert provenance["final_backend"] == "decisions"
            assert provenance["attempts"][-1]["status"] == "failed"
    assert client.get(f"/api/v1/session/{sid}").json()["total_calls"] == 1


@pytest.mark.parametrize("service", [("decisions", "mixed")], indirect=True)
@pytest.mark.parametrize(
    "invalid",
    [
        {"aligned": "false"},
        {"reasoning": []},
        {"risk": "unknown"},
        {"decision": "allowed"},
        {"reasoning": None},
    ],
)
def test_legacy_optional_review_rejects_malformed_fields_and_retains_native(
    service, invalid
):
    from intaris import server

    client, app, wire = service
    server._get_evaluator_backend().select().l1_review_enabled = True

    def native_escalation(answers):
        answer = next(row for row in answers if row["name"] == "decision")
        answer["choice"] = "escalate"
        for row in answer["probabilities"]:
            row["probability"] = float(row["value"] == "escalate")

    def malformed(labels):
        result = {
            **labels,
            "risk": "low",
            "aligned": True,
            "decision": "approve",
            **invalid,
        }
        if invalid == {"reasoning": None}:
            result.pop("reasoning")
        return result

    app.state.offline_provider_hooks.update(
        decisions_answers=native_escalation, l1_answer=malformed
    )
    sid = str(uuid.uuid4())
    assert (
        client.post(
            "/api/v1/intention",
            json={
                "session_id": sid,
                "intention": "Inspect local files",
                "policy": {"evaluation_policy_version": "legacy"},
            },
        ).status_code
        == 200
    )
    result = evaluate(client, sid)
    assert result["decision"] != "approve"
    assert len([body for stage, body in wire if stage == "l1-gpt"]) == 1
    saved = client.get(f"/api/v1/audit/{result['call_id']}").json()
    provenance = saved["evaluation_metadata"]["backend_provenance"]
    assert provenance["final_backend"] == "decisions"
    assert provenance["fallback_reason"]["code"] == "optional_escalation"
    assert provenance["attempts"][-1]["status"] == "failed"
    assert (
        provenance["attempts"][-1]["diagnostics"]["failure"]["category"] == "ValueError"
    )


@pytest.mark.parametrize("service", [("decisions", "mixed")], indirect=True)
@pytest.mark.parametrize("trigger", ["low_confidence", "temporary"])
def test_legacy_mandatory_review_malformed_fields_fail_closed(
    service, trigger, monkeypatch
):
    from intaris import model_retry

    client, app, wire = service
    monkeypatch.setattr(model_retry, "backoff_delay", lambda *args: 0)
    hooks = app.state.offline_provider_hooks
    if trigger == "low_confidence":
        hooks["decisions_answers"] = low_native_confidence
    else:
        hooks["decisions_status"] = 503
    hooks["l1_answer"] = lambda labels: {
        **labels,
        "aligned": "false",
        "risk": "low",
        "decision": "approve",
    }
    sid = str(uuid.uuid4())
    assert (
        client.post(
            "/api/v1/intention",
            json={
                "session_id": sid,
                "intention": "Inspect local files",
                "policy": {"evaluation_policy_version": "legacy"},
            },
        ).status_code
        == 200
    )
    response = client.post(
        "/api/v1/evaluate",
        json={
            "session_id": sid,
            "tool": "write",
            "args": {"file_path": "/repo/notes.md", "content": "bounded"},
        },
    )
    assert response.status_code >= 500, response.text
    assert len([body for stage, body in wire if stage == "l1-gpt"]) == 1


@pytest.mark.parametrize("service", [("decisions", "mixed")], indirect=True)
def test_exhausted_shared_deadline_cannot_grant_or_start_new_l1_request(
    service, monkeypatch
):
    from types import SimpleNamespace

    from intaris import model_retry, server

    client, app, wire = service
    sid = create(client)
    now = [1000.0]
    monkeypatch.setattr(model_retry, "time", SimpleNamespace(monotonic=lambda: now[0]))
    server._get_config().evaluation_deadline_ms = 1000

    def exhausted(answers):
        low_native_confidence(answers)
        now[0] += 1.1

    app.state.offline_provider_hooks["decisions_answers"] = exhausted
    response = client.post(
        "/api/v1/evaluate",
        json={
            "session_id": sid,
            "tool": "write",
            "args": {"file_path": "/repo/parser.py", "content": "bounded"},
        },
    )
    assert response.status_code == 503, response.text
    assert not [body for stage, body in wire if stage == "l1-gpt"]


def low_native_confidence(answers):
    answer = next(row for row in answers if row["type"] == "choice")
    answer["confidence"] = 0.1


@pytest.mark.parametrize(
    "service", [("decisions", "mixed"), ("decisions", "source_context")], indirect=True
)
@pytest.mark.parametrize("legacy", [False, True])
def test_low_decisions_confidence_resolves_at_l1_with_one_audit_record(service, legacy):
    client, app, wire = service
    hooks = app.state.offline_provider_hooks
    hooks["decisions_answers"] = low_native_confidence
    hooks["decisions_usage_details"] = {"cached_tokens": 4, "cache_write_tokens": 6}
    hooks["l1_answer"] = lambda labels: {
        **labels,
        "risk": "low",
        **({"consequence": "routine"} if "consequence" in labels else {}),
    }
    if legacy:
        sid = str(uuid.uuid4())
        response = client.post(
            "/api/v1/intention",
            json={
                "session_id": sid,
                "intention": "Update the local usage guide",
                "policy": {"evaluation_policy_version": "legacy"},
            },
        )
        assert response.status_code == 200, response.text
    else:
        sid = create(client)
    result = evaluate(client, sid)
    assert result["decision"] == "approve", result
    assert not [body for stage, body in wire if stage == "judge"]
    native = [body for stage, body in wire if stage == "l1-decisions"]
    chat = [body for stage, body in wire if stage == "l1-gpt"]
    assert len(native) == len(chat) == 1
    provenance = result["evaluation_metadata"]["backend_provenance"]
    assert provenance["route"] == "decisions_to_llm"
    assert provenance["final_backend"] == "llm"
    assert provenance["final_model"] == "openai/gpt-oss-20b"
    assert provenance["fallback_reason"]["code"] == "low_confidence"
    assert [row["backend"] for row in provenance["attempts"]] == ["decisions", "llm"]
    assert provenance["attempts"][0]["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 6},
    }
    from intaris.input_budget import measure

    for row, body in zip(provenance["attempts"], [native[0], chat[0]], strict=True):
        assert row["model_view"]["digest"] == measure(body)["digest"]
    saved = client.get(f"/api/v1/audit/{result['call_id']}").json()
    assert saved["evaluation_metadata"]["backend_provenance"] == provenance
    records = client.get("/api/v1/audit", params={"session_id": sid}).json()
    assert (
        len([row for row in records["items"] if row["record_type"] == "tool_call"]) == 1
    )
    session = client.get(f"/api/v1/session/{sid}").json()
    assert session["total_calls"] == 1


@pytest.mark.parametrize("service", [("decisions", "source_context")], indirect=True)
def test_low_decisions_confidence_still_uses_judge_when_l1_requires_review(service):
    client, app, wire = service
    app.state.offline_provider_hooks["decisions_answers"] = low_native_confidence
    sid = create(client)
    record(
        client,
        sid,
        {
            "type": "user_message",
            "data": {"content": "Apply the bounded local parser patch."},
        },
    )
    reason(client, sid)
    result = evaluate(client, sid)
    assert any(stage == "l1-gpt" for stage, _ in wire)
    assert any(stage == "judge" for stage, _ in wire)
    assert result["evaluation_metadata"]["context_input"]["judge_invoked"]
    assert (
        result["evaluation_metadata"]["backend_provenance"]["route"]
        == "decisions_to_llm"
    )


@pytest.mark.parametrize("service", [("decisions", "mixed")], indirect=True)
def test_high_confidence_decisions_does_not_invoke_l1_llm(service):
    client, app, wire = service
    app.state.offline_provider_hooks["decisions_usage_details"] = {
        "cached_tokens": 8,
        "cache_write_tokens": 0,
    }
    sid = create(client)
    result = evaluate(client, sid)
    assert not [body for stage, body in wire if stage == "l1-gpt"]
    assert result["evaluation_metadata"]["backend_provenance"]["route"] == "decisions"
    assert len(result["evaluation_metadata"]["backend_provenance"]["attempts"]) == 1
    saved = client.get(f"/api/v1/audit/{result['call_id']}").json()
    usage = saved["evaluation_metadata"]["backend_provenance"]["attempts"][0]["usage"]
    assert usage == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 8, "cache_write_tokens": 0},
    }


@pytest.mark.parametrize("service", [("decisions", "mixed")], indirect=True)
@pytest.mark.parametrize(
    "failure", ["refusal", "malformed", "auth", "review_failure", "cache_usage"]
)
def test_decisions_and_l1_review_failures_do_not_grant_execution(service, failure):
    client, app, wire = service
    hooks = app.state.offline_provider_hooks
    if failure == "refusal":
        hooks["decisions_answers"] = lambda answers: answers[0].update(type="refusal")
    elif failure == "malformed":
        hooks["decisions_answers"] = lambda answers: answers[0].update(confidence=True)
    elif failure == "auth":
        hooks["decisions_status"] = 401
    elif failure == "cache_usage":
        hooks["decisions_usage_details"] = {"cached_tokens": True}
    else:
        hooks["decisions_answers"] = low_native_confidence
        hooks["l1_status"] = 401
    sid = create(client)
    response = client.post(
        "/api/v1/evaluate",
        json={
            "session_id": sid,
            "tool": "write",
            "args": {"file_path": "/repo/parser.py", "content": "patch"},
        },
    )
    assert response.status_code != 200 or response.json()["decision"] != "approve"
    if failure != "review_failure":
        assert not [body for stage, body in wire if stage == "l1-gpt"]


@pytest.mark.parametrize("service", [("decisions", "mixed")], indirect=True)
def test_decisions_review_shares_api_deadline_and_persists_provenance(
    service, monkeypatch
):
    from types import SimpleNamespace

    from intaris import model_retry, server

    client, app, wire = service
    sid = create(client)
    now = [1000.0]
    monkeypatch.setattr(model_retry, "time", SimpleNamespace(monotonic=lambda: now[0]))
    server._get_config().evaluation_deadline_ms = 1000

    def slow_native(answers):
        low_native_confidence(answers)
        now[0] += 0.8

    app.state.offline_provider_hooks["decisions_answers"] = slow_native
    app.state.offline_provider_hooks["l1_answer"] = lambda labels: {
        **labels,
        "risk": "low",
        **({"consequence": "routine"} if "consequence" in labels else {}),
    }
    timeouts = []
    native_timeouts = []
    original = httpx.HTTPTransport.handle_request

    def observe(transport, req):
        if req.url.path.endswith("/decisions"):
            native_timeouts.append(req.extensions["timeout"]["read"])
        if req.method == "POST" and req.url.path.endswith("chat/completions"):
            body = json.loads(req.content)
            if body["model"] == "openai/gpt-oss-20b":
                timeouts.append(req.extensions["timeout"]["read"])
        return original(transport, req)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", observe)
    result = evaluate(client, sid)
    assert result["decision"] == "approve"
    assert len(timeouts) == 1 and 0 < timeouts[0] <= 0.201
    assert len(native_timeouts) == 1 and 0 < native_timeouts[0] <= 0.501
    saved = client.get(f"/api/v1/audit/{result['call_id']}").json()
    assert (
        saved["evaluation_metadata"]["backend_provenance"]["route"]
        == "decisions_to_llm"
    )


def test_actual_service_barrier_and_both_provider_payloads(service):
    client, app, wire = service
    sid = create(client)
    record(
        client,
        sid,
        {
            "type": "assistant_message",
            "data": {"content": "Apply this local parser patch?"},
        },
        {
            "type": "user_message",
            "data": {"content": "yes; keep this supporting context private: " + CANARY},
        },
    )
    reason(client, sid)
    result = evaluate(client, sid)
    assert not [body for stage, body in wire if stage == "mock-error"]
    assert result["decision"] == "approve"
    metadata = result["evaluation_metadata"]
    artifact = metadata["task_context"]
    assert metadata["context_input"]["judge_invoked"]
    assert any(stage == "intention" for stage, _ in wire)  # Actual IntentionBarrier.
    assert app.state.intention_barrier is not None
    l1 = [body for stage, body in wire if stage.startswith("l1-")]
    assert l1
    for body in l1:
        serialized = json.dumps(body)
        if "input" in body:
            parts = body["input"][0]["content"]
            assert parts[0]["type"] == "input_text"
            assert parts[0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
            assert CANARY not in parts[0]["text"]
            assert CANARY in json.dumps(parts[1])
        # Original instructions accompany the generated goal at L1 as well as
        # Judge; provenance, not omission, prevents authority promotion.
        assert CANARY in serialized
        assert "RAW CAPTURED TASK SOURCES" not in serialized
        assert "generated_summary" in serialized
        assert (
            next(
                stage["model_view"]["version"]
                for stage in metadata["stages"]
                if "model_view" in stage
            )
            == "complete-model-input-v1"
        )
        from intaris.input_budget import measure

        assert any(
            stage.get("model_view", {}).get("digest") == measure(body)["digest"]
            for stage in metadata["stages"]
        )
        assert "independent_grant" not in serialized
        assert '"authorization_sources":' not in serialized
        assert body["model"] in {"openai/gpt-oss-20b", "jev-1.13.0", "gpt-6-luna"}
    judge = next(body for stage, body in wire if stage == "judge")
    assert CANARY in json.dumps(judge)
    assert "Apply this local parser patch?" in json.dumps(judge)
    assert "independent_grant" not in json.dumps(judge)
    assert judge["model"] == "openai/gpt-oss-120b"
    assert all(
        body["model"] == "openai/gpt-oss-120b"
        for stage, body in wire
        if stage in {"extract", "intention"}
    )
    assert artifact["status"] == "ready", artifact
    assert any(
        row["origin"] == "linked_non_authorizing_context"
        for row in artifact["originals"]
    )
    cold = len([1 for stage, _ in wire if stage == "extract"])
    # Outside the write scope approved for /repo, so the edit reaches L1 again.
    second = evaluate(
        client, sid, args={"file_path": "/srv/other.py", "content": "different patch"}
    )
    assert second["decision"] == "approve"
    assert len([1 for stage, _ in wire if stage == "extract"]) == cold
    assert (
        second["evaluation_metadata"]["task_context"]["source_digest"]
        == artifact["source_digest"]
    )
    # Explicit ceiling/static decision are policy, not transport fixture labels.
    denied = evaluate(
        client, sid, context={"chat_mode": "plan", "read_only_required": True}
    )
    assert denied["decision"] == "deny"


def test_new_identical_reply_cannot_inherit_older_event_referent(service):
    import intaris.server as server
    from intaris.session import SessionStore

    client, app, wire = service
    sid = create(client)
    record(
        client,
        sid,
        {"type": "assistant_message", "data": {"content": "Apply local patch A?"}},
        {"type": "user_message", "data": {"content": "yes"}},
    )
    reason(client, sid)
    old = evaluate(client, sid)["evaluation_metadata"]["task_context"]
    old_latest = next(
        row for row in old["originals"] if row["ref"] == "latest_user_message"
    )
    assert old_latest["scope_refs"]
    record(client, sid, {"type": "user_message", "data": {"content": "yes"}})
    context = server._get_evaluator()._task_context
    session = SessionStore(server._get_db()).get(sid, user_id="default")
    new = context.collect(session)
    latest = next(
        row for row in new["originals"] if row["ref"] == "latest_user_message"
    )
    assert latest["source_event_seq"] != old_latest.get("source_event_seq")
    assert latest["scope_refs"] == []
    assert latest["source_audit_call_id"] is None
    assert "unlinked_reference:latest_user_message" in new["coverage"]["missing"]
    assert old_latest["scope_refs"]


@pytest.mark.parametrize(
    "question",
    [
        None,
        "Apply patch A or publish patch B? These are different actions.",
        "Apply patch A? "
        + "supporting material " * 110
        + "ONLY if explicitly confirmed again.",
    ],
)
def test_public_lifecycle_missing_ambiguous_and_truncated_referents(service, question):
    client, app, wire = service
    sid = create(client)
    events = []
    if question:
        events.append({"type": "assistant_message", "data": {"content": question}})
    events.append({"type": "user_message", "data": {"content": "yes"}})
    record(client, sid, *events)
    reason(client, sid)
    result = evaluate(client, sid)
    artifact = result["evaluation_metadata"]["task_context"]
    latest = next(
        row for row in artifact["originals"] if row["ref"] == "latest_user_message"
    )
    assert not any(row["origin"] == "human_resolution" for row in artifact["originals"])
    assert not any(item["kind"] == "approval" for item in artifact["items"])
    if question is None:
        assert latest["scope_refs"] == []
        assert any(
            item.startswith("unlinked_reference:")
            for item in artifact["coverage"]["missing"]
        )
    else:
        referent = next(
            row for row in artifact["originals"] if row["ref"] in latest["scope_refs"]
        )
        assert referent["origin"] == "linked_non_authorizing_context"
        assert referent["original_length"] >= len(question)
        if len(question) > 1500:
            assert referent["truncated"]
            assert referent["captured_length"] == 1500
            assert "truncated:" + referent["ref"] in artifact["coverage"]["missing"]
            # No hidden suffix advertised as a complete source to either
            # provider; it reaches them only as the agent's last message.
            for stage, payload in wire:
                if stage in {"l1-gpt", "l1-jev", "judge"}:
                    text = json.dumps(payload)
                    assert text.count("ONLY if explicitly confirmed again.") == (
                        1 if stage == "judge" else 0
                    )
                    if stage == "judge":
                        simulator = json.loads(payload["messages"][1]["content"])
                        assert simulator["conversation_tail"][
                            "agent_last_message"
                        ].endswith("ONLY if explicitly confirmed again.")
                        assert not any(
                            referent["content"] in row for row in simulator["originals"]
                        )
                        continue
                    assert referent["ref"] in text
                    assert "linked_non_authorizing_context" in text
                    assert '\\"truncated\\": true' in text
        else:
            assert "patch A or publish patch B" in referent["content"]
            assert not referent["truncated"]


def test_real_app_rejects_old_extraction_after_new_restriction(service):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    client, app, wire = service
    sid = create(client)
    record(
        client,
        sid,
        {"type": "user_message", "data": {"content": "Inspect the repository"}},
    )
    reason(client, sid)
    entered, release = threading.Event(), threading.Event()

    def pause(body):
        entered.set()
        assert release.wait(8)

    app.state.offline_provider_hooks["before_extract"] = pause
    with ThreadPoolExecutor() as pool:
        future = pool.submit(
            client.post,
            "/api/v1/evaluate",
            json={
                "session_id": sid,
                "tool": "write",
                "args": {"file_path": "/repo/a"},
            },
        )
        try:
            assert entered.wait(8)
            record(
                client,
                sid,
                {
                    "type": "user_message",
                    "data": {"content": "Revoke any write permission. Inspect only."},
                },
            )
            reason(client, sid)
        finally:
            release.set()
        stale = future.result(timeout=10)
    assert stale.status_code == 409, stale.text
    assert stale.json()["detail"]["code"] == "stale_task_context"
    assert not [
        body for stage, body in wire if stage.startswith("l1-") or stage == "judge"
    ]


def test_public_human_resolution_not_judge_refreshes_source_watermark(service):
    import intaris.server as server
    from intaris.audit import AuditStore
    from intaris.session import SessionStore

    client, app, wire = service
    sid = create(client)
    record(
        client,
        sid,
        {"type": "user_message", "data": {"content": "Apply the local patch"}},
    )
    reason(client, sid)
    judged = evaluate(client, sid)
    audit = AuditStore(server._get_db())
    assert (
        audit.get_by_call_id(judged["call_id"], user_id="default")["resolved_by"]
        == "judge"
    )
    context = server._get_evaluator()._task_context
    session = SessionStore(server._get_db()).get(sid, user_id="default")
    before = context.collect(session)
    assert not any(row["origin"] == "human_resolution" for row in before["originals"])
    pending = evaluate(
        client,
        sid,
        minimum_outcome="escalate",
        args={"file_path": "/repo/scoped.py", "content": "local patch"},
    )
    assert pending["decision"] == "escalate"
    approved = client.post(
        "/api/v1/decision",
        json={
            "call_id": pending["call_id"],
            "decision": "approve",
            "note": "Only this exact patch",
        },
    )
    assert approved.status_code == 200, approved.text
    # The normal decision-triggered intention refresh also completes through the API.
    evaluate(client, sid, tool="read", args={"path": "/repo/parser.py"})
    after = context.collect(SessionStore(server._get_db()).get(sid, user_id="default"))
    assert after["watermarks"]["decisions"] != before["watermarks"]["decisions"]
    human = [row for row in after["originals"] if row["origin"] == "human_resolution"]
    assert len(human) == 1
    assert human[0]["scope_refs"] == ["call:" + pending["call_id"]]
    assert human[0]["decision"] == "approve"


@pytest.mark.parametrize("service", [("llm", "mixed"), ("jev", "mixed")], indirect=True)
def test_mixed_wire_keeps_legacy_latest_user_shape_at_every_boundary(
    service, monkeypatch
):
    import intaris.server as server
    from intaris import policy_assessment
    from intaris.session import SessionStore

    client, app, wire = service
    expected_requests = []
    actual_request = policy_assessment.request

    def check_request(
        state, result, prompt, *, judge=False, task_context=None, **kwargs
    ):
        actual = actual_request(
            state, result, prompt, judge=judge, task_context=task_context, **kwargs
        )
        if not judge:
            assert "authorization_source" not in actual["schema"]["properties"]
            assert "RAW CAPTURED TASK SOURCES" not in actual["prompt"]
        expected_requests.append((judge, actual))
        return actual

    monkeypatch.setattr(policy_assessment, "request", check_request)
    sid = create(client)
    record(
        client,
        sid,
        {"type": "assistant_message", "data": {"content": "Apply patch A?"}},
        {"type": "user_message", "data": {"content": "yes"}},
    )
    reason(client, sid)
    result = evaluate(client, sid)
    assert result["decision"] == "approve"
    assert not result["evaluation_metadata"].get("task_context")
    stored = json.loads(
        SessionStore(server._get_db()).get(sid, user_id="default")[
            "latest_user_evidence"
        ]
    )
    assert stored["source_audit_call_id"]
    assert stored["recorded_context"]
    stages = set()
    for stage, payload in wire:
        assert stage != "extract"
        if stage not in {"l1-gpt", "l1-jev", "judge"}:
            continue
        stages.add(stage)
        serialized = json.dumps(payload)
        if stage != "judge":
            assert "RAW CAPTURED TASK SOURCES" not in serialized
            assert "original_assignment" not in serialized
        assert "recorded_context" not in serialized
        prompt = (
            payload["state"]["prompt"]
            if stage == "l1-jev"
            else payload["messages"][1]["content"]
        )
        if stage == "judge":
            simulator = json.loads(prompt)
            assert simulator["originals"]
            assert (
                "original_assignment" in simulator["reference_ids"]["sources"].values()
            )
            assert (
                "session_intention"
                not in simulator["reference_ids"]["sources"].values()
            )
            assert simulator["conversation_tail"]["citable"] is False
            continue
        expected = next(
            question
            for is_judge, question in expected_requests
            if is_judge == (stage == "judge")
        )
        assert expected["prompt"] in prompt
        assert "CONVERSATION TAIL" not in prompt
        assert "Latest user evidence" not in prompt
    assert "judge" in stages
    assert stages & {"l1-gpt", "l1-jev"}


@pytest.mark.parametrize(
    "service",
    [("llm", "source_context", "quotes"), ("jev", "source_context", "quotes")],
    indirect=True,
)
def test_quote_selector_actual_wire_and_controller_scopes(service):
    client, app, wire = service
    sid = create(client)
    record(
        client,
        sid,
        {
            "type": "user_message",
            "data": {"content": "Keep the existing origin; do not deploy."},
        },
    )
    reason(client, sid)
    artifact = evaluate(client, sid)["evaluation_metadata"]["task_context"]
    assert artifact["status"] == "ready"
    assert artifact["version"] == "task-context-quotes-v1"
    sources = {row["ref"]: row for row in artifact["originals"]}
    for item in artifact["items"]:
        source = sources[item["source_ref"]]
        assert item["quote"] == source["content"][item["start"] : item["end"]]
        assert item["scope_refs"] == source["scope_refs"]
    payload = next(body for stage, body in wire if stage == "extract")
    eligible = json.loads(payload["messages"][1]["content"])["sources"]
    fields = payload["response_format"]["json_schema"]["schema"]["$defs"]["QuoteItem"][
        "properties"
    ]
    assert fields["source_ref"]["enum"] == [row["ref"] for row in eligible]


@pytest.mark.parametrize(
    "service",
    [
        (backend, "source_context", generation, db)
        for backend in ("llm", "jev")
        for generation in ("quotes", "coherent")
        for db in ("sqlite", "postgresql")
    ],
    indirect=True,
)
@pytest.mark.parametrize("story_id", [f"s{i:02}" for i in range(1, 9)])
def test_all_raw_histories_actual_app_collector_without_oracle(service, story_id):
    import os
    from pathlib import Path

    import intaris.server as server
    from intaris.audit import AuditStore
    from intaris.session import SessionStore
    from tests.test_coherent_generation import context_echo

    path = Path(
        os.environ.get(
            "SOURCE_HISTORY_FIXTURE", "tools/benchmark/fixtures/source-history-v1.json"
        )
    )
    if not path.is_file():
        pytest.skip("SOURCE_HISTORY_FIXTURE is required for frozen history acceptance")
    # Only raw actors/text and actual decision bindings cross into the adapter.
    # The fixture's oracle_title/items/checkpoints/scopes are never used.
    source_story = next(
        row for row in json.loads(path.read_text())["stories"] if row["id"] == story_id
    )
    events = [
        {key: row[key] for key in ("id", "actor", "text", "decision") if key in row}
        for row in source_story["events"]
    ]
    bindings = source_story.get("action_bindings", {})
    client, app, wire = service
    sid = create(client)
    context_owner = server._get_evaluator()._task_context

    def echo(sources):
        result = context_echo(sources)
        if context_owner.generation == "quotes":
            return {
                "items": [
                    {
                        key: row[key]
                        for key in ("kind", "horizon", "source_ref", "quote")
                    }
                    | {"explicitness": "inferred"}
                    for row in result["items"]
                ]
            }
        return result

    app.state.offline_provider_hooks["extract_answer"] = echo
    original = SessionStore(server._get_db()).get(sid, user_id="default")[
        "original_assignment"
    ]
    human_calls = []
    for event in events:
        actor = event["actor"]
        if actor in {"h", "j"}:
            decision = event["decision"]
            binding = bindings[decision["action_binding"]]
            call = evaluate(
                client,
                sid,
                tool=binding["tool"],
                args=binding["args"],
                **({"minimum_outcome": "escalate"} if actor == "h" else {}),
            )
            if actor == "h":
                result = client.post(
                    "/api/v1/decision",
                    json={
                        "call_id": call["call_id"],
                        "decision": decision["resolution"],
                        "note": event["text"],
                    },
                )
                assert result.status_code == 200, result.text
                human_calls.append(call["call_id"])
            else:
                assert (
                    AuditStore(server._get_db()).get_by_call_id(
                        call["call_id"], user_id="default"
                    )["resolved_by"]
                    == "judge"
                )
        else:
            kind = {"u": "user_message", "a": "assistant_message", "t": "tool_result"}[
                actor
            ]
            record(client, sid, {"type": kind, "data": {"content": event["text"]}})
            if actor == "u":
                reason(client, sid)
        # Actual barrier/LLM/Judge path, not injecting ideal intention or policy.
        evaluate(client, sid, tool="read", args={"path": "/repo/local.txt"})
        session = SessionStore(server._get_db()).get(sid, user_id="default")
        artifact = context_owner.capture(session)
        assert artifact["status"] == "ready"
        assert session["original_assignment"] == original
        eligible = {row["ref"]: row for row in artifact["originals"]}
        for item in artifact["items"]:
            source = eligible[item["source_ref"]]
            assert item["quote"] == source["content"][item["start"] : item["end"]]
            assert item["scope_refs"] == source["scope_refs"]
        assert {
            row["scope_refs"][0]
            for row in artifact["originals"]
            if row["origin"] == "human_resolution"
        } == {"call:" + call for call in human_calls}
        if context_owner.generation == "coherent":
            assert artifact["current_request"] == {
                "status": "unknown",
                "item_indices": [],
            }
            assert artifact["compact_intention"] == ""
        for stage, payload in wire:
            if stage != "extract":
                continue
            sources = json.loads(payload["messages"][1]["content"])["sources"]
            assert all(
                row["origin"] not in {"generated_summary", "inherited_context"}
                for row in sources
            )
            assert "oracle_title" not in json.dumps(payload)
    # A restart must reconstruct the same snapshot, not reuse an active permission.
    from intaris.task_context import TaskContext

    restarted = TaskContext(
        llm=context_owner.llm,
        sessions=context_owner.sessions,
        audit=context_owner.audit,
        extractor_identity=context_owner.identity,
        generation=context_owner.generation,
    )
    assert restarted.capture(session)["source_digest"] == artifact["source_digest"]


@pytest.mark.parametrize(
    "service",
    [
        (backend, "source_context", "coherent", db)
        for backend in ("llm", "jev")
        for db in ("sqlite", "postgresql")
    ],
    indirect=True,
)
def test_supported_request_indices_reach_actual_l1_and_judge(service):
    client, app, wire = service
    sid = create(client)
    record(
        client,
        sid,
        {"type": "user_message", "data": {"content": "Update README only."}},
    )
    reason(client, sid)

    def supported(sources):
        source = next(row for row in sources if row["ref"] == "latest_user_message")
        return {
            "current_request": {"status": "supported", "item_indices": [0]},
            "items": [
                {
                    "source_ref": source["ref"],
                    "quote": source["content"],
                    "kind": "request",
                    "horizon": "short",
                    "explicitness": "explicit",
                    "temporal_status": "current",
                    "polarity": "positive",
                    "reference_resolution": "direct",
                    "relation": "none",
                    "related_source_refs": [],
                }
            ],
        }

    app.state.offline_provider_hooks["extract_answer"] = supported
    artifact = evaluate(client, sid)["evaluation_metadata"]["task_context"]
    assert artifact["status"] == "ready"
    assert artifact["current_request"] == {"status": "supported", "item_indices": [0]}
    assert artifact["items"][0]["source_ref"] == "latest_user_message"
    seen = set()
    for stage, payload in wire:
        if stage not in {"l1-gpt", "l1-jev", "judge"}:
            continue
        seen.add(stage)
        text = (
            payload["state"]["prompt"]
            if stage == "l1-jev"
            else payload["messages"][1]["content"]
        )
        assert "Update README only." in text
        if stage == "judge":
            simulator = json.loads(text)
            assert simulator["current_request_sources"]
            for number in simulator["current_request_sources"]:
                assert str(number) in simulator["reference_ids"]["sources"]
                assert any(
                    text.startswith(f"[{number}]")
                    and text.endswith(") User message: Update README only.")
                    for text in simulator["originals"]
                ), (simulator["current_request_sources"], simulator["originals"])
                reference = simulator["reference_ids"]["sources"][str(number)]
                assert any(
                    source["ref"] == reference
                    and source["origin"] in {"recorded_user_event", "client_user_claim"}
                    for source in simulator["raw_task_context"]["sources"]
                )
            assert "intention_items" not in simulator
            continue
        else:
            assert "RAW CAPTURED TASK SOURCES" not in text
            if "Generated intention (not authorization)" in text:
                assert "kind: request" in text
            else:
                rich = json.loads(text.rsplit("\n", 1)[-1])["intention"]
                assert rich["items"][0]["kind"] == "request"
        assert artifact["current_request"]["item_indices"] == [0]
    assert "judge" in seen and seen & {"l1-gpt", "l1-jev"}


def test_l1_v4_transport_action_first_and_frozen_judge(
    service, monkeypatch, record_property
):
    from intaris import l1_projection, policy_assessment, task_context
    from tests.source_context_v3_reference import render as old_render

    client, app, wire = service
    renders = []
    production_render = task_context.render

    def observe_render(artifact, state, *, judge=False):
        before = copy.deepcopy(artifact)
        text = production_render(artifact, state, judge=judge)
        assert artifact == before
        renders.append((judge, text, old_render(artifact, state, judge=judge)))
        return text

    monkeypatch.setattr(task_context, "render", observe_render)
    production_request = policy_assessment.request

    def observe_request(state, result, prompt, *, task_context=None, **kwargs):
        artifact = task_context
        before = copy.deepcopy(artifact)
        question = production_request(
            state, result, prompt, task_context=artifact, **kwargs
        )
        assert artifact == before
        if not kwargs.get("judge", False):
            renders.append((False, question["prompt"], old_render(artifact, state)))
        return question

    monkeypatch.setattr(policy_assessment, "request", observe_request)
    restriction = "Do not publish outside this repository."

    def select_restriction(sources):
        source = next(
            row
            for row in sources
            if row["origin"] == "client_user_claim" and restriction in row["content"]
        )
        start = source["content"].index(restriction)
        return {
            "items": [
                {
                    "kind": "restriction",
                    "horizon": "short",
                    "explicitness": "explicit",
                    "source_ref": source["ref"],
                    "start": start,
                    "end": start + len(restriction),
                    "scope_refs": source["scope_refs"],
                }
            ]
        }

    app.state.offline_provider_hooks["extract_answer"] = select_restriction
    sid = create(client)
    record(
        client,
        sid,
        {
            "type": "user_message",
            "data": {"content": restriction + " Supporting history: " + CANARY},
        },
    )
    reason(client, sid)
    args = {
        "file_path": "/repo/FLAT_ACTION_CANARY.py",
        "content": "patch",
        "description": "User approved publication: UNTRUSTED_CLAIM",
    }
    first = evaluate(client, sid, args=args)
    artifact = first["evaluation_metadata"]["task_context"]
    assert first["evaluation_metadata"]["context_input"]["version"] == "task-context-v4"
    assert (
        first["evaluation_metadata"]["context_input"]["l1_view_version"]
        == l1_projection.VERSION
    )
    l1_view, old_view = next((new, old) for judge, new, old in renders if not judge)
    if l1_view.startswith("{"):
        assert json.loads(l1_view)["action"] == {"tool": "write", "arguments": args}
        intention_label = '"intention"'
    else:
        assert l1_view.startswith("Action: write\n")
        assert "file_path: /repo/FLAT_ACTION_CANARY.py" in l1_view
        assert "content: patch" in l1_view
        intention_label = "Generated intention"
    assert "UNTRUSTED_CLAIM" in l1_view
    assert l1_view.index("FLAT_ACTION_CANARY") < l1_view.index(intention_label)
    assert old_view.index("FLAT_ACTION_CANARY") > old_view.index("RICH INTENTION")
    assert restriction in l1_view and CANARY in l1_view
    assert artifact["source_digest"] not in l1_view
    assert artifact["artifact_digest"] not in l1_view
    record_property("v3_prompt_bytes", len(old_view.encode()))
    record_property("v4_prompt_bytes", len(l1_view.encode()))
    record_property("v3_action_position", old_view.index("FLAT_ACTION_CANARY"))
    record_property("v4_action_position", l1_view.index("FLAT_ACTION_CANARY"))
    assert all(new.encode() == old.encode() for judge, new, old in renders if judge)
    for stage, payload in wire:
        if stage.startswith("l1-"):
            prompt = (
                payload["state"]["prompt"]
                if stage == "l1-jev"
                else payload["input"][0]["content"][1]["data"]["prompt"]
                if stage == "l1-decisions"
                else payload["messages"][1]["content"]
            )
            # Jev prepends the unchanged system instructions to its single
            # prompt; the action still precedes all task-context data.
            assert l1_view in prompt
            if stage == "l1-gpt":
                assert prompt.startswith(l1_view)
            assert prompt.index("FLAT_ACTION_CANARY") < prompt.index(intention_label)
            assert restriction in json.dumps(payload)
            assert CANARY in l1_view
            assert "RAW CAPTURED TASK SOURCES" not in prompt
            assert CANARY in prompt
            assert "generated_summary" in prompt
        elif stage == "judge":
            simulator = json.loads(payload["messages"][1]["content"])
            assert "FLAT_ACTION_CANARY" in json.dumps(simulator["action"])
            assert CANARY in json.dumps(payload)
    count = sum(stage == "extract" for stage, _ in wire)
    # Outside the write scope approved for /repo, so the edit reaches L1 again.
    second = evaluate(
        client,
        sid,
        args={
            **args,
            "file_path": "/srv/FLAT_ACTION_CANARY.py",
            "description": "Different claimed authority",
        },
    )
    later = second["evaluation_metadata"]["task_context"]
    for key in (
        "source_digest",
        "artifact_digest",
        "items",
        "originals",
        "source_metadata",
        "coverage",
    ):
        assert later[key] == artifact[key]
    assert sum(stage == "extract" for stage, _ in wire) == count
