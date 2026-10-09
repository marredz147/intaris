"""Native Decisions transport and response contract tests."""

from __future__ import annotations

import copy
import json

import httpx
import pytest

from intaris.config import DecisionsConfig
from intaris.decisions import (
    DecisionsClient,
    DecisionsError,
    DecisionsProtocolError,
    DecisionsTemporaryError,
)


def payload():
    return {
        "model": "reported",
        "usage": {"input_tokens": 10, "output_tokens": 3},
        "answers": [
            {"name": "aligned", "type": "predicate", "probability": 0.95},
            {
                "name": "risk",
                "type": "choice",
                "choice": "low",
                "confidence": 0.99,
                "probabilities": [
                    {"value": x, "probability": p}
                    for x, p in [
                        ("low", 0.94),
                        ("medium", 0.02),
                        ("high", 0.02),
                        ("critical", 0.02),
                    ]
                ],
            },
            {
                "name": "decision",
                "type": "choice",
                "choice": "approve",
                "confidence": 0.97,
                "probabilities": [
                    {"value": x, "probability": p}
                    for x, p in [("approve", 0.95), ("deny", 0.02), ("escalate", 0.03)]
                ],
            },
        ],
    }


def client(handler, **settings):
    config = DecisionsConfig(api_key="dedicated", model="any/custom-model", **settings)
    return DecisionsClient(config, transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    "root,expected",
    [
        ("https://api.openai.com/v1", "/v1/decisions"),
        ("https://api.openai.com/v1/", "/v1/decisions"),
        ("https://gateway.example/custom/v1/", "/custom/v1/decisions"),
    ],
)
def test_request_contract(root, expected):
    def handler(request):
        assert request.url.path == expected
        assert request.headers["Authorization"] == "Bearer dedicated"
        body = json.loads(request.content)
        assert body["model"] == "any/custom-model"
        assert body["input"] == "full untrusted context"
        assert [(q["name"], q["type"]) for q in body["questions"]] == [
            ("aligned", "predicate"),
            ("risk", "choice"),
            ("decision", "choice"),
        ]
        assert all("trusted rules" in q["instructions"] for q in body["questions"])
        assert [c["value"] for c in body["questions"][1]["choices"]] == [
            "low",
            "medium",
            "high",
            "critical",
        ]
        assert [c["value"] for c in body["questions"][2]["choices"]] == [
            "approve",
            "deny",
            "escalate",
        ]
        return httpx.Response(200, json=payload())

    instance = client(handler, base_url=root)
    result = instance.evaluate(
        system_prompt="trusted rules",
        user_prompt="full untrusted context",
        timeout_ms=1000,
    )
    assert result.aligned and result.risk == "low" and result.decision == "approve"
    assert result.metadata["model"] == "reported"
    assert result.metadata["requested_model"] == "any/custom-model"
    assert result.metadata["usage"] == {"input_tokens": 10, "output_tokens": 3}
    instance.close()
    instance.close()
    with pytest.raises(DecisionsError, match="closed"):
        instance.evaluate(system_prompt="rules", user_prompt="context")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["answers"].append(dict(p["answers"][0])),
        lambda p: p["answers"].pop(),
        lambda p: p["answers"][1].update(name="unknown"),
        lambda p: p["answers"][0].update(type="choice"),
        lambda p: p["answers"][0].update(probability=float("nan")),
        lambda p: p["answers"][0].update(probability=True),
        lambda p: p["answers"][1].update(confidence=float("inf")),
        lambda p: p["answers"][1]["probabilities"].append(
            {"value": "low", "probability": 0}
        ),
        lambda p: p["answers"][1]["probabilities"][1].update(value="low"),
        lambda p: p["answers"][1]["probabilities"][0].update(probability=0.5),
        lambda p: p["answers"][2].update(choice="other"),
    ],
)
def test_invalid_answers_rejected(mutation):
    data = copy.deepcopy(payload())
    mutation(data)
    instance = client(lambda request: httpx.Response(200, content=json.dumps(data)))
    with pytest.raises(DecisionsProtocolError):
        instance.evaluate(system_prompt="rules", user_prompt="context")


def test_refusal_validates_other_answers():
    data = payload()
    data["answers"][0] = {"name": "aligned", "type": "refusal"}
    instance = client(lambda request: httpx.Response(200, content=json.dumps(data)))
    result = instance.evaluate(system_prompt="rules", user_prompt="context")
    assert (result.aligned, result.risk, result.decision) == (False, "high", "escalate")
    assert result.metadata["status"] == "refusal"
    data["answers"][1]["confidence"] = float("nan")
    with pytest.raises(DecisionsProtocolError):
        instance.evaluate(system_prompt="rules", user_prompt="context")


def test_low_confidence_escalates():
    data = payload()
    data["answers"][0]["probability"] = 0.51
    instance = client(lambda request: httpx.Response(200, json=data))
    result = instance.evaluate(system_prompt="rules", user_prompt="context")
    assert (result.aligned, result.risk, result.decision) == (False, "high", "escalate")
    assert result.metadata["status"] == "low_confidence"


@pytest.mark.parametrize(
    "status,exception",
    [
        (429, DecisionsTemporaryError),
        (503, DecisionsTemporaryError),
        (401, DecisionsError),
        (400, DecisionsError),
    ],
)
def test_http_errors_no_retries(status, exception):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="do not leak provider body")

    instance = client(handler)
    with pytest.raises(exception) as error:
        instance.evaluate(system_prompt="rules", user_prompt="context")
    assert len(calls) == 1
    assert "do not leak" not in str(error.value)


@pytest.mark.parametrize(
    "error,exception",
    [
        (httpx.ConnectError, DecisionsTemporaryError),
        (httpx.ReadTimeout, DecisionsTemporaryError),
        (httpx.ReadError, DecisionsTemporaryError),
        (httpx.WriteError, DecisionsTemporaryError),
        (httpx.RemoteProtocolError, DecisionsTemporaryError),
        (httpx.ProtocolError, DecisionsError),
        (httpx.LocalProtocolError, DecisionsError),
    ],
)
def test_transport_errors(error, exception):
    def handler(request):
        raise error("failure")

    instance = client(handler)
    with pytest.raises(exception):
        instance.evaluate(system_prompt="rules", user_prompt="context")


def test_malformed_json_not_temporary():
    instance = client(lambda request: httpx.Response(200, text="{"))
    with pytest.raises(DecisionsProtocolError):
        instance.evaluate(system_prompt="rules", user_prompt="context")


def test_usage_exposes_only_validated_token_counts():
    data = payload()
    data["usage"] = {
        "input_tokens": 10,
        "output_tokens": 3,
        "input_tokens_details": {
            "cached_tokens": 2,
            "cache_write_tokens": 1,
            "provider_private": {"credential": "secret"},
        },
        "output_tokens_details": {
            "reasoning_tokens": 1,
            "provider_private": "secret",
        },
        "provider_private": "secret",
    }
    instance = client(lambda request: httpx.Response(200, json=data))
    result = instance.evaluate(system_prompt="rules", user_prompt="context")
    assert result.metadata["usage"] == {
        "input_tokens": 10,
        "output_tokens": 3,
        "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 1},
        "output_tokens_details": {"reasoning_tokens": 1},
    }
    assert "secret" not in json.dumps(result.metadata)


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": True},
        {"output_tokens": -1},
        {"input_tokens_details": []},
        {"input_tokens_details": {"cached_tokens": "1"}},
        {"input_tokens_details": {"cache_write_tokens": False}},
        {"output_tokens_details": {"reasoning_tokens": 1.5}},
    ],
)
def test_invalid_known_usage_rejected(usage):
    data = payload()
    data["usage"] = usage
    instance = client(lambda request: httpx.Response(200, json=data))
    with pytest.raises(DecisionsProtocolError):
        instance.evaluate(system_prompt="rules", user_prompt="context")
