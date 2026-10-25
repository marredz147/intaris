"""Transient retries for every model call (LLM and Jev transports)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest

from intaris import model_retry
from intaris.config import JevConfig, LLMConfig
from intaris.jev import JevClient
from intaris.llm import LLMClient, LLMTemporaryError
from tests.test_single_intention import stores as stores_fixture

stores = stores_fixture


class _StatusError(Exception):
    def __init__(self, status: int, message: str = "error", body=None):
        super().__init__(message)
        self.status_code = status
        self.body = body or {"error": {"message": message}}
        self.response = SimpleNamespace(headers={})


class APIConnectionError(Exception):
    """Same class name as the OpenAI SDK's connection error."""


def _response(content='{"ok": true}'):
    return SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content, refusal=None),
            )
        ],
    )


@pytest.fixture
def llm_with(monkeypatch):
    """LLMClient whose provider replays ``outcomes`` (exceptions or content)."""
    sleeps: list[float] = []
    monkeypatch.setattr("intaris.llm.time.sleep", sleeps.append)

    def make(outcomes):
        calls: list[dict] = []

        class _Completions:
            def create(self, **params):
                calls.append({**params, "timeout": options["timeout"]})
                outcome = outcomes[min(len(calls), len(outcomes)) - 1]
                if isinstance(outcome, Exception):
                    raise outcome
                return _response(outcome)

        options = {}

        class _OpenAI:
            def __init__(self, **kwargs):
                assert kwargs["max_retries"] == 0  # One retry policy: ours.
                self.chat = SimpleNamespace(completions=_Completions())
                self.embeddings = _Completions()

            def with_options(self, **kwargs):
                options.update(kwargs)
                assert kwargs["max_retries"] == 0
                return self

            def close(self):
                return None

        monkeypatch.setattr("intaris.llm.OpenAI", _OpenAI)
        client = LLMClient(LLMConfig(api_key="k", base_url="http://llm.test"))
        return client, calls, sleeps

    return make


SCHEMA = {"name": "x", "schema": {"type": "object"}}


@pytest.mark.parametrize(
    "error",
    [
        _StatusError(503),
        _StatusError(502, body={"error": {"code": "upstream_error"}}),
        _StatusError(500),
        _StatusError(504),
        _StatusError(429, "Rate limit reached"),
        APIConnectionError("Connection error."),
    ],
)
def test_single_attempt_call_retries_transient_errors_then_succeeds(
    llm_with, error, caplog
):
    """The Judge, L1 and intention paths use single_attempt calls."""
    client, calls, sleeps = llm_with([error, error, '{"ok": true}'])
    with caplog.at_level(logging.WARNING, logger="intaris.llm"):
        raw = client.generate(
            [{"role": "user", "content": "hi"}],
            json_schema=SCHEMA,
            single_attempt=True,
            response_mode="json_object",
        )
    assert raw == '{"ok": true}'
    assert len(calls) == 3
    assert client.generation_metrics["calls"] == 3
    assert len(sleeps) == 2 and all(0 <= s <= 2.0 for s in sleeps)
    retries = [r.getMessage() for r in caplog.records if "retrying" in r.message]
    assert "attempt 1/4" in retries[0] and type(error).__name__ in retries[0]
    assert "attempt 2/4" in retries[1]


def test_retries_are_exhausted_after_three(llm_with):
    client, calls, sleeps = llm_with([_StatusError(503)])
    with pytest.raises(LLMTemporaryError):
        client.generate([{"role": "user", "content": "hi"}])
    assert len(calls) == 4
    assert len(sleeps) == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_validation_and_auth_errors_never_retry(llm_with, status):
    error = _StatusError(status, "connection field is invalid; timed out")
    client, calls, sleeps = llm_with([error, '{"ok": true}'])
    with pytest.raises(_StatusError):
        client.generate(
            [{"role": "user", "content": "hi"}],
            json_schema=SCHEMA,
            single_attempt=True,
            response_mode="json_object",
        )
    assert len(calls) == 1
    assert sleeps == []


def test_no_retry_when_the_deadline_leaves_no_room(llm_with):
    client, calls, sleeps = llm_with([_StatusError(503), '{"ok": true}'])
    with model_retry.evaluation_deadline(0.5):
        with pytest.raises(LLMTemporaryError):
            client.generate([{"role": "user", "content": "hi"}])
    assert len(calls) == 1
    assert sleeps == []


def test_retry_timeout_is_capped_to_the_remaining_budget(llm_with):
    client, calls, _ = llm_with([_StatusError(503), '{"ok": true}'])
    with model_retry.evaluation_deadline(3.0):
        client.generate([{"role": "user", "content": "hi"}])
    assert 1.0 <= calls[0]["timeout"] <= 3.0
    assert 1.0 <= calls[1]["timeout"] <= 3.0


def test_nested_deadline_never_extends_the_outer_one():
    with model_retry.evaluation_deadline(2.0):
        with model_retry.evaluation_deadline(60.0):
            assert 1.0 <= model_retry.retry_timeout(0.0, 30.0) <= 2.0
    assert model_retry.retry_timeout(0.0, 30.0) == 30.0


@pytest.mark.parametrize("outer", [None, 3.0, 60.0])
def test_model_attempt_timeout_does_not_replace_request_retry_deadline(
    monkeypatch, outer
):
    now = [0.0]
    monkeypatch.setattr(model_retry.time, "monotonic", lambda: now[0])
    with model_retry.evaluation_deadline(outer):
        with model_retry.model_call_deadline(4.0):
            assert model_retry.retry_timeout(0, 4, first_attempt=True) == min(
                outer or 4, 4
            )
            now[0] = 4.0
            remaining = model_retry.retry_timeout(0, 4)
            assert remaining == (4.0 if outer == 60.0 else None)
    assert model_retry.remaining_seconds() is None


def test_jev_recovers_after_full_attempt_timeout_within_outer_budget(monkeypatch):
    now = [0.0]
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            now[0] += request.extensions["timeout"]["read"]
            raise httpx.ReadTimeout("first attempt expired", request=request)
        return httpx.Response(200, json={"ok": True})

    client = _jev(handler, monkeypatch)
    monkeypatch.setattr(model_retry.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(model_retry, "backoff_delay", lambda *args: 0.0)
    try:
        with model_retry.evaluation_deadline(60):
            assert client._request(state={}, questions={}) == {"ok": True}
            assert model_retry.remaining_seconds() == 56
    finally:
        client.close()
    assert len(calls) == 2
    assert all(call.extensions["timeout"]["read"] == 4 for call in calls)


@pytest.mark.parametrize("embedding", [False, True])
def test_llm_recovers_after_full_attempt_timeout_within_outer_budget(
    llm_with, monkeypatch, embedding
):
    client, calls, _ = llm_with([APIConnectionError("timed out"), "ok"])
    now = [0.0]
    monkeypatch.setattr(model_retry.time, "monotonic", lambda: now[0])

    def after_timeout(*args, **kwargs):
        # The provider stub failed its full four-second first attempt.
        now[0] = 4.0
        return 0.0

    monkeypatch.setattr(model_retry, "backoff_delay", after_timeout)
    with model_retry.evaluation_deadline(60):
        if embedding:
            assert client.embed(["text"]).choices[0].message.content == "ok"
        else:
            assert client.generate([{"role": "user", "content": "hi"}]) == "ok"
        assert model_retry.remaining_seconds() == 56
    assert len(calls) == 2
    assert all(call["timeout"] == 4 for call in calls)


def test_embedding_uses_same_retry_owner(llm_with):
    client, calls, _ = llm_with([_StatusError(503), "ok"])
    assert client.embed(["text"]).choices[0].message.content == "ok"
    assert len(calls) == 2


def test_intention_barrier_does_not_restart_exhausted_transport(stores, llm_with):
    import asyncio

    from tests.test_rich_context_recovery import barrier, finish, record

    record(stores, "Read README only.")
    client, calls, _ = llm_with([_StatusError(503)])
    owner = barrier(stores, client)

    async def run():
        await owner.trigger("owner", "session")
        await finish(owner)

    asyncio.run(run())
    assert len(calls) == 4
    assert owner.recovery_attempts == 0


def test_llm_http_date_retry_after_is_honored(llm_with, monkeypatch):
    error = _StatusError(429)
    error.response.headers["retry-after"] = "Thu, 01 Jan 1970 00:00:30 GMT"
    monkeypatch.setattr(model_retry.time, "time", lambda: 0)
    client, calls, sleeps = llm_with([error, "ok"])
    client.generate([{"role": "user", "content": "hi"}])
    assert len(calls) == 2
    assert sleeps == [2.0]


def test_jev_http_date_retry_after_is_honored(monkeypatch):
    handler, seen = _replay(
        [
            httpx.Response(
                429, headers={"retry-after": "Thu, 01 Jan 1970 00:00:30 GMT"}
            ),
            httpx.Response(200, json={}),
        ]
    )
    client = _jev(handler, monkeypatch)
    sleeps = []
    monkeypatch.setattr(model_retry.time, "time", lambda: 0)
    monkeypatch.setattr(model_retry.time, "sleep", sleeps.append)
    assert client._request(state={}, questions={}) == {}
    assert len(seen) == 2
    assert sleeps == [2.0]


def test_jev_retries_connection_reset_while_reading(monkeypatch):
    handler, seen = _replay(
        [
            httpx.ReadError("connection reset"),
            httpx.Response(200, json={}),
        ]
    )
    assert _jev(handler, monkeypatch)._request(state={}, questions={}) == {}
    assert len(seen) == 2


def test_deadline_is_rechecked_after_backoff(llm_with, monkeypatch):
    client, calls, _ = llm_with([_StatusError(503), "ok"])
    now = [0.0]
    monkeypatch.setattr(model_retry.time, "monotonic", lambda: now[0])
    # Simulate a delayed wakeup: the budget was available before sleeping,
    # but expired while the worker was descheduled.
    monkeypatch.setattr(model_retry.time, "sleep", lambda _: now.__setitem__(0, 5.0))
    with model_retry.evaluation_deadline(4.0):
        with pytest.raises(LLMTemporaryError, match="deadline"):
            client.generate([{"role": "user", "content": "hi"}])
    assert len(calls) == 1


def test_jev_deadline_is_rechecked_after_backoff(monkeypatch):
    handler, seen = _replay([httpx.Response(503), httpx.Response(200, json={})])
    client = _jev(handler, monkeypatch)
    now = [0.0]
    monkeypatch.setattr(model_retry.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(model_retry.time, "sleep", lambda _: now.__setitem__(0, 5.0))
    with pytest.raises(httpx.TimeoutException):
        client._request(state={}, questions={})
    assert len(seen) == 1


def test_jitter_is_bounded_and_retry_after_is_capped():
    for retry in range(1, 6):
        window = min(2.0, 0.25 * 2 ** (retry - 1))
        samples = [model_retry.backoff_delay(retry) for _ in range(200)]
        assert all(0.0 <= s <= window for s in samples)
        assert len(set(samples)) > 1  # Full jitter, not a fixed delay.
    assert model_retry.backoff_delay(1, retry_after=0.3) == 0.3
    assert model_retry.backoff_delay(1, retry_after=30.0) == 2.0


# Jev transport ---------------------------------------------------------------


def _jev(handler, monkeypatch):
    monkeypatch.setattr("intaris.finite_questions.time.sleep", lambda _s: None)

    def transport(request):
        if request.method == "GET":
            return httpx.Response(404)
        return handler(request)

    return JevClient(
        JevConfig(enabled=True, api_key="k", base_url="http://jev.test"),
        transport=httpx.MockTransport(transport),
    )


def _replay(responses):
    seen = []

    def handler(request):
        seen.append(request)
        outcome = responses[min(len(seen), len(responses)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return handler, seen


def test_jev_retries_transient_errors_then_succeeds(monkeypatch, caplog):
    handler, seen = _replay(
        [
            httpx.ConnectError("refused"),
            httpx.Response(503, json={"error": {"code": "model_unavailable"}}),
            httpx.Response(429, headers={"retry-after": "1"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    with caplog.at_level(logging.WARNING, logger="intaris.jev"):
        payload = _jev(handler, monkeypatch)._request(state={}, questions={})
    assert payload == {"ok": True}
    assert len(seen) == 4
    retries = [r.getMessage() for r in caplog.records if "retrying" in r.message]
    assert "ConnectError" in retries[0] and "attempt 1/4" in retries[0]
    assert "HTTP 503" in retries[1] and "HTTP 429" in retries[2]


def test_jev_exhausted_retries_raise(monkeypatch):
    handler, seen = _replay([httpx.Response(502)])
    with pytest.raises(httpx.HTTPStatusError):
        _jev(handler, monkeypatch)._request(state={}, questions={})
    assert len(seen) == 4
    handler, seen = _replay([httpx.ReadTimeout("slow")])
    with pytest.raises(httpx.ReadTimeout):
        _jev(handler, monkeypatch)._request(state={}, questions={})
    assert len(seen) == 4


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_jev_validation_errors_never_retry(monkeypatch, status):
    handler, seen = _replay([httpx.Response(status, json={"error": "bad"})])
    with pytest.raises(httpx.HTTPStatusError):
        _jev(handler, monkeypatch)._request(state={}, questions={})
    assert len(seen) == 1


def test_jev_respects_the_deadline(monkeypatch):
    handler, seen = _replay([httpx.Response(503), httpx.Response(200, json={})])
    with model_retry.evaluation_deadline(0.5):
        with pytest.raises(httpx.HTTPStatusError):
            _jev(handler, monkeypatch)._request(state={}, questions={})
    assert len(seen) == 1
