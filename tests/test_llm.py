"""Tests for LLM response parsing and key validation (intaris/llm.py).

Covers alias remapping, missing key detection, extra key stripping,
and edge cases in _validate_keys().
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from intaris.config import LLMConfig
from intaris.llm import (
    _KEY_ALIASES,
    LLMClient,
    LLMTemporaryError,
    _validate_keys,
    parse_json_response,
)
from intaris.prompts_analysis import SESSION_SUMMARY_SCHEMA

# ── _validate_keys: basic behavior ────────────────────────────────────


class TestValidateKeysBasic:
    """Tests for basic key validation (no alias remapping needed)."""

    def test_no_expected_keys_passes_through(self):
        result = {"foo": 1, "bar": 2}
        assert _validate_keys(result, None) == {"foo": 1, "bar": 2}

    def test_exact_match_passes_through(self):
        result = {
            "aligned": True,
            "risk": "low",
            "reasoning": "ok",
            "decision": "approve",
        }
        expected = {"aligned", "risk", "reasoning", "decision"}
        assert _validate_keys(result, expected) == result

    def test_extra_keys_stripped(self):
        result = {
            "aligned": True,
            "risk": "low",
            "reasoning": "ok",
            "decision": "approve",
            "extra": "junk",
        }
        expected = {"aligned", "risk", "reasoning", "decision"}
        out = _validate_keys(result, expected)
        assert "extra" not in out
        assert out == {
            "aligned": True,
            "risk": "low",
            "reasoning": "ok",
            "decision": "approve",
        }

    def test_missing_keys_raises_valueerror(self):
        result = {"aligned": True}
        expected = {"aligned", "risk", "reasoning", "decision"}
        with pytest.raises(ValueError, match="missing required keys"):
            _validate_keys(result, expected)

    def test_empty_result_raises_valueerror(self):
        expected = {"summary", "intent_alignment", "tools_used", "risk_indicators"}
        with pytest.raises(ValueError, match="missing required keys"):
            _validate_keys({}, expected)


# ── _validate_keys: alias remapping ───────────────────────────────────


class TestValidateKeysAliasRemapping:
    """Tests for context-aware alias remapping."""

    def test_remap_compatible_to_aligned(self):
        """Alignment check: compatible → aligned."""
        result = {"compatible": True, "reasoning": "ok"}
        expected = {"aligned", "reasoning"}
        out = _validate_keys(result, expected)
        assert out == {"aligned": True, "reasoning": "ok"}

    def test_remap_reason_to_reasoning(self):
        """Alignment check: reason → reasoning."""
        result = {"aligned": True, "reason": "looks good"}
        expected = {"aligned", "reasoning"}
        out = _validate_keys(result, expected)
        assert out == {"aligned": True, "reasoning": "looks good"}

    def test_remap_compatible_and_reason(self):
        """Alignment check: both compatible → aligned and reason → reasoning."""
        result = {"compatible": True, "reason": "fine"}
        expected = {"aligned", "reasoning"}
        out = _validate_keys(result, expected)
        assert out == {"aligned": True, "reasoning": "fine"}

    def test_remap_narrative_to_summary(self):
        """Session summary: narrative → summary."""
        result = {
            "narrative": "Session did X.",
            "intent_alignment": "aligned",
            "tools_used": ["read"],
            "risk_indicators": [],
        }
        expected = {"summary", "intent_alignment", "tools_used", "risk_indicators"}
        out = _validate_keys(result, expected)
        assert out["summary"] == "Session did X."
        assert "narrative" not in out

    def test_remap_overall_alignment_to_intent_alignment(self):
        """Session summary: overall_alignment → intent_alignment."""
        result = {
            "summary": "ok",
            "overall_alignment": "aligned",
            "tools_used": ["read"],
            "risk_indicators": [],
        }
        expected = {"summary", "intent_alignment", "tools_used", "risk_indicators"}
        out = _validate_keys(result, expected)
        assert out["intent_alignment"] == "aligned"
        assert "overall_alignment" not in out

    def test_remap_all_summary_hallucinations(self):
        """Session summary: all four keys hallucinated (worst case)."""
        result = {
            "narrative": "Session narrative.",
            "overall_alignment": "aligned",
            "trajectory": "stable",  # extra, no target after overall_alignment takes intent_alignment
            "delegated_work_alignment": "ok",  # extra, same
        }
        expected = {"summary", "intent_alignment", "tools_used", "risk_indicators"}
        # narrative → summary, overall_alignment → intent_alignment
        # trajectory and delegated_work_alignment have no remaining targets
        # tools_used and risk_indicators are still missing → ValueError
        with pytest.raises(ValueError, match="missing required keys"):
            _validate_keys(result, expected)

    def test_remap_notes_to_reasoning_in_safety_eval(self):
        """Safety eval: notes → reasoning when reasoning is missing."""
        result = {
            "aligned": True,
            "risk": "low",
            "notes": "This is fine.",
            "decision": "approve",
        }
        expected = {"aligned", "risk", "reasoning", "decision"}
        out = _validate_keys(result, expected)
        assert out["reasoning"] == "This is fine."
        assert "notes" not in out

    def test_remap_alignment_to_aligned_in_safety_eval(self):
        """Safety eval: alignment → aligned (ambiguous key, context resolves)."""
        result = {
            "alignment": True,
            "risk": "low",
            "reasoning": "ok",
            "decision": "approve",
        }
        expected = {"aligned", "risk", "reasoning", "decision"}
        out = _validate_keys(result, expected)
        assert out["aligned"] is True
        assert "alignment" not in out

    def test_remap_alignment_to_intent_alignment_in_summary(self):
        """Session summary: alignment → intent_alignment (context resolves)."""
        result = {
            "summary": "ok",
            "alignment": "aligned",
            "tools_used": ["read"],
            "risk_indicators": [],
        }
        expected = {"summary", "intent_alignment", "tools_used", "risk_indicators"}
        out = _validate_keys(result, expected)
        assert out["intent_alignment"] == "aligned"
        assert "alignment" not in out

    def test_no_remap_when_expected_key_present(self):
        """Don't overwrite existing expected key with alias."""
        result = {
            "aligned": True,
            "compatible": False,  # alias, but aligned already present
            "reasoning": "ok",
        }
        expected = {"aligned", "reasoning"}
        out = _validate_keys(result, expected)
        assert out["aligned"] is True  # original value preserved
        assert "compatible" not in out  # stripped as extra

    def test_remap_with_extra_keys_stripped(self):
        """Alias remapping + extra key stripping in one pass."""
        result = {
            "compatible": True,
            "reason": "fine",
            "injected_field": "malicious",
        }
        expected = {"aligned", "reasoning"}
        out = _validate_keys(result, expected)
        assert out == {"aligned": True, "reasoning": "fine"}
        assert "injected_field" not in out


# ── parse_json_response integration ───────────────────────────────────


class TestParseJsonResponse:
    """Integration tests for parse_json_response with alias remapping."""

    def test_parse_with_alias_remapping(self):
        text = json.dumps({"compatible": True, "reason": "ok"})
        result = parse_json_response(text, expected_keys={"aligned", "reasoning"})
        assert result == {"aligned": True, "reasoning": "ok"}

    def test_parse_raises_on_missing_keys(self):
        text = json.dumps({"narrative": "text", "trajectory": "stable"})
        with pytest.raises(ValueError, match="missing required keys"):
            parse_json_response(
                text,
                expected_keys={
                    "summary",
                    "intent_alignment",
                    "tools_used",
                    "risk_indicators",
                },
            )

    def test_parse_with_markdown_fences(self):
        text = '```json\n{"aligned": true, "reasoning": "ok"}\n```'
        result = parse_json_response(text, expected_keys={"aligned", "reasoning"})
        assert result == {"aligned": True, "reasoning": "ok"}

    def test_parse_no_expected_keys(self):
        text = json.dumps({"any": "keys", "are": "fine"})
        result = parse_json_response(text)
        assert result == {"any": "keys", "are": "fine"}

    def test_parse_normalizes_intent_alignment_alias(self):
        text = json.dumps(
            {
                "summary": "ok",
                "intent_alignment": "partial_aligned",
                "tools_used": ["read"],
                "risk_indicators": [],
            }
        )
        result = parse_json_response(
            text,
            expected_keys={
                "summary",
                "intent_alignment",
                "tools_used",
                "risk_indicators",
            },
        )
        assert result["intent_alignment"] == "partially_aligned"


# ── _KEY_ALIASES coverage ─────────────────────────────────────────────


class TestKeyAliasesCoverage:
    """Verify all aliases in _KEY_ALIASES are valid."""

    def test_all_alias_targets_are_strings(self):
        for key, targets in _KEY_ALIASES.items():
            assert isinstance(key, str), f"Alias key {key!r} is not a string"
            assert isinstance(targets, list), f"Targets for {key!r} is not a list"
            for t in targets:
                assert isinstance(t, str), (
                    f"Target {t!r} for alias {key!r} is not a string"
                )

    def test_observed_hallucinations_have_aliases(self):
        """All hallucinated keys from production logs have aliases."""
        observed = {
            "compatible",
            "reason",
            "notes",
            "alignment",
            "narrative",
            "delegated_work_alignment",
            "trajectory",
            "overall_alignment",
        }
        for key in observed:
            assert key in _KEY_ALIASES, (
                f"Observed hallucination {key!r} has no alias mapping"
            )


class _FakeRateLimitError(Exception):
    def __init__(self, message: str, *, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status_code = 429
        self.body = {"error": {"message": message}}
        self.response = SimpleNamespace(headers=headers or {})


def _make_response(content: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content, refusal=None),
            )
        ]
    )


class _ParameterError(Exception):
    def __init__(
        self,
        param: str | None,
        code: str = "unsupported_parameter",
        message: str = "provider rejected request",
    ):
        super().__init__(message)
        self.body = {"error": {"param": param, "code": code, "message": message}}


def _parameter_client(monkeypatch, create, *, reasoning_effort=None):
    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        def close(self):
            pass

    monkeypatch.setattr("intaris.llm.OpenAI", _FakeOpenAI)
    monkeypatch.setattr("intaris.llm.BadRequestError", _ParameterError)
    return LLMClient(LLMConfig(api_key="mock", reasoning_effort=reasoning_effort))


class TestParameterAdaptation:
    def test_rejected_reasoning_effort_omitted_on_retry_and_next_call(
        self, monkeypatch
    ):
        calls = []

        def create(**params):
            calls.append(params.copy())
            if "reasoning_effort" in params:
                raise _ParameterError("reasoning_effort")
            return _make_response("ok")

        client = _parameter_client(monkeypatch, create, reasoning_effort="low")
        assert client.generate([], reasoning_effort="high") == "ok"
        assert client.generate([], reasoning_effort="high") == "ok"
        assert ["reasoning_effort" in call for call in calls] == [True, False, False]

    def test_rejected_max_tokens_replaced_on_retry_and_next_call(self, monkeypatch):
        calls = []

        def create(**params):
            calls.append(params.copy())
            if "max_tokens" in params:
                raise _ParameterError("max_tokens")
            return _make_response("ok")

        client = _parameter_client(monkeypatch, create)
        assert client.generate([], max_tokens=20) == "ok"
        assert client.generate([], max_tokens=30) == "ok"
        assert [call.get("max_tokens") for call in calls] == [20, None, None]
        assert [call.get("max_completion_tokens") for call in calls] == [None, 20, 30]

    def test_concurrent_stale_params_apply_cached_fix(self, monkeypatch):
        first_entered = Event()
        release_first = Event()
        calls = []

        def create(**params):
            calls.append(params.copy())
            if "max_tokens" in params:
                if not first_entered.is_set():
                    first_entered.set()
                    assert release_first.wait(5)
                raise _ParameterError("max_tokens")
            return _make_response("ok")

        client = _parameter_client(monkeypatch, create)
        schema = {"name": "x", "schema": {"type": "object"}}
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                client.generate, [], max_tokens=20, json_schema=schema
            )
            try:
                assert first_entered.wait(5)
                assert client.generate([], max_tokens=30, json_schema=schema) == "ok"
            finally:
                release_first.set()
            assert first.result(timeout=5) == "ok"
        assert [call.get("max_tokens") for call in calls] == [20, 30, None, None]
        assert sorted(
            call["max_completion_tokens"]
            for call in calls
            if "max_completion_tokens" in call
        ) == [20, 30]
        assert all(call["response_format"]["type"] == "json_schema" for call in calls)
        assert client._supports_structured is True

    def test_unrelated_400_does_not_disable_structured_output(self, monkeypatch):
        calls = []

        def create(**params):
            calls.append(params)
            raise _ParameterError("messages", code="invalid_request_error")

        client = _parameter_client(monkeypatch, create)
        for _ in range(2):
            with pytest.raises(_ParameterError):
                client.generate(
                    [], json_schema={"name": "x", "schema": {"type": "object"}}
                )
        assert client._supports_structured is None
        assert [call["response_format"]["type"] for call in calls] == [
            "json_schema",
            "json_schema",
        ]

    @pytest.mark.parametrize(
        ("param", "code", "message"),
        [
            (
                "messages",
                "unsupported_value",
                "Unsupported messages value in json_schema request",
            ),
            (
                "response_format.json_schema.schema",
                "invalid_json_schema",
                "Unsupported schema keyword in json_schema",
            ),
        ],
    )
    def test_unrelated_or_invalid_schema_error_does_not_fall_back(
        self, monkeypatch, param, code, message
    ):
        calls = []

        def create(**params):
            calls.append(params)
            raise _ParameterError(param, code=code, message=message)

        client = _parameter_client(monkeypatch, create)
        with pytest.raises(_ParameterError):
            client.generate([], json_schema={"name": "x", "schema": {"type": "object"}})
        assert client._supports_structured is None
        assert len(calls) == 1
        assert calls[0]["response_format"]["type"] == "json_schema"

    def test_explicit_unsupported_format_message_without_metadata_falls_back(
        self, monkeypatch
    ):
        calls = []

        def create(**params):
            calls.append(params)
            if params["response_format"]["type"] == "json_schema":
                raise _ParameterError(
                    None,
                    code="invalid_request_error",
                    message="json_schema not supported",
                )
            return _make_response('{"ok": true}')

        client = _parameter_client(monkeypatch, create)
        assert (
            client.generate([], json_schema={"name": "x", "schema": {"type": "object"}})
            == '{"ok": true}'
        )
        assert [call["response_format"]["type"] for call in calls] == [
            "json_schema",
            "json_object",
        ]

    @pytest.mark.parametrize("code", ["unsupported_parameter", "not_supported"])
    def test_explicit_unsupported_schema_falls_back_and_caches(self, monkeypatch, code):
        calls = []

        def create(**params):
            calls.append(params.copy())
            if params["response_format"]["type"] == "json_schema":
                raise _ParameterError("response_format", code=code)
            return _make_response('{"ok": true}')

        client = _parameter_client(monkeypatch, create)
        schema = {"name": "x", "schema": {"type": "object"}}
        assert client.generate([], json_schema=schema) == '{"ok": true}'
        assert client.generate([], json_schema=schema) == '{"ok": true}'
        assert client._supports_structured is False
        assert [call["response_format"]["type"] for call in calls] == [
            "json_schema",
            "json_object",
            "json_object",
        ]


class TestTransientLLMFailures:
    def test_generate_retries_rate_limit_and_succeeds(self, monkeypatch):
        calls = []
        sleeps = []

        class _FakeCompletions:
            def create(self, **params):
                calls.append(params)
                if len(calls) == 1:
                    raise _FakeRateLimitError(
                        "Rate limit reached. Please try again in 4.7052s."
                    )
                return _make_response('{"ok": true}')

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=_FakeCompletions())

            def close(self):
                return None

        monkeypatch.setattr("intaris.llm.OpenAI", _FakeOpenAI)
        monkeypatch.setattr(
            "intaris.llm.time.sleep", lambda seconds: sleeps.append(seconds)
        )

        client = LLMClient(
            LLMConfig(api_key="test-key", base_url="http://example.test"),
            transient_retries=1,
        )

        assert client.generate([{"role": "user", "content": "hi"}]) == '{"ok": true}'
        assert len(calls) == 2
        assert sleeps == [pytest.approx(4.7052)]

    def test_generate_raises_friendly_error_after_retry_exhausted(self, monkeypatch):
        class _FakeCompletions:
            def create(self, **params):
                raise _FakeRateLimitError("Rate limit reached")

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=_FakeCompletions())

            def close(self):
                return None

        monkeypatch.setattr("intaris.llm.OpenAI", _FakeOpenAI)
        monkeypatch.setattr("intaris.llm.time.sleep", lambda seconds: None)

        client = LLMClient(
            LLMConfig(api_key="test-key", base_url="http://example.test"),
            transient_retries=1,
        )

        with pytest.raises(LLMTemporaryError, match="temporarily rate limited"):
            client.generate([{"role": "user", "content": "hi"}])

    def test_structured_output_does_not_swallow_temporary_error(self, monkeypatch):
        class _FakeCompletions:
            def create(self, **params):
                raise _FakeRateLimitError("Rate limit reached")

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=_FakeCompletions())

            def close(self):
                return None

        monkeypatch.setattr("intaris.llm.OpenAI", _FakeOpenAI)

        client = LLMClient(
            LLMConfig(api_key="test-key", base_url="http://example.test"),
            transient_retries=0,
        )

        with pytest.raises(LLMTemporaryError):
            client.generate(
                [{"role": "user", "content": "hi"}],
                json_schema={"name": "x", "schema": {"type": "object"}},
            )

        assert client._supports_structured is None

    def test_generate_recovers_failed_generation_schema_error(self, monkeypatch):
        class _FakeBadRequestError(Exception):
            def __init__(self, body):
                super().__init__("schema validation failed")
                self.body = body

        class _FakeCompletions:
            def create(self, **params):
                raise _FakeBadRequestError(
                    {
                        "error": {
                            "code": "json_validate_failed",
                            "message": "Generated JSON does not match the expected schema",
                            "failed_generation": json.dumps(
                                {
                                    "summary": "ok",
                                    "intent_alignment": "partial_aligned",
                                    "tools_used": ["read"],
                                    "risk_indicators": [],
                                }
                            ),
                        }
                    }
                )

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=_FakeCompletions())

            def close(self):
                return None

        monkeypatch.setattr("intaris.llm.BadRequestError", _FakeBadRequestError)
        monkeypatch.setattr("intaris.llm.OpenAI", _FakeOpenAI)

        client = LLMClient(
            LLMConfig(api_key="test-key", base_url="http://example.test"),
        )

        raw = client.generate(
            [{"role": "user", "content": "hi"}],
            json_schema=SESSION_SUMMARY_SCHEMA,
        )
        result = json.loads(raw)

        assert result["intent_alignment"] == "partially_aligned"
        assert client._supports_structured is True

    def test_generate_falls_back_to_json_mode_after_schema_error(self, monkeypatch):
        calls = []

        class _FakeBadRequestError(Exception):
            def __init__(self, body):
                super().__init__("schema validation failed")
                self.body = body

        class _FakeCompletions:
            def create(self, **params):
                calls.append(params)
                response_format = params.get("response_format", {})
                if response_format.get("type") == "json_schema":
                    raise _FakeBadRequestError(
                        {
                            "error": {
                                "code": "json_validate_failed",
                                "message": "Generated JSON does not match the expected schema",
                                "failed_generation": "not valid json",
                            }
                        }
                    )
                return _make_response(
                    json.dumps(
                        {
                            "summary": "ok",
                            "intent_alignment": "partially_aligned",
                            "tools_used": ["read"],
                            "risk_indicators": [],
                        }
                    )
                )

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=_FakeCompletions())

            def close(self):
                return None

        monkeypatch.setattr("intaris.llm.BadRequestError", _FakeBadRequestError)
        monkeypatch.setattr("intaris.llm.OpenAI", _FakeOpenAI)

        client = LLMClient(
            LLMConfig(api_key="test-key", base_url="http://example.test"),
        )

        raw = client.generate(
            [{"role": "user", "content": "hi"}],
            json_schema=SESSION_SUMMARY_SCHEMA,
        )

        assert json.loads(raw)["intent_alignment"] == "partially_aligned"
        assert [call.get("response_format", {}).get("type") for call in calls] == [
            "json_schema",
            "json_object",
        ]
