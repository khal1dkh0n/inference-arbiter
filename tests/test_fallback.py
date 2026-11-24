"""Fallback behaviour and the client error classification that drives it."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from inference_arbiter.clients import ChatCall, OllamaNativeClient, OpenRouterClient
from inference_arbiter.schemas import AskRequest

from conftest import HarnessKit, error_result, make_config, ok_result, triage_json


def ask(**overrides) -> AskRequest:
    overrides.setdefault("mode", "route")
    return AskRequest.from_question("explain idempotency keys", **overrides)


# --- router: single hop, double failure, skipped hop -------------------------------------


def test_answer_failure_hops_once_to_on_error_target(kit):
    kit.client.script("local-model", error_result("timeout"))
    kit.client.script("mid-model", ok_result("mid rescued it"))

    result = kit.harness.answer(ask(tier="local"))

    assert result.status == "ok"
    assert result.answer == "mid rescued it"
    assert result.selected_tier == "local"
    assert result.final_tier == "mid"
    assert result.final_model == "mid-model"
    assert result.fallback_used is True
    assert kit.client.models_called() == ["local-model", "mid-model"]
    assert [record.status for record in kit.call_records] == ["timeout", "ok"]
    assert kit.run_record.fallback_used is True
    assert kit.run_record.final_tier == "mid"
    assert kit.run_record.error_type is None


def test_double_failure_returns_explicit_error_and_stops(kit):
    # mid's own on_error is local; a third hop would call local-model again and
    # FakeClient would raise on the unscripted call.
    kit.client.script("local-model", error_result("connection"))
    kit.client.script("mid-model", error_result("http_5xx"))

    result = kit.harness.answer(ask(tier="local"))

    assert result.status == "error"
    assert result.answer is None
    assert result.error_type == "http_5xx"
    assert result.fallback_used is True
    assert result.final_tier == "mid"
    assert kit.client.models_called() == ["local-model", "mid-model"]
    assert kit.run_record.status == "error"
    assert kit.run_record.error_type == "http_5xx"
    assert [record.error_type for record in kit.call_records] == ["connection", "http_5xx"]


def test_cloud_fallback_is_skipped_when_disallowed():
    kit = HarnessKit(make_config(allow_cloud_fallback=False))
    kit.client.script("local-model", error_result("timeout"))

    result = kit.harness.answer(ask(tier="local"))

    assert result.status == "error"
    assert result.answer is None
    assert result.fallback_used is False
    assert result.fallback_skipped_reason == "cloud_fallback_disabled"
    assert result.final_tier == "local"
    assert kit.client.models_called() == ["local-model"]


def test_local_fallback_still_allowed_when_cloud_fallback_is_disallowed():
    kit = HarnessKit(make_config(allow_cloud_fallback=False))
    kit.client.script("mid-model", error_result("connection"))
    kit.client.script("local-model", ok_result("local rescued it"))

    result = kit.harness.answer(ask(tier="mid"))

    assert result.status == "ok"
    assert result.final_tier == "local"
    assert result.fallback_used is True


def test_tier_without_on_error_fails_without_a_hop():
    kit = HarnessKit(make_config())
    kit.config.tiers["local"].on_error = None
    kit.client.script("local-model", error_result("empty"))

    result = kit.harness.answer(ask(tier="local"))

    assert result.status == "error"
    assert result.fallback_used is False
    assert result.fallback_skipped_reason == "no_on_error_target"
    assert kit.client.models_called() == ["local-model"]


def test_failed_triage_then_failed_answer_still_gets_one_hop(kit):
    kit.client.script("triage-model", error_result("timeout"))
    kit.client.script("mid-model", error_result("http_429"))
    kit.client.script("local-model", ok_result("local rescued it"))

    result = kit.harness.answer(ask())

    assert result.route_reason == "triage_timeout"
    assert result.selected_tier == "mid"
    assert result.final_tier == "local"
    assert result.status == "ok"
    assert [record.role for record in kit.call_records] == ["triage", "answer", "answer"]


# --- router: --simulate ------------------------------------------------------------------


def test_simulated_triage_timeout_calls_no_triage_model(kit):
    kit.client.script("mid-model", ok_result("mid answer"))

    result = kit.harness.answer(ask(simulate="triage-timeout"))

    assert result.status == "ok"
    assert result.triage_status == "timeout"
    assert result.route_reason == "triage_timeout"
    assert result.selected_tier == "mid"
    assert kit.client.models_called() == ["mid-model"]
    assert kit.call_records[0].role == "triage"
    assert kit.call_records[0].status == "timeout"
    assert kit.call_records[0].error_type == "simulated_timeout"


def test_simulated_answer_error_takes_the_real_fallback_hop(kit):
    kit.client.script("triage-model", triage_json("local", 0.95))
    kit.client.script("mid-model", ok_result("mid answer"))

    result = kit.harness.answer(ask(simulate="answer-error"))

    assert result.status == "ok"
    assert result.selected_tier == "local"
    assert result.final_tier == "mid"
    assert result.fallback_used is True
    assert kit.client.models_called() == ["triage-model", "mid-model"]
    assert kit.call_records[1].error_type == "simulated_error"


# --- OllamaNativeClient ------------------------------------------------------------------


def chat_call(**overrides) -> ChatCall:
    settings = {
        "model": "some-model",
        "messages": [{"role": "user", "content": "hi"}],
        "timeout_s": 5,
    }
    settings.update(overrides)
    return ChatCall(**settings)


def ollama_client(handler) -> OllamaNativeClient:
    return OllamaNativeClient("http://ollama.invalid", transport=httpx.MockTransport(handler))


def test_ollama_success_converts_nanoseconds_and_sends_native_options():
    seen_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen_payloads.append(json.loads(request.content))
        assert request.url.path == "/api/chat"
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "hello"},
                "done_reason": "stop",
                "total_duration": 2_500_000_000,
                "load_duration": 1_200_000_000,
                "prompt_eval_count": 11,
                "prompt_eval_duration": 40_000_000,
                "eval_count": 7,
                "eval_duration": 350_000_000,
            },
        )

    result = ollama_client(handler).chat(
        chat_call(
            num_ctx=8192,
            keep_alive="10m",
            think=False,
            temperature=0.0,
            max_output_tokens=64,
            json_schema={"type": "object"},
        )
    )

    assert result.status == "ok"
    assert result.text == "hello"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 7
    assert result.ollama_load_duration_ms == 1200.0
    assert result.ollama_total_duration_ms == 2500.0
    assert result.ollama_prompt_eval_ms == 40.0
    assert result.ollama_eval_ms == 350.0
    assert result.cost_usd == 0.0
    assert result.cost_source == "local_zero"

    payload = seen_payloads[0]
    assert payload["stream"] is False
    assert payload["options"] == {"num_ctx": 8192, "temperature": 0.0, "num_predict": 64}
    assert payload["keep_alive"] == "10m"
    assert payload["think"] is False
    assert payload["format"] == {"type": "object"}


def test_ollama_omits_unset_optional_fields():
    seen_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen_payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "x"}})

    ollama_client(handler).chat(chat_call())

    assert set(seen_payloads[0]) == {"model", "messages", "stream"}


def test_ollama_estimates_cost_when_prices_are_configured():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"content": "x"}, "prompt_eval_count": 1000, "eval_count": 500},
        )

    result = ollama_client(handler).chat(
        chat_call(input_price_per_mtok=1.0, output_price_per_mtok=2.0)
    )

    assert result.cost_source == "estimated"
    assert result.cost_usd == pytest.approx(0.002)


def raising_handler(exception_factory):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exception_factory(request)

    return handler


@pytest.mark.parametrize(
    ("handler", "expected_error_type"),
    [
        (raising_handler(lambda request: httpx.ReadTimeout("slow", request=request)), "timeout"),
        (raising_handler(lambda request: httpx.ConnectError("refused", request=request)), "connection"),
        (lambda request: httpx.Response(429, text="slow down"), "http_429"),
        (lambda request: httpx.Response(503, text="overloaded"), "http_5xx"),
        (lambda request: httpx.Response(400, text="does not support thinking"), "http_4xx"),
        (lambda request: httpx.Response(200, json={"message": {"content": "  "}}), "empty"),
        (lambda request: httpx.Response(200, text="<html>not json</html>"), "invalid_response"),
    ],
)
def test_ollama_errors_are_classified_not_raised(handler, expected_error_type):
    result = ollama_client(handler).chat(chat_call())

    assert result.status == "error"
    assert result.error_type == expected_error_type
    assert result.text.strip() == ""


# --- OpenRouterClient --------------------------------------------------------------------


class FakeCompletions:
    def __init__(self, response=None, exception=None) -> None:
        self.kwargs = None
        self._response = response
        self._exception = exception

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self._exception is not None:
            raise self._exception
        return self._response


def fake_sdk(response=None, exception=None):
    completions = FakeCompletions(response, exception)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def sdk_response(text: str, **usage_fields):
    usage = SimpleNamespace(
        prompt_tokens=usage_fields.get("prompt_tokens", 1000),
        completion_tokens=usage_fields.get("completion_tokens", 500),
        prompt_tokens_details=SimpleNamespace(cached_tokens=usage_fields.get("cached_tokens", 200)),
    )
    if "cost" in usage_fields:
        usage.cost = usage_fields["cost"]
    if "reasoning_tokens" in usage_fields:
        usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=usage_fields["reasoning_tokens"])
    message = SimpleNamespace(content=text)
    if "refusal" in usage_fields:
        message.refusal = usage_fields["refusal"]
    choice = SimpleNamespace(message=message)
    if "finish_reason" in usage_fields:
        choice.finish_reason = usage_fields["finish_reason"]
    if "native_finish_reason" in usage_fields:
        choice.native_finish_reason = usage_fields["native_finish_reason"]
    return SimpleNamespace(choices=[choice], usage=usage)


def openrouter_client(sdk) -> OpenRouterClient:
    return OpenRouterClient("http://openrouter.invalid", "TEST_OPENROUTER_KEY", sdk_client=sdk)


def test_openrouter_sends_effort_and_usage_accounting_and_reads_reported_cost():
    sdk, completions = fake_sdk(sdk_response("frontier answer", cost=0.0123))

    result = openrouter_client(sdk).chat(chat_call(effort="high", timeout_s=42))

    assert result.status == "ok"
    assert result.text == "frontier answer"
    assert result.prompt_tokens == 1000
    assert result.completion_tokens == 500
    assert result.cached_tokens == 200
    assert result.cost_usd == 0.0123
    assert result.cost_source == "provider_reported"
    assert completions.kwargs["extra_body"] == {
        "usage": {"include": True},
        "reasoning": {"effort": "high"},
    }
    assert completions.kwargs["timeout"] == 42
    assert "temperature" not in completions.kwargs


def test_openrouter_estimates_cost_when_provider_reports_none():
    sdk, _ = fake_sdk(sdk_response("answer"))

    result = openrouter_client(sdk).chat(
        chat_call(input_price_per_mtok=8.0, output_price_per_mtok=40.0)
    )

    assert result.cost_source == "estimated"
    assert result.cost_usd == pytest.approx(0.028)


def http_status_error(status_code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "http://openrouter.invalid/chat/completions")
    response = httpx.Response(status_code, request=request)
    return openai.APIStatusError("provider said no", response=response, body=None)


def sdk_request() -> httpx.Request:
    return httpx.Request("POST", "http://openrouter.invalid/chat/completions")


@pytest.mark.parametrize(
    ("exception", "expected_error_type"),
    [
        (openai.APITimeoutError(request=sdk_request()), "timeout"),
        (openai.APIConnectionError(request=sdk_request()), "connection"),
        (http_status_error(429), "http_429"),
        (http_status_error(502), "http_5xx"),
        (http_status_error(401), "http_4xx"),
        (RuntimeError("something odd"), "client_error"),
    ],
)
def test_openrouter_errors_are_classified_not_raised(exception, expected_error_type):
    sdk, _ = fake_sdk(exception=exception)

    result = openrouter_client(sdk).chat(chat_call())

    assert result.status == "error"
    assert result.error_type == expected_error_type


def test_openrouter_empty_content_is_an_error():
    sdk, _ = fake_sdk(sdk_response(""))

    result = openrouter_client(sdk).chat(chat_call())

    assert result.status == "error"
    assert result.error_type == "empty"


def test_openrouter_records_finish_reason_and_reasoning_tokens():
    sdk, _ = fake_sdk(
        sdk_response(
            "answer",
            completion_tokens=377,
            reasoning_tokens=120,
            finish_reason="stop",
            native_finish_reason="end_turn",
        )
    )

    result = openrouter_client(sdk).chat(chat_call())

    assert result.status == "ok"
    assert result.finish_reason == "stop"
    assert result.native_finish_reason == "end_turn"
    assert result.reasoning_tokens == 120


def test_openrouter_response_without_finish_fields_leaves_them_unset():
    sdk, _ = fake_sdk(sdk_response("answer"))

    result = openrouter_client(sdk).chat(chat_call())

    assert result.status == "ok"
    assert result.finish_reason is None
    assert result.native_finish_reason is None
    assert result.reasoning_tokens is None


def test_openrouter_empty_content_says_why_the_reply_stopped():
    sdk, _ = fake_sdk(
        sdk_response(
            "",
            completion_tokens=150,
            reasoning_tokens=150,
            finish_reason="length",
            native_finish_reason="max_tokens",
        )
    )

    result = openrouter_client(sdk).chat(chat_call())

    assert result.error_type == "empty"
    assert result.finish_reason == "length"
    assert result.reasoning_tokens == 150
    assert result.error_message == (
        "model returned no content (finish_reason=length, native_finish_reason=max_tokens, "
        "completion_tokens=150, reasoning_tokens=150)"
    )


@pytest.mark.parametrize("partial_text", ["", '{"working":"S1-P16 holds'])
def test_openrouter_refusal_is_its_own_error_and_drops_partial_text(partial_text):
    sdk, _ = fake_sdk(
        sdk_response(
            partial_text,
            completion_tokens=80,
            reasoning_tokens=0,
            finish_reason="content_filter",
            native_finish_reason="refusal",
            refusal="This request\nwas blocked.",
        )
    )

    result = openrouter_client(sdk).chat(chat_call())

    assert result.status == "error"
    assert result.error_type == "refused"
    assert result.text == ""
    assert result.finish_reason == "content_filter"
    assert result.error_message == (
        "the model refused the request (finish_reason=content_filter, native_finish_reason=refusal, "
        "completion_tokens=80, reasoning_tokens=0, refusal='This request was blocked.')"
    )


def test_openrouter_without_key_fails_cleanly_and_makes_no_request(monkeypatch):
    monkeypatch.delenv("TEST_OPENROUTER_KEY", raising=False)
    client = OpenRouterClient("http://openrouter.invalid", "TEST_OPENROUTER_KEY")

    result = client.chat(chat_call())

    assert result.status == "error"
    assert result.error_type == "missing_api_key"
    assert "TEST_OPENROUTER_KEY" in result.error_message


def test_no_fallback_request_fails_on_the_selected_tier_without_a_second_call(kit):
    kit.client.script("local-model", error_result("timeout"))

    result = kit.harness.answer(ask(tier="local", no_fallback=True))

    assert result.status == "error"
    assert result.answer is None
    assert result.error_type == "timeout"
    assert result.fallback_used is False
    assert result.fallback_skipped_reason == "no_fallback_requested"
    assert result.final_tier == "local"
    assert kit.client.models_called() == ["local-model"]
    assert [record.status for record in kit.call_records] == ["timeout"]


def test_no_fallback_request_changes_nothing_when_the_tier_answers(kit):
    kit.client.script("local-model", ok_result("local answered"))

    result = kit.harness.answer(ask(tier="local", no_fallback=True))

    assert result.status == "ok"
    assert result.answer == "local answered"
    assert result.fallback_used is False
    assert result.fallback_skipped_reason is None


def test_openrouter_with_an_unresolved_model_fails_cleanly_and_makes_no_request(monkeypatch):
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "test-key")
    client = OpenRouterClient("http://openrouter.invalid", "TEST_OPENROUTER_KEY")

    result = client.chat(chat_call(model="${QAH_TEST_FRONTIER_MODEL}"))

    assert result.status == "error"
    assert result.error_type == "missing_model"
    assert "QAH_TEST_FRONTIER_MODEL" in result.error_message
