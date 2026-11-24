"""Provider clients. Both return a CallResult and never raise provider errors raw.

OllamaNativeClient talks to /api/chat with httpx because the OpenAI-compatible /v1
endpoint omits load_duration. OpenRouterClient uses the openai SDK.
Neither client retries: the single fallback hop is the router's decision.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import openai

from inference_arbiter.config import HarnessConfig, ProviderConfig, unresolved_model_variable
from inference_arbiter.schemas import STATUS_ERROR, STATUS_OK, CallResult

NANOSECONDS_PER_MILLISECOND = 1_000_000
TOKENS_PER_MILLION = 1_000_000
MAX_ERROR_MESSAGE_CHARS = 300
MAX_REFUSAL_CHARS = 120

# OpenRouter's normalised finish reason for a reply the model or provider declined
# (the native reason is then "refusal"). The content is empty or cut off mid-output.
FINISH_REASON_REFUSED = "content_filter"

ERROR_TIMEOUT = "timeout"
ERROR_CONNECTION = "connection"
ERROR_HTTP_429 = "http_429"
ERROR_HTTP_5XX = "http_5xx"
ERROR_HTTP_4XX = "http_4xx"
ERROR_EMPTY = "empty"
ERROR_REFUSED = "refused"
ERROR_INVALID_RESPONSE = "invalid_response"
ERROR_MISSING_API_KEY = "missing_api_key"
ERROR_MISSING_MODEL = "missing_model"
ERROR_CLIENT = "client_error"


@dataclass
class ChatCall:
    """One fully resolved model call. Built by the router from a tier config."""

    model: str
    messages: list[dict[str, str]]
    timeout_s: float
    effort: str | None = None
    think: bool | str | None = None
    num_ctx: int | None = None
    keep_alive: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    json_schema: dict[str, Any] | None = None
    input_price_per_mtok: float | None = None
    output_price_per_mtok: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ChatClient(Protocol):
    def chat(self, call: ChatCall) -> CallResult: ...


@dataclass
class EmbedCall:
    """One request for embedding vectors, one vector per input text."""

    model: str
    inputs: list[str]
    timeout_s: float
    keep_alive: str | None = None
    num_ctx: int | None = None


@dataclass
class EmbedResult:
    """The call as telemetry sees it, plus the vectors (empty when the call failed)."""

    call: CallResult
    vectors: list[list[float]] = field(default_factory=list)


def _elapsed_ms(started_at: float) -> float:
    return round((time.monotonic() - started_at) * 1000, 1)


def _error_result(
    error_type: str, message: str, started_at: float, secret: str | None = None
) -> CallResult:
    safe_message = message
    if secret:
        safe_message = safe_message.replace(secret, "[redacted]")
    return CallResult(
        status=STATUS_ERROR,
        error_type=error_type,
        error_message=safe_message[:MAX_ERROR_MESSAGE_CHARS],
        latency_ms=_elapsed_ms(started_at),
    )


def _classify_http_status(status_code: int) -> str:
    if status_code == 429:
        return ERROR_HTTP_429
    if status_code >= 500:
        return ERROR_HTTP_5XX
    return ERROR_HTTP_4XX


def _estimate_cost(call: ChatCall, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
    if call.input_price_per_mtok is None or call.output_price_per_mtok is None:
        return None
    input_cost = (prompt_tokens or 0) * call.input_price_per_mtok / TOKENS_PER_MILLION
    output_cost = (completion_tokens or 0) * call.output_price_per_mtok / TOKENS_PER_MILLION
    return round(input_cost + output_cost, 8)


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def describe_finish(result: CallResult, refusal: str | None = None) -> str:
    """Why a reply stopped, for error messages: finish reasons and the token split."""
    parts = [f"finish_reason={result.finish_reason}"]
    if result.native_finish_reason is not None:
        parts.append(f"native_finish_reason={result.native_finish_reason}")
    parts.append(f"completion_tokens={result.completion_tokens}")
    parts.append(f"reasoning_tokens={result.reasoning_tokens}")
    if refusal:
        parts.append(f"refusal={' '.join(refusal.split())[:MAX_REFUSAL_CHARS]!r}")
    return ", ".join(parts)


def _nanoseconds_to_ms(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(value / NANOSECONDS_PER_MILLISECOND, 1)


class OllamaNativeClient:
    def __init__(self, base_url: str, transport: httpx.BaseTransport | None = None) -> None:
        self._chat_url = base_url.rstrip("/") + "/api/chat"
        self._embed_url = base_url.rstrip("/") + "/api/embed"
        self._transport = transport

    def _build_payload(self, call: ChatCall) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if call.num_ctx is not None:
            options["num_ctx"] = call.num_ctx
        if call.temperature is not None:
            options["temperature"] = call.temperature
        if call.max_output_tokens is not None:
            options["num_predict"] = call.max_output_tokens

        payload: dict[str, Any] = {
            "model": call.model,
            "messages": call.messages,
            "stream": False,
        }
        if options:
            payload["options"] = options
        if call.keep_alive is not None:
            payload["keep_alive"] = call.keep_alive
        if call.think is not None:
            payload["think"] = call.think
        if call.json_schema is not None:
            payload["format"] = call.json_schema
        return payload

    def chat(self, call: ChatCall) -> CallResult:
        started_at = time.monotonic()
        missing_variable = unresolved_model_variable(call.model)
        if missing_variable is not None:
            return _error_result(
                ERROR_MISSING_MODEL,
                f"environment variable {missing_variable} is not set",
                started_at,
            )
        try:
            with httpx.Client(timeout=call.timeout_s, transport=self._transport) as http_client:
                response = http_client.post(self._chat_url, json=self._build_payload(call))
        except httpx.TimeoutException as timeout_error:
            return _error_result(ERROR_TIMEOUT, repr(timeout_error), started_at)
        except httpx.TransportError as transport_error:
            return _error_result(ERROR_CONNECTION, repr(transport_error), started_at)
        except Exception as unexpected_error:  # client boundary: never raise raw
            return _error_result(ERROR_CLIENT, repr(unexpected_error), started_at)

        if response.status_code >= 400:
            return _error_result(
                _classify_http_status(response.status_code),
                f"HTTP {response.status_code}: {response.text}",
                started_at,
            )

        try:
            body = response.json()
            text = body["message"]["content"] or ""
        except (json.JSONDecodeError, KeyError, TypeError) as parse_error:
            return _error_result(ERROR_INVALID_RESPONSE, repr(parse_error), started_at)

        prompt_tokens = body.get("prompt_eval_count")
        completion_tokens = body.get("eval_count")
        estimated_cost = _estimate_cost(call, prompt_tokens, completion_tokens)
        if estimated_cost is None:
            cost_usd = 0.0
            cost_source = "local_zero"
        else:
            cost_usd = estimated_cost
            cost_source = "estimated"

        result = CallResult(
            text=text,
            status=STATUS_OK,
            latency_ms=_elapsed_ms(started_at),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=body.get("prompt_eval_cached_count"),
            cost_usd=cost_usd,
            cost_source=cost_source,
            ollama_load_duration_ms=_nanoseconds_to_ms(body.get("load_duration")),
            ollama_total_duration_ms=_nanoseconds_to_ms(body.get("total_duration")),
            ollama_prompt_eval_ms=_nanoseconds_to_ms(body.get("prompt_eval_duration")),
            ollama_eval_ms=_nanoseconds_to_ms(body.get("eval_duration")),
            finish_reason=_optional_text(body.get("done_reason")),
        )
        if not text.strip():
            result.status = STATUS_ERROR
            result.error_type = ERROR_EMPTY
            result.error_message = f"model returned no content (done_reason={body.get('done_reason')})"
        return result


    def embed(self, call: EmbedCall) -> EmbedResult:
        """Native /api/embed. Fail-closed: a wrong vector count or shape is an error."""
        started_at = time.monotonic()
        payload: dict[str, Any] = {"model": call.model, "input": call.inputs}
        if call.keep_alive is not None:
            payload["keep_alive"] = call.keep_alive
        if call.num_ctx is not None:
            payload["options"] = {"num_ctx": call.num_ctx}
        try:
            with httpx.Client(timeout=call.timeout_s, transport=self._transport) as http_client:
                response = http_client.post(self._embed_url, json=payload)
        except httpx.TimeoutException as timeout_error:
            return EmbedResult(_error_result(ERROR_TIMEOUT, repr(timeout_error), started_at))
        except httpx.TransportError as transport_error:
            return EmbedResult(_error_result(ERROR_CONNECTION, repr(transport_error), started_at))
        except Exception as unexpected_error:  # client boundary: never raise raw
            return EmbedResult(_error_result(ERROR_CLIENT, repr(unexpected_error), started_at))

        if response.status_code >= 400:
            return EmbedResult(
                _error_result(
                    _classify_http_status(response.status_code),
                    f"HTTP {response.status_code}: {response.text}",
                    started_at,
                )
            )

        try:
            body = response.json()
            vectors = body["embeddings"]
            if not isinstance(vectors, list) or len(vectors) != len(call.inputs):
                raise TypeError("embeddings does not hold one vector per input")
            for vector in vectors:
                if not isinstance(vector, list) or not vector:
                    raise TypeError("an embedding is not a non-empty list")
                if not all(isinstance(value, (int, float)) for value in vector):
                    raise TypeError("an embedding holds a non-numeric value")
        except (json.JSONDecodeError, KeyError, TypeError) as parse_error:
            return EmbedResult(_error_result(ERROR_INVALID_RESPONSE, repr(parse_error), started_at))

        call_result = CallResult(
            status=STATUS_OK,
            latency_ms=_elapsed_ms(started_at),
            prompt_tokens=body.get("prompt_eval_count"),
            cost_usd=0.0,
            cost_source="local_zero",
            ollama_load_duration_ms=_nanoseconds_to_ms(body.get("load_duration")),
            ollama_total_duration_ms=_nanoseconds_to_ms(body.get("total_duration")),
        )
        return EmbedResult(call_result, vectors)


class OpenRouterClient:
    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        sdk_client: Any | None = None,
    ) -> None:
        self._base_url = base_url
        self._api_key_env = api_key_env
        self._sdk_client = sdk_client

    def _build_kwargs(self, call: ChatCall) -> dict[str, Any]:
        extra_body: dict[str, Any] = {"usage": {"include": True}}
        if call.effort is not None:
            extra_body["reasoning"] = {"effort": call.effort}

        kwargs: dict[str, Any] = {
            "model": call.model,
            "messages": call.messages,
            "timeout": call.timeout_s,
            "extra_body": extra_body,
        }
        if call.temperature is not None:
            kwargs["temperature"] = call.temperature
        if call.max_output_tokens is not None:
            kwargs["max_tokens"] = call.max_output_tokens
        if call.json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": call.json_schema},
            }
        return kwargs

    def chat(self, call: ChatCall) -> CallResult:
        started_at = time.monotonic()
        missing_variable = unresolved_model_variable(call.model)
        if missing_variable is not None:
            return _error_result(
                ERROR_MISSING_MODEL,
                f"environment variable {missing_variable} is not set",
                started_at,
            )
        api_key = os.environ.get(self._api_key_env, "")
        sdk_client = self._sdk_client
        if sdk_client is None:
            if not api_key:
                return _error_result(
                    ERROR_MISSING_API_KEY,
                    f"environment variable {self._api_key_env} is not set",
                    started_at,
                )
            sdk_client = openai.OpenAI(base_url=self._base_url, api_key=api_key, max_retries=0)

        try:
            response = sdk_client.chat.completions.create(**self._build_kwargs(call))
        except openai.APITimeoutError as timeout_error:
            return _error_result(ERROR_TIMEOUT, repr(timeout_error), started_at, api_key)
        except openai.APIConnectionError as connection_error:
            return _error_result(ERROR_CONNECTION, repr(connection_error), started_at, api_key)
        except openai.APIStatusError as status_error:
            return _error_result(
                _classify_http_status(status_error.status_code),
                f"HTTP {status_error.status_code}: {status_error.message}",
                started_at,
                api_key,
            )
        except Exception as unexpected_error:  # client boundary: never raise raw
            return _error_result(ERROR_CLIENT, repr(unexpected_error), started_at, api_key)

        try:
            choice = response.choices[0]
            text = choice.message.content or ""
        except (AttributeError, IndexError, TypeError) as parse_error:
            return _error_result(ERROR_INVALID_RESPONSE, repr(parse_error), started_at, api_key)
        refusal = _optional_text(getattr(choice.message, "refusal", None))

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = getattr(prompt_details, "cached_tokens", None)
        completion_details = getattr(usage, "completion_tokens_details", None)

        reported_cost = getattr(usage, "cost", None)
        cost_usd: float | None
        cost_source: str | None
        if isinstance(reported_cost, (int, float)):
            cost_usd = float(reported_cost)
            cost_source = "provider_reported"
        else:
            cost_usd = _estimate_cost(call, prompt_tokens, completion_tokens)
            cost_source = "estimated" if cost_usd is not None else None

        result = CallResult(
            text=text,
            status=STATUS_OK,
            latency_ms=_elapsed_ms(started_at),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
            cost_source=cost_source,
            finish_reason=_optional_text(getattr(choice, "finish_reason", None)),
            native_finish_reason=_optional_text(getattr(choice, "native_finish_reason", None)),
            reasoning_tokens=_optional_count(getattr(completion_details, "reasoning_tokens", None)),
        )
        if result.finish_reason == FINISH_REASON_REFUSED:
            # A refusal is final. Any partial text is dropped: it is not an answer.
            result.text = ""
            result.status = STATUS_ERROR
            result.error_type = ERROR_REFUSED
            result.error_message = (
                f"the model refused the request ({describe_finish(result, refusal)})"
            )[:MAX_ERROR_MESSAGE_CHARS]
        elif not text.strip():
            result.status = STATUS_ERROR
            result.error_type = ERROR_EMPTY
            result.error_message = (
                f"model returned no content ({describe_finish(result, refusal)})"
            )[:MAX_ERROR_MESSAGE_CHARS]
        return result


@dataclass
class DecisionsCall:
    """One request to a decisions model: typed questions about the supplied state."""

    model: str
    state: dict[str, Any]
    questions: dict[str, dict[str, Any]]
    timeout_s: float


class OpenRouterDecisionsClient:
    """OpenRouter Decisions API (alpha). Returns the typed answers as JSON text.

    The API shape may change; a response without an `answers` object is reported as
    invalid_response, never as an answer.
    """

    def __init__(
        self, base_url: str, api_key_env: str, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._decisions_url = base_url.rstrip("/") + "/alpha/decisions"
        self._api_key_env = api_key_env
        self._transport = transport

    def decide(self, call: DecisionsCall) -> CallResult:
        started_at = time.monotonic()
        api_key = os.environ.get(self._api_key_env, "")
        if not api_key:
            return _error_result(
                ERROR_MISSING_API_KEY,
                f"environment variable {self._api_key_env} is not set",
                started_at,
            )

        payload = {"model": call.model, "state": call.state, "questions": call.questions}
        try:
            with httpx.Client(timeout=call.timeout_s, transport=self._transport) as http_client:
                response = http_client.post(
                    self._decisions_url,
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
        except httpx.TimeoutException as timeout_error:
            return _error_result(ERROR_TIMEOUT, repr(timeout_error), started_at, api_key)
        except httpx.TransportError as transport_error:
            return _error_result(ERROR_CONNECTION, repr(transport_error), started_at, api_key)
        except Exception as unexpected_error:  # client boundary: never raise raw
            return _error_result(ERROR_CLIENT, repr(unexpected_error), started_at, api_key)

        if response.status_code >= 400:
            return _error_result(
                _classify_http_status(response.status_code),
                f"HTTP {response.status_code}: {response.text}",
                started_at,
                api_key,
            )

        try:
            body = response.json()
            answers = body["answers"]
            if not isinstance(answers, dict) or not answers:
                raise TypeError("answers is not a non-empty object")
        except (json.JSONDecodeError, KeyError, TypeError) as parse_error:
            return _error_result(ERROR_INVALID_RESPONSE, repr(parse_error), started_at, api_key)

        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        reported_cost = usage.get("cost")
        has_reported_cost = isinstance(reported_cost, (int, float))
        return CallResult(
            text=json.dumps(answers),
            status=STATUS_OK,
            latency_ms=_elapsed_ms(started_at),
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            cost_usd=float(reported_cost) if has_reported_cost else None,
            cost_source="provider_reported" if has_reported_cost else None,
        )


AnyClient = ChatClient | OpenRouterDecisionsClient


def build_client(provider: ProviderConfig) -> AnyClient:
    if provider.kind == "ollama_native":
        return OllamaNativeClient(provider.base_url)
    if provider.kind == "openrouter":
        return OpenRouterClient(provider.base_url, provider.api_key_env or "")
    if provider.kind == "openrouter_decisions":
        return OpenRouterDecisionsClient(provider.base_url, provider.api_key_env or "")
    raise ValueError(f"unknown provider kind '{provider.kind}'")


def build_clients(config: HarnessConfig) -> dict[str, AnyClient]:
    return {name: build_client(provider) for name, provider in config.providers.items()}
