"""Decisions-model triage: client behaviour, truth table and fail-closed handling.

No network: the Decisions API is replaced by an httpx MockTransport.
"""

from __future__ import annotations

import copy
import json

import httpx
import pytest

from inference_arbiter import jev_triage
from inference_arbiter.clients import DecisionsCall, OpenRouterDecisionsClient
from inference_arbiter.config import HarnessConfig
from inference_arbiter.schemas import Message
from inference_arbiter.triage import run_triage

from conftest import BASE_CONFIG

KEY_ENV = "TEST_DECISIONS_KEY"
FAKE_KEY = "test-key-not-a-real-secret"


def answers_body(is_code: float, depth_probabilities: dict[str, float]) -> dict:
    return {
        "id": "gen-dec-test",
        "model": "typesafe/jev-test",
        "answers": {
            "is_code": {"type": "noul", "noul": is_code},
            "reasoning_depth": {
                "type": "score",
                "score": 0.0,
                "confidence": 0.9,
                "probabilities": depth_probabilities,
            },
        },
        "usage": {"input_tokens": 120, "output_tokens": 20, "cost": 0.000005},
    }


def jev_config() -> HarnessConfig:
    raw_config = copy.deepcopy(BASE_CONFIG)
    raw_config["providers"]["decisions"] = {
        "kind": "openrouter_decisions",
        "base_url": "http://decisions.invalid/api",
        "api_key_env": KEY_ENV,
    }
    raw_config["tiers"]["triage_jev"] = {
        "model": "typesafe/jev-test",
        "provider": "decisions",
        "role": "triage",
        "cloud": True,
        "timeout_s": 5,
    }
    raw_config["triage_tier"] = "triage_jev"
    return HarnessConfig.model_validate(raw_config)


def client_returning(handler) -> OpenRouterDecisionsClient:
    return OpenRouterDecisionsClient(
        "http://decisions.invalid/api", KEY_ENV, transport=httpx.MockTransport(handler)
    )


def triage_with(monkeypatch, handler, question: str = "What is the capital of Australia?"):
    monkeypatch.setenv(KEY_ENV, FAKE_KEY)
    messages = [Message(role="user", content=question)]
    return run_triage(jev_config(), client_returning(handler), messages)


@pytest.mark.parametrize(
    ("is_code", "depth_level", "depth_probability", "expected_label", "expected_confidence"),
    [
        (0.02, 0, 0.95, "local", 0.95),
        (0.02, 1, 0.80, "local", 0.80),
        (0.02, 2, 0.85, "mid", 0.85),
        (0.95, 1, 0.90, "code", 0.95),
        (0.95, 3, 0.70, "frontier", 0.70),  # expert depth wins over code
        (0.05, 3, 0.60, "frontier", 0.60),
        (0.50, 0, 0.95, "local", 0.50),  # code judgment not settled: confidence drops
    ],
)
def test_truth_table(is_code, depth_level, depth_probability, expected_label, expected_confidence):
    decision = jev_triage.decide_route(is_code, depth_level, depth_probability)
    assert decision.label == expected_label
    assert decision.confidence == pytest.approx(expected_confidence)


def test_request_sends_only_the_question_text_and_typed_questions(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=answers_body(0.01, {"0": 0.9, "1": 0.1, "2": 0, "3": 0}))

    outcome = triage_with(monkeypatch, handler)

    assert seen["url"] == "http://decisions.invalid/api/alpha/decisions"
    assert seen["auth"] == f"Bearer {FAKE_KEY}"
    assert seen["body"]["model"] == "typesafe/jev-test"
    assert seen["body"]["state"] == {"user_question": "What is the capital of Australia?"}
    assert set(seen["body"]["questions"]) == {"is_code", "reasoning_depth"}
    assert outcome.status == "ok"
    assert outcome.decision.label == "local"
    assert outcome.call.cost_usd == pytest.approx(0.000005)
    assert outcome.call.cost_source == "provider_reported"
    assert outcome.call.prompt_tokens == 120


def test_missing_key_is_an_error_not_a_label(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    client = client_returning(lambda request: httpx.Response(200, json={}))
    result = client.decide(DecisionsCall("m", {"user_question": "q"}, jev_triage.QUESTIONS, 5))
    assert result.status == "error"
    assert result.error_type == "missing_api_key"


def test_timeout_fails_closed(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    outcome = triage_with(monkeypatch, handler)
    assert outcome.status == "timeout"
    assert outcome.decision is None


def test_http_error_fails_closed_and_redacts_the_key(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, text=f"no credits for {FAKE_KEY}")

    outcome = triage_with(monkeypatch, handler)
    assert outcome.status == "error"
    assert outcome.call.error_type == "http_4xx"
    assert FAKE_KEY not in outcome.call.error_message


@pytest.mark.parametrize(
    "body",
    [
        {"unexpected": "shape"},
        {"answers": {}},
        {"answers": {"is_code": {"type": "noul", "noul": 0.1}}},  # depth answer missing
        {"answers": {"is_code": {"noul": 0.1}, "reasoning_depth": {"probabilities": {"9": 1.0}}}},
    ],
)
def test_unexpected_response_shape_is_never_a_label(monkeypatch, body):
    outcome = triage_with(monkeypatch, lambda request: httpx.Response(200, json=body))
    assert outcome.status in ("error", "invalid")
    assert outcome.decision is None


def test_decisions_tier_cannot_be_an_answer_tier():
    raw_config = copy.deepcopy(BASE_CONFIG)
    raw_config["providers"]["decisions"] = {
        "kind": "openrouter_decisions",
        "base_url": "http://decisions.invalid/api",
        "api_key_env": KEY_ENV,
    }
    raw_config["tiers"]["bad"] = {"model": "m", "provider": "decisions", "timeout_s": 5}
    with pytest.raises(ValueError, match="must set role: triage"):
        HarnessConfig.model_validate(raw_config)


def test_triage_tier_must_be_a_triage_tier():
    raw_config = copy.deepcopy(BASE_CONFIG)
    raw_config["triage_tier"] = "local"
    with pytest.raises(ValueError, match="is not a triage tier"):
        HarnessConfig.model_validate(raw_config)
