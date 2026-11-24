"""Verifier: chat verdict validation, the Jev truth table and fail-closed handling.

No network: chat calls use the scripted FakeClient and the Decisions API is replaced by
an httpx MockTransport.
"""

from __future__ import annotations

import json

import httpx
import pytest

from inference_arbiter import verifier
from inference_arbiter.clients import OpenRouterDecisionsClient
from inference_arbiter.config import TierConfig
from inference_arbiter.schemas import CallResult, DraftAnswer, VerifierVerdict

from conftest import FakeClient, error_result, ok_result, verdict_json

KEY_ENV = "TEST_VERIFIER_DECISIONS_KEY"
FAKE_KEY = "test-key-not-a-real-secret"

QUESTION = "How many legs do three spiders have?"
DRAFT = DraftAnswer(
    answer="Three spiders have 18 legs.",
    key_claims=["A spider has 6 legs.", "3 x 6 = 18."],
    assumptions=[],
)

CHAT_TIER = TierConfig(model="verify-model", provider="ollama", role="verifier", timeout_s=30)
JEV_TIER = TierConfig(model="typesafe/jev-test", provider="decisions", role="verifier", timeout_s=5)


def jev_answers(answers_question: float, false_premise: float, contradiction: float) -> dict:
    return {
        "answers_question": {"type": "noul", "noul": answers_question},
        "unflagged_false_premise": {"type": "noul", "noul": false_premise},
        "self_contradiction": {"type": "noul", "noul": contradiction},
    }


def jev_call_result(answers: dict) -> CallResult:
    return ok_result(json.dumps(answers))


def decisions_client(handler) -> OpenRouterDecisionsClient:
    return OpenRouterDecisionsClient(
        "http://decisions.invalid/api", KEY_ENV, transport=httpx.MockTransport(handler)
    )


# --- chat verifier ---


def test_chat_verifier_call_carries_question_draft_and_verdict_schema():
    call = verifier.build_verifier_call(CHAT_TIER, QUESTION, DRAFT)

    assert call.model == "verify-model"
    assert call.json_schema == VerifierVerdict.model_json_schema()
    assert call.temperature == 0.0
    assert call.messages[0] == {"role": "system", "content": verifier.VERIFIER_SYSTEM_PROMPT}
    user_text = call.messages[1]["content"]
    assert QUESTION in user_text
    assert "Three spiders have 18 legs." in user_text
    assert "A spider has 6 legs." in user_text


@pytest.mark.parametrize("verdict", ["supported", "unsupported", "uncertain"])
def test_chat_verifier_returns_a_validated_verdict(verdict):
    client = FakeClient()
    client.script("verify-model", verdict_json(verdict, "spiders have 8 legs", ["A spider has 6 legs."]))

    outcome = verifier.run_verifier(CHAT_TIER, client, QUESTION, DRAFT)

    assert outcome.status == "ok"
    assert outcome.verdict.verdict == verdict
    assert outcome.verdict.critique == "spiders have 8 legs"
    assert outcome.verdict.failed_claims == ["A spider has 6 legs."]


@pytest.mark.parametrize(
    "bad_text",
    [
        "the draft looks fine to me",
        json.dumps({"verdict": "probably", "critique": "x", "failed_claims": []}),
        json.dumps({"verdict": "supported"}),
        json.dumps({"verdict": "supported", "critique": "x", "failed_claims": [], "score": 9}),
    ],
)
def test_invalid_chat_verdict_is_a_verifier_error_not_a_verdict(bad_text):
    outcome = verifier.interpret_chat_verdict(ok_result(bad_text))

    assert outcome.status == "invalid"
    assert outcome.verdict is None
    assert outcome.call.error_type == verifier.ERROR_VERDICT_INVALID


@pytest.mark.parametrize(
    ("error_type", "expected_status"), [("timeout", "timeout"), ("http_5xx", "error")]
)
def test_failed_chat_verifier_call_has_no_verdict(error_type, expected_status):
    outcome = verifier.interpret_chat_verdict(error_result(error_type))

    assert outcome.status == expected_status
    assert outcome.verdict is None


# --- Jev truth table ---


@pytest.mark.parametrize(
    ("answers_question", "false_premise", "contradiction", "expected_verdict"),
    [
        (0.95, 0.05, 0.05, "supported"),
        (0.70, 0.30, 0.30, "supported"),  # thresholds are inclusive on the settled side
        (0.10, 0.05, 0.05, "unsupported"),  # does not answer the question
        (0.30, 0.05, 0.05, "unsupported"),
        (0.95, 0.90, 0.05, "unsupported"),  # swallows a false premise
        (0.95, 0.70, 0.05, "unsupported"),
        (0.95, 0.05, 0.85, "unsupported"),  # contradicts its key claims
        (0.50, 0.05, 0.05, "uncertain"),  # band on answers_question
        (0.95, 0.50, 0.05, "uncertain"),  # band on false premise
        (0.95, 0.05, 0.31, "uncertain"),  # band on contradiction
        (0.69, 0.05, 0.05, "uncertain"),
        (0.50, 0.90, 0.05, "unsupported"),  # a settled defect wins over an unsettled check
        (0.10, 0.90, 0.90, "unsupported"),
    ],
)
def test_jev_truth_table(answers_question, false_premise, contradiction, expected_verdict):
    verdict = verifier.decide_verdict(answers_question, false_premise, contradiction)

    assert verdict.verdict == expected_verdict
    assert verdict.failed_claims == []
    assert verdict.critique


def test_jev_critique_names_each_failed_check():
    verdict = verifier.decide_verdict(0.10, 0.90, 0.50)

    assert verifier.CRITIQUE_NOT_ANSWERED in verdict.critique
    assert verifier.CRITIQUE_FALSE_PREMISE in verdict.critique
    assert "Not settled whether the draft contradicts" in verdict.critique
    assert verifier.CRITIQUE_CONTRADICTION not in verdict.critique


def test_jev_answers_are_interpreted_into_a_verdict():
    outcome = verifier.interpret_jev_answers(jev_call_result(jev_answers(0.95, 0.90, 0.05)))

    assert outcome.status == "ok"
    assert outcome.verdict.verdict == "unsupported"


# --- Jev fail-closed handling ---


@pytest.mark.parametrize(
    "bad_text",
    [
        "not json",
        json.dumps({"answers_question": {"noul": 0.9}}),  # two judgments missing
        json.dumps(
            {
                "answers_question": {"noul": 0.9},
                "unflagged_false_premise": {"noul": 0.1},
                "self_contradiction": {"choice": "no"},  # wrong answer type
            }
        ),
        json.dumps(jev_answers(1.7, 0.1, 0.1)),  # probability out of range
        json.dumps(
            {
                "answers_question": {"noul": "high"},
                "unflagged_false_premise": {"noul": 0.1},
                "self_contradiction": {"noul": 0.1},
            }
        ),
        json.dumps(["answers_question"]),
    ],
)
def test_unusable_jev_answers_fail_closed(bad_text):
    outcome = verifier.interpret_jev_answers(ok_result(bad_text))

    assert outcome.status == "invalid"
    assert outcome.verdict is None
    assert outcome.call.error_type == verifier.ERROR_VERDICT_INVALID


@pytest.mark.parametrize(
    ("error_type", "expected_status"), [("timeout", "timeout"), ("http_429", "error")]
)
def test_failed_jev_call_fails_closed(error_type, expected_status):
    outcome = verifier.interpret_jev_answers(error_result(error_type))

    assert outcome.status == expected_status
    assert outcome.verdict is None


def test_jev_verifier_sends_state_and_three_typed_questions(monkeypatch):
    monkeypatch.setenv(KEY_ENV, FAKE_KEY)
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "answers": jev_answers(0.95, 0.05, 0.05),
                "usage": {"input_tokens": 200, "output_tokens": 30, "cost": 0.00001},
            },
        )

    outcome = verifier.run_verifier(JEV_TIER, decisions_client(handler), QUESTION, DRAFT)

    assert outcome.status == "ok"
    assert outcome.verdict.verdict == "supported"
    assert outcome.call.cost_usd == 0.00001
    payload = seen_payloads[0]
    assert payload["model"] == "typesafe/jev-test"
    assert payload["state"] == {"user_question": QUESTION, "draft": DRAFT.model_dump()}
    assert set(payload["questions"]) == {
        "answers_question",
        "unflagged_false_premise",
        "self_contradiction",
    }
    assert all(question["type"] == "noul" for question in payload["questions"].values())


def test_jev_verifier_http_failure_and_missing_key_fail_closed(monkeypatch):
    monkeypatch.setenv(KEY_ENV, FAKE_KEY)
    failing_client = decisions_client(lambda request: httpx.Response(503, text="unavailable"))
    outcome = verifier.run_verifier(JEV_TIER, failing_client, QUESTION, DRAFT)
    assert outcome.status == "error"
    assert outcome.verdict is None
    assert FAKE_KEY not in (outcome.call.error_message or "")

    monkeypatch.delenv(KEY_ENV)
    outcome = verifier.run_verifier(JEV_TIER, failing_client, QUESTION, DRAFT)
    assert outcome.status == "error"
    assert outcome.call.error_type == "missing_api_key"


def test_jev_verifier_malformed_response_body_fails_closed(monkeypatch):
    monkeypatch.setenv(KEY_ENV, FAKE_KEY)
    client = decisions_client(lambda request: httpx.Response(200, json={"answers": {}}))

    outcome = verifier.run_verifier(JEV_TIER, client, QUESTION, DRAFT)

    assert outcome.verdict is None
    assert outcome.status == "error"
    assert outcome.call.error_type == "invalid_response"
