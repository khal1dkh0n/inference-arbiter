"""Grounding: the deterministic quote check, the Jev and chat judgments, the truth table."""

from __future__ import annotations

import json

import httpx
import pytest

from inference_arbiter import grounding
from inference_arbiter.clients import OpenRouterDecisionsClient
from inference_arbiter.config import TierConfig
from inference_arbiter.schemas import CallResult, Claim
from inference_arbiter.sources import Passage

from conftest import FakeClient, error_result, ok_result

QUESTION = "What confidence value triggers escalation?"

PASSAGES = [
    Passage(
        passage_id="S1-P1",
        source_index=0,
        source="guide.md",
        position=0,
        heading="Thresholds",
        text="## Thresholds\n\nThe router escalates when\nconfidence is below 0.80.",
    ),
    Passage(
        passage_id="S2-P1",
        source_index=1,
        source="models.md",
        position=0,
        heading="Code",
        text="The code specialist is qwen2.5-coder:32b.",
    ),
]

GOOD_CLAIM = Claim(
    text="Escalation happens below a confidence of 0.80.",
    passage_id="S1-P1",
    quote="The router escalates when confidence is below 0.80.",
)
SECOND_CLAIM = Claim(
    text="The code specialist is qwen2.5-coder:32b.",
    passage_id="S2-P1",
    quote="the code specialist is QWEN2.5-coder:32b",
)

JEV_TIER = TierConfig(model="jev-model", provider="openrouter_decisions", role="verifier", timeout_s=5)
CHAT_TIER = TierConfig(model="ground-model", provider="ollama", role="verifier", timeout_s=30)


def jev_answers(answerable: float, *claim_probabilities: float) -> CallResult:
    answers = {grounding.QUESTION_ANSWERABLE: {"noul": answerable}}
    for claim_number, probability in enumerate(claim_probabilities, start=1):
        answers[grounding.claim_key(claim_number)] = {"noul": probability}
    return ok_result(json.dumps(answers))


def chat_report(answerable: str, *verdicts: str) -> CallResult:
    return ok_result(
        json.dumps(
            {
                "claims": [
                    {"claim_number": number, "reason": f"reason {number}", "verdict": verdict}
                    for number, verdict in enumerate(verdicts, start=1)
                ],
                "answerable": answerable,
            }
        )
    )


# --- deterministic quote check ---


def test_quote_matches_after_whitespace_and_case_normalisation():
    checks = grounding.check_quotes([GOOD_CLAIM, SECOND_CLAIM], PASSAGES)

    assert [check.quote_ok for check in checks] == [True, True]
    assert [check.quote_reason for check in checks] == [None, None]


def test_fabricated_quote_is_caught_without_a_model():
    fabricated = Claim(
        text="Escalation happens below 0.50.",
        passage_id="S1-P1",
        quote="The router escalates when confidence is below 0.50.",
    )

    [check] = grounding.check_quotes([fabricated], PASSAGES)

    assert check.quote_ok is False
    assert check.quote_reason == "quote_not_in_passage"
    assert check.supported is False


def test_quote_from_a_different_passage_than_the_cited_one_fails():
    wrong_passage = Claim(text="x", passage_id="S2-P1", quote="The router escalates when confidence is below 0.80.")

    [check] = grounding.check_quotes([wrong_passage], PASSAGES)

    assert check.quote_reason == "quote_not_in_passage"


def test_unknown_passage_id_fails():
    [check] = grounding.check_quotes(
        [Claim(text="x", passage_id="S9-P9", quote="The router escalates when")], PASSAGES
    )

    assert check.quote_ok is False
    assert check.quote_reason == "unknown_passage"


@pytest.mark.parametrize("quote", ["", "   ", "0.80", "the"])
def test_trivially_short_quote_fails(quote):
    [check] = grounding.check_quotes([Claim(text="x", passage_id="S1-P1", quote=quote)], PASSAGES)

    assert check.quote_ok is False
    assert check.quote_reason == "quote_too_short"


def test_a_claim_needs_both_the_quote_check_and_a_supported_verdict():
    passed_quote = grounding.ClaimCheck(GOOD_CLAIM, quote_ok=True)
    assert passed_quote.supported is False

    passed_quote.verdict = "uncertain"
    assert passed_quote.supported is False

    passed_quote.verdict = "supported"
    assert passed_quote.supported is True

    failed_quote = grounding.ClaimCheck(GOOD_CLAIM, quote_ok=False, verdict="supported")
    assert failed_quote.supported is False


# --- Jev: thresholds ---


@pytest.mark.parametrize(
    ("probability", "verdict", "answerability"),
    [
        (1.0, "supported", "yes"),
        (0.7, "supported", "yes"),
        (0.69, "uncertain", "uncertain"),
        (0.5, "uncertain", "uncertain"),
        (0.31, "uncertain", "uncertain"),
        (0.3, "unsupported", "no"),
        (0.0, "unsupported", "no"),
    ],
)
def test_noul_thresholds(probability, verdict, answerability):
    assert grounding.NOUL_YES == 0.7
    assert grounding.NOUL_NO == 0.3
    assert grounding.noul_to_verdict(probability) == verdict
    assert grounding.noul_to_answerability(probability) == answerability


# --- the truth table ---


@pytest.mark.parametrize(
    ("drafter_answerable", "total", "supported", "answerability", "status", "action"),
    [
        # row 1: the verifier positively says the passages hold the answer
        (False, 0, 0, "yes", "ungrounded", "escalate"),
        # row 2: the verifier agrees or is not sure, so the abstention stands
        (False, 0, 0, "no", "ungrounded", "abstain"),
        (False, 0, 0, "uncertain", "ungrounded", "abstain"),
        # row 3: failed answerability judgment, whatever the claims say
        (True, 2, 2, "no", "ungrounded", "escalate"),
        # row 4: no supported claim
        (True, 2, 0, "yes", "ungrounded", "escalate"),
        (True, 0, 0, "yes", "ungrounded", "escalate"),
        # row 5: all supported
        (True, 2, 2, "yes", "grounded", "return"),
        (True, 1, 1, "uncertain", "grounded", "return"),
        # row 6: a mix
        (True, 3, 1, "yes", "partly_grounded", "escalate"),
        (True, 2, 1, "uncertain", "partly_grounded", "escalate"),
    ],
)
def test_truth_table(drafter_answerable, total, supported, answerability, status, action):
    decision = grounding.decide(drafter_answerable, total, supported, answerability)

    assert (decision.status, decision.action) == (status, action)


# --- Jev: request and answers ---


def test_jev_request_has_one_noul_per_claim_plus_answerability_and_refers_to_state_paths():
    state, questions = grounding.build_jev_request(QUESTION, PASSAGES, [GOOD_CLAIM, SECOND_CLAIM])

    assert list(questions) == ["passages_answer_question", "claim_1", "claim_2"]
    assert all(question["type"] == "noul" for question in questions.values())
    assert state["user_question"] == QUESTION
    assert state["passages"] == {"S1-P1": PASSAGES[0].text, "S2-P1": PASSAGES[1].text}
    assert state["claims"]["claim_1"] == {"statement": GOOD_CLAIM.text, "cited_passage": PASSAGES[0].text}
    assert state["claims"]["claim_2"]["cited_passage"] == PASSAGES[1].text
    assert "`claims.claim_2.cited_passage`" in questions["claim_2"]["instructions"]
    assert "`claims.claim_2.statement`" in questions["claim_2"]["instructions"]
    assert "state or directly entail" in questions["claim_1"]["instructions"]
    assert "`passages`" in questions["passages_answer_question"]["instructions"]
    assert "`user_question`" in questions["passages_answer_question"]["instructions"]
    for question in questions.values():
        assert set(question["criteria"]) == {"true", "false"}


def test_jev_request_without_claims_asks_only_about_answerability():
    state, questions = grounding.build_jev_request(QUESTION, PASSAGES, [])

    assert list(questions) == ["passages_answer_question"]
    assert "claims" not in state


def test_jev_answers_become_per_claim_verdicts():
    outcome = grounding.interpret_jev_answers(jev_answers(0.95, 0.9, 0.5, 0.1), claim_count=3)

    assert outcome.status == "ok"
    assert outcome.judgment.claim_verdicts == ["supported", "uncertain", "unsupported"]
    assert outcome.judgment.claim_probabilities == [0.9, 0.5, 0.1]
    assert outcome.judgment.answerability == "yes"
    assert outcome.judgment.answerable_probability == 0.95


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        json.dumps({"claim_1": {"noul": 0.9}}),  # answerability missing
        json.dumps({"passages_answer_question": {"noul": 0.9}}),  # a claim missing
        json.dumps({"passages_answer_question": {"noul": 0.9}, "claim_1": {"noul": 1.4}}),
        json.dumps({"passages_answer_question": {"noul": 0.9}, "claim_1": {"choice": "yes"}}),
        json.dumps({"passages_answer_question": {"noul": 0.9}, "claim_1": {"noul": None}}),
        json.dumps(["a list"]),
    ],
)
def test_unusable_jev_answers_fail_closed(text):
    outcome = grounding.interpret_jev_answers(ok_result(text), claim_count=1)

    assert outcome.status == "invalid"
    assert outcome.judgment is None
    assert outcome.call.error_type == "grounding_invalid"


def test_failed_jev_call_is_an_error_never_a_verdict():
    failed = grounding.interpret_jev_answers(error_result("http_5xx"), claim_count=1)
    timed_out = grounding.interpret_jev_answers(error_result("timeout"), claim_count=1)

    assert (failed.status, failed.judgment) == ("error", None)
    assert (timed_out.status, timed_out.judgment) == ("timeout", None)


def test_jev_verifier_sends_every_judgment_in_one_request(monkeypatch):
    monkeypatch.setenv("TEST_DECISIONS_KEY", "test-key-value")
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "answers": {
                    "passages_answer_question": {"noul": 0.92},
                    "claim_1": {"noul": 0.88},
                    "claim_2": {"noul": 0.2},
                }
            },
        )

    client = OpenRouterDecisionsClient(
        "http://decisions.invalid", "TEST_DECISIONS_KEY", transport=httpx.MockTransport(handler)
    )
    outcome = grounding.run_grounding_verifier(
        JEV_TIER, client, QUESTION, PASSAGES, [GOOD_CLAIM, SECOND_CLAIM]
    )

    assert len(requests) == 1
    assert requests[0]["model"] == "jev-model"
    assert set(requests[0]["questions"]) == {"passages_answer_question", "claim_1", "claim_2"}
    assert outcome.judgment.claim_verdicts == ["supported", "unsupported"]
    assert outcome.judgment.answerability == "yes"


def test_jev_verifier_without_a_key_fails_closed(monkeypatch):
    monkeypatch.delenv("TEST_DECISIONS_KEY", raising=False)
    client = OpenRouterDecisionsClient("http://decisions.invalid", "TEST_DECISIONS_KEY")

    outcome = grounding.run_grounding_verifier(JEV_TIER, client, QUESTION, PASSAGES, [GOOD_CLAIM])

    assert outcome.status == "error"
    assert outcome.judgment is None
    assert outcome.call.error_type == "missing_api_key"


# --- chat verifier ---


def test_chat_verifier_call_carries_the_passages_the_claims_and_the_schema():
    client = FakeClient()
    client.script("ground-model", chat_report("yes", "supported", "unsupported"))

    outcome = grounding.run_grounding_verifier(
        CHAT_TIER, client, QUESTION, PASSAGES, [GOOD_CLAIM, SECOND_CLAIM]
    )

    [call] = client.calls
    user_text = call.messages[1]["content"]
    assert call.messages[0]["content"] == grounding.GROUNDING_SYSTEM_PROMPT
    assert QUESTION in user_text
    assert "[S1-P1] source: guide.md | section: Thresholds" in user_text
    assert "1. (cites S1-P1) Escalation happens below a confidence of 0.80." in user_text
    assert "2. (cites S2-P1)" in user_text
    assert call.temperature == 0.0
    assert call.json_schema == grounding.GroundingReport.model_json_schema()
    assert outcome.status == "ok"
    assert outcome.judgment.claim_verdicts == ["supported", "unsupported"]
    assert outcome.judgment.claim_reasons == ["reason 1", "reason 2"]
    assert outcome.judgment.answerability == "yes"


def test_chat_report_entries_are_matched_by_claim_number_not_position():
    report = ok_result(
        json.dumps(
            {
                "claims": [
                    {"claim_number": 2, "reason": "second", "verdict": "unsupported"},
                    {"claim_number": 1, "reason": "first", "verdict": "supported"},
                ],
                "answerable": "uncertain",
            }
        )
    )

    outcome = grounding.interpret_chat_report(report, claim_count=2)

    assert outcome.judgment.claim_verdicts == ["supported", "unsupported"]
    assert outcome.judgment.answerability == "uncertain"


@pytest.mark.parametrize(
    "report",
    [
        ok_result("not json"),
        chat_report("yes", "supported"),  # one of two claims not judged
        chat_report("yes", "supported", "supported", "supported"),  # a claim that does not exist
        chat_report("maybe", "supported", "supported"),
        chat_report("yes", "supported", "probably"),
        ok_result(
            json.dumps(
                {
                    "claims": [
                        {"claim_number": 1, "reason": "a", "verdict": "supported"},
                        {"claim_number": 1, "reason": "b", "verdict": "supported"},
                    ],
                    "answerable": "yes",
                }
            )
        ),
    ],
)
def test_incomplete_or_malformed_chat_report_fails_closed(report):
    outcome = grounding.interpret_chat_report(report, claim_count=2)

    assert outcome.status == "invalid"
    assert outcome.judgment is None
    assert outcome.call.error_type == "grounding_invalid"


def test_failed_chat_verifier_call_is_an_error():
    outcome = grounding.interpret_chat_report(error_result("timeout"), claim_count=1)

    assert outcome.status == "timeout"
    assert outcome.judgment is None


def test_chat_verifier_with_no_claims_still_judges_answerability():
    client = FakeClient()
    client.script("ground-model", chat_report("no"))

    outcome = grounding.run_grounding_verifier(CHAT_TIER, client, QUESTION, PASSAGES, [])

    assert "(no claims" in client.calls[0].messages[1]["content"]
    assert outcome.judgment.claim_verdicts == []
    assert outcome.judgment.answerability == "no"


# --- applying and describing ---


def test_verdicts_are_applied_only_to_claims_that_passed_the_quote_check():
    fabricated = Claim(text="x", passage_id="S1-P1", quote="words that are not in the passage")
    checks = grounding.check_quotes([fabricated, GOOD_CLAIM, SECOND_CLAIM], PASSAGES)
    judgment = grounding.interpret_jev_answers(jev_answers(0.9, 0.95, 0.4), claim_count=2).judgment

    grounding.apply_judgment(checks, judgment)

    assert [check.verdict for check in checks] == [None, "supported", "uncertain"]
    assert [check.supported for check in checks] == [False, True, False]
    notes = grounding.describe_checks(checks, judgment)
    assert notes == [
        "claim 1 [S1-P1]: quote check failed (quote_not_in_passage)",
        "claim 3 [S2-P1]: uncertain (p=0.40)",
        "passages answer the question: yes (p=0.90)",
    ]


@pytest.mark.parametrize(
    ("drafter_answerable", "escalate_on_verifier_error", "action"),
    [
        # row 7: an abstention nobody disputed is never escalated
        (False, False, "abstain"),
        (False, True, "abstain"),
        # row 8
        (True, False, "return"),
        # row 9
        (True, True, "escalate"),
    ],
)
def test_truth_table_without_a_verdict(drafter_answerable, escalate_on_verifier_error, action):
    decision = grounding.decide_unverified(drafter_answerable, escalate_on_verifier_error)

    assert (decision.status, decision.action) == ("unavailable", action)


def test_no_quote_passed_row_escalates_as_ungrounded():
    decision = grounding.DECISION_NO_QUOTE_PASSED

    assert (decision.status, decision.action) == ("ungrounded", "escalate")
