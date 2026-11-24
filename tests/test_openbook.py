"""Openbook and stuffed modes through Harness.answer. Fake clients, temporary files, no network."""

from __future__ import annotations

import json
import sqlite3

import httpx
import pytest
from pydantic import ValidationError

from inference_arbiter import cli, openbook
from inference_arbiter.clients import EmbedResult, OpenRouterDecisionsClient
from inference_arbiter.config import HarnessConfig, load_config
from inference_arbiter.router import Harness, OpenbookNotConfiguredError, UnknownTierError
from inference_arbiter.schemas import AskRequest, CallResult
from inference_arbiter.sources import Passage
from inference_arbiter.telemetry import Telemetry

from conftest import (
    BASE_CONFIG,
    REAL_CONFIG_PATH,
    FakeClient,
    FakeTelemetry,
    HarnessKit,
    draft_json,
    error_result,
    make_chain_config,
    make_openbook_config,
    ok_result,
    triage_json,
    verdict_json,
)

QUESTION = "What confidence value triggers escalation, and which model is the code specialist?"

GUIDE_TEXT = """# Router Guide

## Thresholds

The router escalates when confidence is below 0.80.

Timeouts fall back to the mid tier.

## Sandboxes

Phase 3 runs tools inside MicroVM sandboxes.
"""

MODELS_TEXT = """# Model Inventory

## Code

The code specialist is qwen2.5-coder:32b.
"""

THRESHOLD_CLAIM = {
    "text": "Escalation happens when confidence is below 0.80.",
    "passage_id": "S1-P1",
    "quote": "The router escalates when confidence is below 0.80.",
}
CODE_CLAIM = {
    "text": "The code specialist is qwen2.5-coder:32b.",
    "passage_id": "S2-P1",
    "quote": "The code specialist is qwen2.5-coder:32b.",
}
FABRICATED_CLAIM = {
    "text": "Escalation also happens after three retries.",
    "passage_id": "S1-P1",
    "quote": "The router escalates after three retries.",
}
GOOD_ANSWER = "Below 0.80 the router escalates; the code specialist is qwen2.5-coder:32b."


def openbook_draft(
    claims: list[dict] | None = None, answer: str = GOOD_ANSWER, answerable: bool = True
) -> CallResult:
    selected_claims = [THRESHOLD_CLAIM, CODE_CLAIM] if claims is None else claims
    return ok_result(
        json.dumps(
            {"working": "read the passages", "answerable": answerable, "answer": answer, "claims": selected_claims}
        )
    )


def unanswerable_draft() -> CallResult:
    return openbook_draft(claims=[], answer="The passages do not say.", answerable=False)


def grounding_report(answerable: str, *verdicts: str) -> CallResult:
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


@pytest.fixture
def source_paths(tmp_path) -> list[str]:
    guide_path = tmp_path / "guide.md"
    models_path = tmp_path / "models.md"
    guide_path.write_text(GUIDE_TEXT, encoding="utf-8")
    models_path.write_text(MODELS_TEXT, encoding="utf-8")
    return [str(guide_path), str(models_path)]


@pytest.fixture
def ob_kit(tmp_path) -> HarnessKit:
    return HarnessKit(make_openbook_config(tmp_path / "cache"))


def ask(source_paths: list[str], **overrides) -> AskRequest:
    return AskRequest.from_question(QUESTION, mode="openbook", sources=source_paths, **overrides)


def call_trail(kit: HarnessKit) -> list[tuple[str, str, str]]:
    return [(record.role, record.tier, record.status) for record in kit.call_records]


# --- request contract ---


def test_openbook_and_stuffed_need_sources():
    with pytest.raises(ValidationError, match="needs at least one source"):
        AskRequest.from_question(QUESTION, mode="openbook")
    with pytest.raises(ValidationError, match="needs at least one source"):
        AskRequest.from_question(QUESTION, mode="stuffed", sources=[])


def test_sources_are_refused_outside_the_source_modes():
    with pytest.raises(ValidationError, match="sources need mode"):
        AskRequest.from_question(QUESTION, mode="route", sources=["a.md"])
    with pytest.raises(ValidationError, match="sources need mode"):
        AskRequest.from_question(QUESTION, mode="chain", sources=["a.md"])


def test_openbook_only_simulations_need_openbook_mode():
    with pytest.raises(ValidationError, match="needs mode 'openbook'"):
        AskRequest.from_question(QUESTION, simulate="quote-mismatch")
    with pytest.raises(ValidationError, match="needs mode 'openbook'"):
        AskRequest.from_question(QUESTION, mode="stuffed", sources=["a.md"], simulate="grounding-error")
    with pytest.raises(ValidationError, match="needs mode 'chain'"):
        AskRequest.from_question(QUESTION, mode="openbook", sources=["a.md"], simulate="draft-invalid")


def test_from_question_accepts_the_contract_arguments():
    request = AskRequest.from_question(QUESTION, tier="mid", mode="stuffed", sources=["a.md", "https://x.test/p"])

    assert request.mode == "stuffed"
    assert request.sources == ["a.md", "https://x.test/p"]
    assert AskRequest.from_question(QUESTION).sources == []


def test_source_modes_need_an_openbook_block(kit, source_paths):
    with pytest.raises(OpenbookNotConfiguredError):
        kit.harness.answer(ask(source_paths))
    with pytest.raises(OpenbookNotConfiguredError):
        kit.harness.answer(AskRequest.from_question(QUESTION, mode="stuffed", sources=source_paths))


def test_unknown_grounding_verifier_tier_is_refused(ob_kit, source_paths):
    with pytest.raises(UnknownTierError, match="unknown verifier tier"):
        ob_kit.harness.answer(ask(source_paths, verifier_tier="mid"))


# --- grounded ---


def test_grounded_answer_is_returned_with_citations(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("yes", "supported", "supported"))

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.status == "ok"
    assert result.mode == "openbook"
    assert result.answer == GOOD_ANSWER
    assert result.grounding_status == "grounded"
    assert result.abstained is False
    assert result.repaired is False
    assert result.passages_retrieved == 3
    assert result.sources_count == 2
    assert result.retrieval_method == "lexical"
    assert (result.claims_total, result.claims_supported) == (2, 2)
    assert result.grounding_verifier_tier == "ground_chat"
    assert [citation.model_dump() for citation in result.citations] == [
        {"source": source_paths[0], "passage_id": "S1-P1", "quote": THRESHOLD_CLAIM["quote"]},
        {"source": source_paths[1], "passage_id": "S2-P1", "quote": CODE_CLAIM["quote"]},
    ]
    assert result.triage_status == "skipped"
    assert result.route_reason == "configured"
    assert (result.selected_tier, result.final_tier, result.final_model) == ("mid", "mid", "mid-model")
    assert result.verification_status == "skipped"
    assert ob_kit.client.models_called() == ["mid-model", "ground-model"]
    assert [(call.seq, call.role, call.tier) for call in result.calls] == [
        (1, "draft", "mid"),
        (2, "ground", "ground_chat"),
    ]


def test_draft_prompt_holds_the_numbered_passages_the_question_and_the_schema(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("yes", "supported", "supported"))

    ob_kit.harness.answer(ask(source_paths))

    draft_call = ob_kit.client.calls[0]
    user_text = draft_call.messages[1]["content"]
    assert draft_call.messages[0]["content"] == openbook.DRAFT_SYSTEM_PROMPT
    assert f"[S1-P1] source: {source_paths[0]} | section: Router Guide > Thresholds" in user_text
    assert "[S1-P2]" in user_text and "[S2-P1]" in user_text
    assert user_text.index("[S1-P1]") < user_text.index("[S1-P2]") < user_text.index("[S2-P1]")
    assert user_text.endswith(f"QUESTION:\n{QUESTION}")
    assert draft_call.json_schema["required"][0] == "working"
    assert set(draft_call.json_schema["required"]) == {"working", "answerable", "answer", "claims"}


def test_forced_tier_drafts_instead_of_the_configured_drafter(ob_kit, source_paths):
    ob_kit.client.script("local-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("yes", "supported", "supported"))

    result = ob_kit.harness.answer(ask(source_paths, tier="local"))

    assert result.route_reason == "forced"
    assert result.final_tier == "local"
    assert "triage-model" not in ob_kit.client.models_called()


def test_top_k_limits_the_passages_given_to_the_drafter(tmp_path, source_paths):
    kit = HarnessKit(make_openbook_config(tmp_path / "cache", {"top_k": 1}))
    kit.client.script("mid-model", openbook_draft([CODE_CLAIM]))
    kit.client.script("ground-model", grounding_report("yes", "supported"))

    result = kit.harness.answer(
        AskRequest.from_question("Which model is the code specialist?", mode="openbook", sources=source_paths)
    )

    assert result.passages_retrieved == 1
    user_text = kit.client.calls[0].messages[1]["content"]
    assert "[S2-P1]" in user_text
    assert "[S1-P1]" not in user_text


def test_small_corpus_within_the_budget_skips_retrieval(tmp_path, source_paths):
    kit = HarnessKit(make_openbook_config(tmp_path / "cache", {"full_context_token_budget": 5000, "top_k": 1}))
    kit.client.script("mid-model", openbook_draft())
    kit.client.script("ground-model", grounding_report("yes", "supported", "supported"))

    result = kit.harness.answer(ask(source_paths))

    assert result.retrieval_method == "all"
    assert result.passages_retrieved == 3


# --- the deterministic quote check ---


def test_fabricated_quote_is_caught_in_code_and_never_reaches_the_verifier(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft([THRESHOLD_CLAIM, FABRICATED_CLAIM]))
    ob_kit.client.script("ground-model", grounding_report("yes", "supported"))
    ob_kit.client.script("frontier-model", openbook_draft([THRESHOLD_CLAIM], answer="Below 0.80."))

    result = ob_kit.harness.answer(ask(source_paths))

    verifier_text = ob_kit.client.calls[1].messages[1]["content"]
    assert "1. (cites S1-P1) Escalation happens when confidence is below 0.80." in verifier_text
    assert "three retries" not in verifier_text
    assert result.grounding_status == "partly_grounded"
    assert "claim 2 [S1-P1]: quote check failed (quote_not_in_passage)" in result.grounding_notes
    assert result.repaired is True
    assert result.answer == "Below 0.80."


def test_draft_whose_every_quote_is_fabricated_skips_the_verifier_and_escalates(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft([FABRICATED_CLAIM]))
    ob_kit.client.script("frontier-model", openbook_draft())

    result = ob_kit.harness.answer(ask(source_paths))

    assert ob_kit.client.models_called() == ["mid-model", "frontier-model"]
    assert result.grounding_status == "ungrounded"
    assert result.grounding_verifier_tier is None
    assert openbook.NO_QUOTE_PASSED_NOTE in result.grounding_notes
    assert result.repaired is True
    assert result.answer == GOOD_ANSWER
    assert len(result.citations) == 2
    assert call_trail(ob_kit) == [("draft", "mid", "ok"), ("escalate", "frontier", "ok")]


def test_unknown_passage_id_fails_the_quote_check(ob_kit, source_paths):
    wrong_id_claim = dict(THRESHOLD_CLAIM, passage_id="S7-P1")
    ob_kit.client.script("mid-model", openbook_draft([wrong_id_claim]))
    ob_kit.client.script("frontier-model", error_result("http_5xx"))

    result = ob_kit.harness.answer(ask(source_paths))

    assert "claim 1 [S7-P1]: quote check failed (unknown_passage)" in result.grounding_notes
    assert result.citations == []


# --- partly grounded, escalation ---


def test_partly_grounded_draft_is_escalated_once_with_the_same_input(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft(answer="a draft the verifier rejects in part"))
    ob_kit.client.script("ground-model", grounding_report("yes", "supported", "unsupported"))
    ob_kit.client.script("frontier-model", openbook_draft([CODE_CLAIM], answer="The escalated answer."))

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.status == "ok"
    assert result.grounding_status == "partly_grounded"
    assert result.repaired is True
    assert result.abstained is False
    assert result.answer == "The escalated answer."
    assert (result.final_tier, result.final_model) == ("frontier", "frontier-model")
    assert result.selected_tier == "mid"
    assert (result.claims_total, result.claims_supported) == (1, 1)
    assert [citation.passage_id for citation in result.citations] == ["S2-P1"]
    assert "claim 2 [S2-P1]: unsupported (reason 2)" in result.grounding_notes
    assert call_trail(ob_kit) == [
        ("draft", "mid", "ok"),
        ("ground", "ground_chat", "ok"),
        ("escalate", "frontier", "ok"),
    ]

    draft_call, _, escalation_call = ob_kit.client.calls
    assert escalation_call.messages == draft_call.messages
    assert "a draft the verifier rejects in part" not in json.dumps(escalation_call.messages)
    assert escalation_call.json_schema == draft_call.json_schema
    # No second verifier call: the verifier model was called exactly once.
    assert ob_kit.client.models_called().count("ground-model") == 1


def test_uncertain_claim_counts_as_not_supported(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("yes", "supported", "uncertain"))
    ob_kit.client.script("frontier-model", openbook_draft())

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.grounding_status == "partly_grounded"
    assert result.repaired is True


def test_escalated_answer_passes_the_quote_check_again(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("yes", "supported", "unsupported"))
    ob_kit.client.script("frontier-model", openbook_draft([CODE_CLAIM, FABRICATED_CLAIM], answer="Escalated."))

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.answer == "Escalated."
    assert (result.claims_total, result.claims_supported) == (2, 1)
    assert [citation.passage_id for citation in result.citations] == ["S2-P1"]
    assert "escalated claim 2 [S1-P1]: quote check failed (quote_not_in_passage)" in result.grounding_notes


def test_failed_escalation_returns_the_first_draft_with_its_status(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("yes", "supported", "unsupported"))
    ob_kit.client.script("frontier-model", error_result("http_5xx"))

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.status == "ok"
    assert result.answer == GOOD_ANSWER
    assert result.grounding_status == "partly_grounded"
    assert result.repaired is False
    assert result.repair_error_type == "http_5xx"
    assert result.final_tier == "mid"
    assert [citation.passage_id for citation in result.citations] == ["S1-P1"]
    assert (result.claims_total, result.claims_supported) == (2, 1)
    # The escalation never takes the on_error hop.
    assert ob_kit.client.models_called() == ["mid-model", "ground-model", "frontier-model"]


def test_invalid_escalation_output_is_a_failed_escalation(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("yes", "supported", "unsupported"))
    ob_kit.client.script("frontier-model", ok_result("not json"))

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.repaired is False
    assert result.repair_error_type == "schema_invalid"
    assert call_trail(ob_kit)[-1] == ("escalate", "frontier", "invalid")


# --- ungrounded, abstention ---


def test_ungrounded_draft_then_escalation_says_unanswerable_abstains(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("yes", "unsupported", "unsupported"))
    ob_kit.client.script("frontier-model", unanswerable_draft())

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.status == "ok"
    assert result.grounding_status == "ungrounded"
    assert result.abstained is True
    assert result.repaired is True
    assert result.answer == openbook.ABSTAIN_ANSWER
    assert result.citations == []
    assert result.claims_supported == 0
    assert result.final_tier == "frontier"


def test_escalated_answer_with_no_claim_passing_the_quote_check_abstains(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("yes", "unsupported", "unsupported"))
    ob_kit.client.script("frontier-model", openbook_draft([FABRICATED_CLAIM], answer="A confident invention."))

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.abstained is True
    assert result.answer == openbook.ABSTAIN_ANSWER
    assert (result.claims_total, result.claims_supported) == (1, 0)


def test_failed_answerability_judgment_is_ungrounded_even_with_supported_claims(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("no", "supported", "supported"))
    ob_kit.client.script("frontier-model", openbook_draft())

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.grounding_status == "ungrounded"
    assert result.repaired is True
    assert "passages answer the question: no" in result.grounding_notes


def test_drafter_says_unanswerable_and_verifier_agrees_abstains_without_escalating(ob_kit, source_paths):
    ob_kit.client.script("mid-model", unanswerable_draft())
    ob_kit.client.script("ground-model", grounding_report("no"))

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.status == "ok"
    assert result.abstained is True
    assert result.answer == openbook.ABSTAIN_ANSWER
    assert result.grounding_status == "ungrounded"
    assert result.repaired is False
    assert result.citations == []
    assert result.final_tier == "mid"
    assert ob_kit.client.models_called() == ["mid-model", "ground-model"]
    assert "(no claims" in ob_kit.client.calls[1].messages[1]["content"]


def test_abstention_the_verifier_is_unsure_about_stands_without_an_escalation(ob_kit, source_paths):
    ob_kit.client.script("mid-model", unanswerable_draft())
    ob_kit.client.script("ground-model", grounding_report("uncertain"))

    result = ob_kit.harness.answer(ask(source_paths))

    assert ob_kit.client.models_called() == ["mid-model", "ground-model"]
    assert result.grounding_status == "ungrounded"
    assert result.repaired is False
    assert result.abstained is True
    assert result.answer == openbook.ABSTAIN_ANSWER


def test_drafter_says_unanswerable_but_verifier_says_answerable_escalates(ob_kit, source_paths):
    ob_kit.client.script("mid-model", unanswerable_draft())
    ob_kit.client.script("ground-model", grounding_report("yes"))
    ob_kit.client.script("frontier-model", openbook_draft())

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.grounding_status == "ungrounded"
    assert result.repaired is True
    assert result.abstained is False
    assert result.answer == GOOD_ANSWER
    assert len(result.citations) == 2


def test_disputed_unanswerable_draft_abstains_when_the_escalation_fails(ob_kit, source_paths):
    ob_kit.client.script("mid-model", unanswerable_draft())
    ob_kit.client.script("ground-model", grounding_report("yes"))
    ob_kit.client.script("frontier-model", error_result("missing_api_key"))

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.abstained is True
    assert result.answer == openbook.ABSTAIN_ANSWER
    assert result.repaired is False
    assert result.repair_error_type == "missing_api_key"


# --- verifier failure ---


def test_verifier_error_returns_the_draft_as_unavailable_by_default(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft())
    ob_kit.client.script("ground-model", error_result("timeout"))

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.status == "ok"
    assert result.answer == GOOD_ANSWER
    assert result.grounding_status == "unavailable"
    assert result.repaired is False
    assert result.abstained is False
    # The citations are the claims that passed the quote check; no verifier saw them.
    assert len(result.citations) == 2
    assert call_trail(ob_kit) == [("draft", "mid", "ok"), ("ground", "ground_chat", "timeout")]
    assert "frontier-model" not in ob_kit.client.models_called()


def test_invalid_verifier_report_is_a_verifier_error_not_a_verdict(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("yes", "supported"))

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.grounding_status == "unavailable"
    assert call_trail(ob_kit)[-1] == ("ground", "ground_chat", "invalid")
    assert ob_kit.call_records[-1].error_type == "grounding_invalid"


def test_verifier_error_with_an_unanswerable_draft_abstains_unverified(ob_kit, source_paths):
    ob_kit.client.script("mid-model", unanswerable_draft())
    ob_kit.client.script("ground-model", error_result("http_5xx"))

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.grounding_status == "unavailable"
    assert result.abstained is True
    assert result.answer == openbook.ABSTAIN_ANSWER


def test_verifier_error_escalates_when_configured(tmp_path, source_paths):
    kit = HarnessKit(make_openbook_config(tmp_path / "cache", {"on_verifier_error": "escalate"}))
    kit.client.script("mid-model", openbook_draft())
    kit.client.script("ground-model", error_result("http_5xx"))
    kit.client.script("frontier-model", openbook_draft([CODE_CLAIM], answer="Escalated."))

    result = kit.harness.answer(ask(source_paths))

    assert result.grounding_status == "unavailable"
    assert result.repaired is True
    assert result.answer == "Escalated."


def test_verifier_error_never_escalates_an_abstention(tmp_path, source_paths):
    kit = HarnessKit(make_openbook_config(tmp_path / "cache", {"on_verifier_error": "escalate"}))
    kit.client.script("mid-model", unanswerable_draft())
    kit.client.script("ground-model", error_result("http_5xx"))

    result = kit.harness.answer(ask(source_paths))

    assert kit.client.models_called() == ["mid-model", "ground-model"]
    assert result.grounding_status == "unavailable"
    assert result.abstained is True
    assert result.repaired is False


def test_truncated_escalation_reply_records_why_it_stopped(ob_kit, source_paths):
    truncated = ok_result('{"working":"S1-P1 holds the thresh').model_copy(
        update={"finish_reason": "content_filter", "native_finish_reason": "refusal", "reasoning_tokens": 0}
    )
    ob_kit.client.script("mid-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("yes", "supported", "unsupported"))
    ob_kit.client.script("frontier-model", truncated)

    result = ob_kit.harness.answer(ask(source_paths))

    escalate_record = ob_kit.call_records[-1]
    assert (escalate_record.role, escalate_record.status) == ("escalate", "invalid")
    assert escalate_record.finish_reason == "content_filter"
    assert escalate_record.native_finish_reason == "refusal"
    assert escalate_record.reasoning_tokens == 0
    escalate_step_message = [call.error_message for call in result.calls if call.role == "escalate"]
    assert "finish_reason=content_filter" in escalate_step_message[0]
    assert result.answer == GOOD_ANSWER


def test_a_model_never_judges_its_own_draft(tmp_path, source_paths):
    config = make_openbook_config(tmp_path / "cache")
    config.tiers["ground_chat"].model = "mid-model"
    kit = HarnessKit(config)
    kit.client.script("mid-model", openbook_draft())

    result = kit.harness.answer(ask(source_paths))

    assert kit.client.models_called() == ["mid-model"]
    assert result.grounding_status == "unavailable"
    assert result.grounding_verifier_tier is None
    assert openbook.SAME_MODEL_NOTE in result.grounding_notes
    assert result.answer == GOOD_ANSWER


def test_alt_verifier_is_used_when_the_drafter_shares_the_verifier_model(tmp_path, source_paths):
    raw_tiers = {"ground_alt": {"model": "alt-model", "provider": "ollama", "role": "verifier", "timeout_s": 30}}
    config = make_openbook_config(
        tmp_path / "cache",
        {"alt_grounding_verifier_tier": "ground_alt"},
        tiers={**make_openbook_config(tmp_path / "cache").model_dump()["tiers"], **raw_tiers},
    )
    config.tiers["ground_chat"].model = "mid-model"
    kit = HarnessKit(config)
    kit.client.script("mid-model", openbook_draft())
    kit.client.script("alt-model", grounding_report("yes", "supported", "supported"))

    result = kit.harness.answer(ask(source_paths))

    assert result.grounding_status == "grounded"
    assert result.grounding_verifier_tier == "ground_alt"
    assert kit.client.models_called() == ["mid-model", "alt-model"]


# --- schema gate and call failure ---


def test_invalid_draft_gets_one_retry_on_the_escalation_tier(ob_kit, source_paths):
    ob_kit.client.script("mid-model", ok_result('{"answer": "no claims field"}'))
    ob_kit.client.script("frontier-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("yes", "supported", "supported"))

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.status == "ok"
    assert result.schema_escalated is True
    assert result.grounding_status == "grounded"
    assert result.final_tier == "frontier"
    assert result.repaired is False
    assert call_trail(ob_kit) == [
        ("draft", "mid", "invalid"),
        ("draft", "frontier", "ok"),
        ("ground", "ground_chat", "ok"),
    ]


def test_draft_already_from_the_escalation_tier_is_not_escalated_again(ob_kit, source_paths):
    ob_kit.client.script("mid-model", ok_result("not json"))
    ob_kit.client.script("frontier-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("yes", "supported", "unsupported"))

    result = ob_kit.harness.answer(ask(source_paths))

    assert ob_kit.client.models_called() == ["mid-model", "frontier-model", "ground-model"]
    assert result.grounding_status == "partly_grounded"
    assert result.repaired is False
    assert [citation.passage_id for citation in result.citations] == ["S1-P1"]
    assert openbook.ALREADY_ON_ESCALATION_TIER_NOTE in result.grounding_notes


def test_two_invalid_drafts_fail_the_run(ob_kit, source_paths):
    ob_kit.client.script("mid-model", ok_result("not json"))
    ob_kit.client.script("frontier-model", ok_result("still not json"))

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.status == "error"
    assert result.answer is None
    assert result.error_type == "draft_schema_invalid"
    assert result.grounding_status == "skipped"


def test_answerable_draft_with_an_empty_answer_is_invalid(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft(answer="  "))
    ob_kit.client.script("frontier-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("yes", "supported", "supported"))

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.schema_escalated is True
    assert result.answer == GOOD_ANSWER


def test_failed_draft_call_takes_the_single_on_error_hop(ob_kit, source_paths):
    ob_kit.client.script("mid-model", error_result("http_5xx"))
    ob_kit.client.script("local-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("yes", "supported", "supported"))

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.status == "ok"
    assert result.fallback_used is True
    assert result.final_tier == "local"
    assert call_trail(ob_kit)[:2] == [("draft", "mid", "error"), ("draft", "local", "ok")]


# --- simulations ---


def test_simulated_quote_mismatch_fails_every_quote_of_the_first_draft(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft())
    ob_kit.client.script("frontier-model", openbook_draft())

    result = ob_kit.harness.answer(ask(source_paths, simulate="quote-mismatch"))

    assert result.grounding_status == "ungrounded"
    assert "claim 1 [S1-P1]: quote check failed (simulated_mismatch)" in result.grounding_notes
    assert result.repaired is True
    assert len(result.citations) == 2
    assert "ground-model" not in ob_kit.client.models_called()


def test_simulated_grounding_error_calls_no_verifier(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft())

    result = ob_kit.harness.answer(ask(source_paths, simulate="grounding-error"))

    assert result.grounding_status == "unavailable"
    assert ob_kit.client.models_called() == ["mid-model"]
    assert ob_kit.call_records[-1].role == "ground"
    assert ob_kit.call_records[-1].error_type == "simulated_error"


def test_simulated_answer_error_in_openbook_mode_takes_the_on_error_hop(ob_kit, source_paths):
    ob_kit.client.script("local-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("yes", "supported", "supported"))

    result = ob_kit.harness.answer(ask(source_paths, simulate="answer-error"))

    assert result.fallback_used is True
    assert ob_kit.call_records[0].error_type == "simulated_error"
    assert "mid-model" not in ob_kit.client.models_called()


# --- source loading through the harness ---


def test_source_that_cannot_be_loaded_fails_the_run_before_any_model_call(ob_kit, source_paths, tmp_path):
    result = ob_kit.harness.answer(ask([source_paths[0], str(tmp_path / "absent.md")]))

    assert result.status == "error"
    assert result.answer is None
    assert result.error_type == "source_load_failed"
    assert "absent.md" in result.error_message
    assert ob_kit.client.calls == []
    assert result.calls == []
    assert ob_kit.run_record.status == "error"
    assert ob_kit.run_record.sources_count == 2


def test_url_source_is_fetched_extracted_and_cited(tmp_path, source_paths):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html><body><nav>menu</nav><h1>Limits</h1>"
            b"<p>The daily request limit is 500 calls.</p></body></html>",
        )

    config = make_openbook_config(tmp_path / "cache")
    client = FakeClient()
    telemetry = FakeTelemetry()
    harness = Harness(
        config,
        {"ollama": client, "openrouter": client},
        telemetry,
        source_transport=httpx.MockTransport(handler),
    )
    url = "https://docs.example.test/limits"
    limit_claim = {
        "text": "The daily limit is 500 calls.",
        "passage_id": "S1-P1",
        "quote": "The daily request limit is 500 calls.",
    }
    client.script("mid-model", openbook_draft([limit_claim], answer="500 calls per day."))
    client.script("ground-model", grounding_report("yes", "supported"))

    result = harness.answer(AskRequest.from_question("What is the daily limit?", mode="openbook", sources=[url]))

    assert result.grounding_status == "grounded"
    assert result.citations[0].source == url
    assert "menu" not in client.calls[0].messages[1]["content"]


# --- embeddings through the harness ---


def embedding_kit(tmp_path) -> HarnessKit:
    return HarnessKit(make_openbook_config(tmp_path / "cache", {"embedding_tier": "embed"}))


def test_embedding_request_is_recorded_as_an_embed_call(tmp_path, source_paths):
    kit = embedding_kit(tmp_path)
    vectors = [[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.8, 0.2]]
    kit.client.script_embed(EmbedResult(CallResult(status="ok", latency_ms=4.0, prompt_tokens=30), vectors))
    kit.client.script("mid-model", openbook_draft())
    kit.client.script("ground-model", grounding_report("yes", "supported", "supported"))

    result = kit.harness.answer(ask(source_paths))

    assert result.retrieval_method == "hybrid"
    assert result.retrieval_note is None
    assert call_trail(kit)[0] == ("embed", "embed", "ok")
    assert kit.call_records[0].model == "embed-model"
    assert kit.client.embed_calls[0].model == "embed-model"
    assert len(kit.client.embed_calls[0].inputs) == 4


def test_failed_embedding_falls_back_to_lexical_and_the_run_continues(tmp_path, source_paths):
    kit = embedding_kit(tmp_path)
    kit.client.script_embed(EmbedResult(error_result("connection")))
    kit.client.script("mid-model", openbook_draft())
    kit.client.script("ground-model", grounding_report("yes", "supported", "supported"))

    result = kit.harness.answer(ask(source_paths))

    assert result.status == "ok"
    assert result.retrieval_method == "lexical"
    assert result.retrieval_note == "embedding call failed (connection)"
    assert call_trail(kit)[0] == ("embed", "embed", "error")
    assert result.grounding_status == "grounded"


# --- Jev as the grounding verifier ---


def jev_kit(tmp_path, monkeypatch, answers: dict | None, status_code: int = 200):
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "test-key-value")
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(status_code, json={"answers": answers} if answers is not None else {})

    raw_config = make_openbook_config(tmp_path / "cache").model_dump(exclude={"sha256", "source_path"})
    raw_config["providers"]["decisions"] = {
        "kind": "openrouter_decisions",
        "base_url": "http://decisions.invalid",
        "api_key_env": "TEST_OPENROUTER_KEY",
    }
    raw_config["tiers"]["ground_jev"] = {
        "model": "jev-model",
        "provider": "decisions",
        "role": "verifier",
        "cloud": True,
        "timeout_s": 5,
    }
    raw_config["openbook"]["grounding_verifier_tier"] = "ground_jev"
    config = HarnessConfig.model_validate(raw_config)
    client = FakeClient()
    decisions_client = OpenRouterDecisionsClient(
        "http://decisions.invalid", "TEST_OPENROUTER_KEY", transport=httpx.MockTransport(handler)
    )
    harness = Harness(
        config, {"ollama": client, "openrouter": client, "decisions": decisions_client}, FakeTelemetry()
    )
    return harness, client, requests


def test_jev_grounds_a_draft_with_one_bundled_request(tmp_path, monkeypatch, source_paths):
    harness, client, requests = jev_kit(
        tmp_path,
        monkeypatch,
        {"passages_answer_question": {"noul": 0.93}, "claim_1": {"noul": 0.91}, "claim_2": {"noul": 0.84}},
    )
    client.script("mid-model", openbook_draft())

    result = harness.answer(ask(source_paths))

    assert len(requests) == 1
    assert set(requests[0]["questions"]) == {"passages_answer_question", "claim_1", "claim_2"}
    assert requests[0]["state"]["user_question"] == QUESTION
    assert result.grounding_status == "grounded"
    assert result.grounding_verifier_tier == "ground_jev"
    assert "passages answer the question: yes (p=0.93)" in result.grounding_notes


def test_jev_middle_band_is_uncertain_and_escalates(tmp_path, monkeypatch, source_paths):
    harness, client, _ = jev_kit(
        tmp_path,
        monkeypatch,
        {"passages_answer_question": {"noul": 0.9}, "claim_1": {"noul": 0.9}, "claim_2": {"noul": 0.5}},
    )
    client.script("mid-model", openbook_draft())
    client.script("frontier-model", openbook_draft())

    result = harness.answer(ask(source_paths))

    assert result.grounding_status == "partly_grounded"
    assert "claim 2 [S2-P1]: uncertain (p=0.50)" in result.grounding_notes
    assert result.repaired is True


def test_jev_missing_judgment_fails_closed(tmp_path, monkeypatch, source_paths):
    harness, client, _ = jev_kit(
        tmp_path, monkeypatch, {"passages_answer_question": {"noul": 0.9}, "claim_1": {"noul": 0.9}}
    )
    client.script("mid-model", openbook_draft())

    result = harness.answer(ask(source_paths))

    assert result.grounding_status == "unavailable"
    assert result.calls[-1].status == "invalid"
    assert result.calls[-1].error_type == "grounding_invalid"


def test_jev_http_error_fails_closed(tmp_path, monkeypatch, source_paths):
    harness, client, _ = jev_kit(tmp_path, monkeypatch, None, status_code=503)
    client.script("mid-model", openbook_draft())

    result = harness.answer(ask(source_paths))

    assert result.grounding_status == "unavailable"
    assert result.calls[-1].error_type == "http_5xx"
    assert result.answer == GOOD_ANSWER


# --- stuffed baseline ---


def stuffed(source_paths: list[str], **overrides) -> AskRequest:
    return AskRequest.from_question(QUESTION, mode="stuffed", sources=source_paths, **overrides)


def test_stuffed_mode_is_one_call_with_every_source_before_the_question(ob_kit, source_paths):
    ob_kit.client.script("mid-model", ok_result("A plain stuffed answer."))

    result = ob_kit.harness.answer(stuffed(source_paths, tier="mid"))

    assert result.status == "ok"
    assert result.mode == "stuffed"
    assert result.answer == "A plain stuffed answer."
    assert ob_kit.client.models_called() == ["mid-model"]
    assert [(call.role, call.tier) for call in result.calls] == [("answer", "mid")]
    assert result.citations == []
    assert result.grounding_status == "skipped"
    assert result.abstained is False
    assert result.passages_retrieved == 0
    assert result.repaired is False
    assert result.sources_count == 2
    assert result.retrieval_method == "none"

    [call] = ob_kit.client.calls
    user_text = call.messages[-1]["content"]
    assert call.json_schema is None
    assert GUIDE_TEXT.strip() in user_text
    assert MODELS_TEXT.strip() in user_text
    assert user_text.index(MODELS_TEXT.strip()) < user_text.index(QUESTION)
    assert user_text.endswith(f"QUESTION:\n{QUESTION}")


def test_stuffed_mode_without_a_forced_tier_uses_triage(ob_kit, source_paths):
    ob_kit.client.script("triage-model", triage_json("local", 0.95))
    ob_kit.client.script("local-model", ok_result("local stuffed answer"))

    result = ob_kit.harness.answer(stuffed(source_paths))

    assert result.route_reason == "triage_label"
    assert result.final_tier == "local"
    assert [(call.role, call.tier) for call in result.calls] == [("triage", "triage"), ("answer", "local")]


def test_stuffed_mode_failure_is_an_error_without_a_fallback_hop(ob_kit, source_paths):
    ob_kit.client.script("mid-model", error_result("http_5xx"))

    result = ob_kit.harness.answer(stuffed(source_paths, tier="mid"))

    assert result.status == "error"
    assert result.answer is None
    assert result.error_type == "http_5xx"
    assert result.fallback_used is False
    assert ob_kit.client.models_called() == ["mid-model"]


def test_stuffed_mode_refuses_a_prompt_above_the_known_context_window(ob_kit, source_paths):
    ob_kit.config.tiers["mid"].context_window_tokens = 10

    result = ob_kit.harness.answer(stuffed(source_paths, tier="mid"))

    assert result.status == "error"
    assert result.error_type == "context_overflow"
    assert "10-token context window" in result.error_message
    assert ob_kit.client.models_called() == []


def test_stuffed_mode_calls_the_model_when_the_prompt_fits_the_context_window(ob_kit, source_paths):
    ob_kit.config.tiers["mid"].context_window_tokens = 1_000_000
    ob_kit.client.script("mid-model", ok_result("fits"))

    result = ob_kit.harness.answer(stuffed(source_paths, tier="mid"))

    assert result.status == "ok"
    assert result.answer == "fits"


def test_stuffed_mode_fails_closed_on_a_bad_source(ob_kit, tmp_path):
    result = ob_kit.harness.answer(stuffed([str(tmp_path / "absent.md")], tier="mid"))

    assert result.error_type == "source_load_failed"
    assert ob_kit.client.calls == []


# --- other modes are unchanged ---


def test_route_mode_is_unchanged_and_reports_empty_openbook_fields(kit):
    kit.client.script("triage-model", triage_json("local", 0.95))
    kit.client.script("local-model", ok_result("Paris"))

    result = kit.harness.answer(AskRequest.from_question("What is the capital of France?", mode="route"))

    assert result.mode == "route"
    assert result.answer == "Paris"
    assert [(call.role, call.tier) for call in result.calls] == [("triage", "triage"), ("answer", "local")]
    assert result.citations == []
    assert result.grounding_status == "skipped"
    assert result.abstained is False
    assert result.passages_retrieved == 0
    assert result.repaired is False
    assert kit.run_record.mode == "route"
    assert kit.run_record.grounding_status == "skipped"
    assert kit.run_record.sources_count == 0


def test_chain_mode_is_unchanged_and_reports_empty_openbook_fields():
    chain_kit = HarnessKit(make_chain_config())
    chain_kit.client.script("local-model", draft_json("24 legs"))
    chain_kit.client.script("verify-model", verdict_json("supported"))

    result = chain_kit.harness.answer(
        AskRequest.from_question("How many legs do three spiders have?", mode="chain", tier="local")
    )

    assert result.mode == "chain"
    assert result.answer == "24 legs"
    assert result.verification_status == "supported"
    assert [(call.role, call.tier) for call in result.calls] == [("draft", "local"), ("verify", "verify")]
    assert result.citations == []
    assert result.grounding_status == "skipped"
    assert result.abstained is False
    assert result.passages_retrieved == 0


def test_route_mode_works_on_a_config_with_an_openbook_block(ob_kit):
    ob_kit.client.script("triage-model", triage_json("mid", 0.9))
    ob_kit.client.script("mid-model", ok_result("plain answer"))

    result = ob_kit.harness.answer(AskRequest.from_question("A closed-book question"))

    assert result.mode == "route"
    assert result.answer == "plain answer"


# --- config ---


def test_shipped_config_has_an_openbook_block():
    config = load_config(REAL_CONFIG_PATH)

    assert config.openbook is not None
    assert config.openbook.drafter_tier == "mid"
    assert config.openbook.escalation_tier == "frontier"
    assert config.openbook.grounding_verifier_tier == "ground_jev"
    assert config.openbook.on_verifier_error == "return_unverified"
    assert config.openbook.top_k == 6
    assert config.providers[config.tiers["ground_jev"].provider].kind == "openrouter_decisions"
    assert config.tiers["ground_chat"].model != config.tiers["mid"].model
    assert config.tiers[config.openbook.embedding_tier].role == "embedding"
    assert config.tiers[config.openbook.embedding_tier].model == "nomic-embed-text:latest"
    for tier_name in ("ground_jev", "ground_chat", "embed"):
        assert tier_name not in config.answer_tier_names()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"drafter_tier": "ground_chat"}, "drafter_tier"),
        ({"escalation_tier": "triage"}, "escalation_tier"),
        ({"grounding_verifier_tier": "mid"}, "grounding_verifier_tier"),
        ({"alt_grounding_verifier_tier": "mid"}, "alt_grounding_verifier_tier"),
        ({"alt_grounding_verifier_tier": "ground_chat"}, "different model"),
        ({"embedding_tier": "mid"}, "embedding_tier"),
        ({"top_k": 0}, "top_k"),
    ],
)
def test_openbook_config_references_are_validated(tmp_path, overrides, message):
    with pytest.raises(ValidationError, match=message):
        make_openbook_config(tmp_path / "cache", overrides)


def test_base_config_without_an_openbook_block_is_still_valid():
    assert HarnessConfig.model_validate(BASE_CONFIG).openbook is None


# --- telemetry through the harness ---


def test_openbook_run_is_written_to_a_real_database(tmp_path, source_paths):
    db_path = tmp_path / "telemetry.db"
    config = make_openbook_config(tmp_path / "cache")
    client = FakeClient()
    harness = Harness(config, {"ollama": client, "openrouter": client}, Telemetry(db_path))
    client.script("mid-model", openbook_draft())
    client.script("ground-model", grounding_report("yes", "supported", "unsupported"))
    client.script("frontier-model", unanswerable_draft())

    result = harness.answer(ask(source_paths))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        run_row = dict(conn.execute("SELECT * FROM runs WHERE trace_id = ?", (result.trace_id,)).fetchone())
        roles = [row["role"] for row in conn.execute("SELECT role FROM calls ORDER BY seq")]
    finally:
        conn.close()

    assert run_row["mode"] == "openbook"
    assert run_row["sources_count"] == 2
    assert run_row["passages_retrieved"] == 3
    assert run_row["retrieval_method"] == "lexical"
    assert run_row["grounding_status"] == "partly_grounded"
    assert run_row["abstained"] == 1
    assert run_row["repaired"] == 1
    assert run_row["claims_total"] == 0
    assert run_row["claims_supported"] == 0
    assert run_row["grounding_verifier_tier"] == "ground_chat"
    assert run_row["route_reason"] == "configured"
    assert roles == ["draft", "ground", "escalate"]


# --- CLI ---


def run_cli(monkeypatch, kit: HarnessKit, arguments: list[str]) -> int:
    monkeypatch.setattr(cli, "build_harness", lambda *args, **kwargs: kit.harness)
    return cli.main(arguments)


def test_cli_prints_the_answer_then_numbered_citations_and_a_trace(monkeypatch, capsys, ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("yes", "supported", "supported"))

    exit_code = run_cli(monkeypatch, ob_kit, ["ask", QUESTION, "--sources", *source_paths])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.splitlines() == [
        GOOD_ANSWER,
        "",
        "Citations:",
        f'[1] {source_paths[0]} (S1-P1): "{THRESHOLD_CLAIM["quote"]}"',
        f'[2] {source_paths[1]} (S2-P1): "{CODE_CLAIM["quote"]}"',
    ]
    assert "retrieval     lexical, 3 passages (no embedding tier configured)" in captured.err
    assert "status grounded, claims 2/2 supported" in captured.err
    assert "draft" in captured.err and "ground" in captured.err


def test_cli_stuffed_flag_selects_the_baseline(monkeypatch, capsys, ob_kit, source_paths):
    ob_kit.client.script("mid-model", ok_result("stuffed answer"))

    exit_code = run_cli(
        monkeypatch, ob_kit, ["ask", QUESTION, "--sources", *source_paths, "--stuffed", "--tier", "mid"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "stuffed answer"
    assert ob_kit.run_record.mode == "stuffed"


def test_cli_passes_the_simulation_and_verifier_tier_through(monkeypatch, capsys, ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft())

    exit_code = run_cli(
        monkeypatch,
        ob_kit,
        ["ask", QUESTION, "--sources", *source_paths, "--simulate", "grounding-error", "--verifier-tier", "ground_chat"],
    )

    assert exit_code == 0
    assert "status unavailable" in capsys.readouterr().err


def test_cli_reports_a_failed_source_load_as_a_failed_run(monkeypatch, capsys, ob_kit, tmp_path):
    exit_code = run_cli(monkeypatch, ob_kit, ["ask", QUESTION, "--sources", str(tmp_path / "absent.md")])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "source_load_failed" in captured.err
    assert "no such file" in captured.err


def test_cli_usage_errors(monkeypatch, capsys, ob_kit, source_paths):
    assert run_cli(monkeypatch, ob_kit, ["ask", QUESTION, "--stuffed"]) == 2
    assert "needs at least one source" in capsys.readouterr().err

    assert run_cli(monkeypatch, ob_kit, ["ask", QUESTION, "--chain", "--sources", *source_paths]) == 2
    assert "cannot be combined" in capsys.readouterr().err

    assert run_cli(monkeypatch, ob_kit, ["ask", QUESTION, "--simulate", "quote-mismatch"]) == 2
    assert "needs mode 'openbook'" in capsys.readouterr().err


# --- escalation input: cited passages plus the best-ranked, under a token cap ---


def sized_passage(passage_number: int, tokens: int) -> Passage:
    return Passage(
        passage_id=f"S1-P{passage_number}",
        source_index=0,
        source="doc.md",
        position=passage_number - 1,
        heading="",
        text="x" * (tokens * 4),
    )


FIVE_PASSAGES = [sized_passage(number, 100) for number in range(1, 6)]


def selected_ids(cited: list[str], ranked: list[str], budget: int | None) -> list[str]:
    selected = openbook.select_escalation_passages(FIVE_PASSAGES, cited, ranked, budget)
    return [passage.passage_id for passage in selected]


def test_escalation_passages_are_cited_first_then_best_ranked_in_document_order():
    ranked = ["S1-P5", "S1-P4", "S1-P3", "S1-P2", "S1-P1"]

    assert selected_ids(["S1-P2"], ranked, 300) == ["S1-P2", "S1-P4", "S1-P5"]


def test_escalation_passages_deduplicate_and_ignore_unknown_ids():
    ranked = ["S1-P2", "S1-P1", "S1-P3", "S1-P4", "S1-P5"]

    assert selected_ids(["S1-P2", "S9-P9", "S1-P2"], ranked, 200) == ["S1-P1", "S1-P2"]


def test_escalation_passages_without_citations_are_the_best_ranked():
    ranked = ["S1-P3", "S1-P5", "S1-P1", "S1-P2", "S1-P4"]

    assert selected_ids([], ranked, 250) == ["S1-P3", "S1-P5"]


def test_escalation_always_gets_one_passage_even_over_the_cap():
    assert selected_ids([], ["S1-P4", "S1-P1"], 10) == ["S1-P4"]


@pytest.mark.parametrize("budget", [None, 0])
def test_no_escalation_budget_sends_every_passage(budget):
    assert selected_ids(["S1-P2"], ["S1-P5"], budget) == [f"S1-P{number}" for number in range(1, 6)]


def test_a_smaller_passage_further_down_the_ranking_still_fits():
    passages = [sized_passage(1, 100), sized_passage(2, 500), sized_passage(3, 100)]

    selected = openbook.select_escalation_passages(passages, [], ["S1-P1", "S1-P2", "S1-P3"], 250)

    assert [passage.passage_id for passage in selected] == ["S1-P1", "S1-P3"]


def test_escalation_is_sent_only_the_capped_passages_and_never_the_draft(tmp_path, source_paths):
    kit = HarnessKit(make_openbook_config(tmp_path / "cache", {"escalation_token_budget": 1}))
    kit.client.script("mid-model", openbook_draft([THRESHOLD_CLAIM, FABRICATED_CLAIM]))
    kit.client.script("ground-model", grounding_report("yes", "supported"))
    kit.client.script("frontier-model", openbook_draft([THRESHOLD_CLAIM], answer="Below 0.80."))

    result = kit.harness.answer(ask(source_paths))

    draft_prompt = kit.client.calls[0].messages[1]["content"]
    escalation_prompt = kit.client.calls[-1].messages[1]["content"]
    assert "[S1-P1]" in draft_prompt and "[S2-P1]" in draft_prompt
    assert "[S1-P1]" in escalation_prompt
    assert "[S2-P1]" not in escalation_prompt
    assert FABRICATED_CLAIM["text"] not in escalation_prompt
    assert result.repaired is True
    assert result.answer == "Below 0.80."
    assert result.escalation_passages == 1
    assert result.escalation_tokens_estimated > 0
    assert kit.run_record.escalation_passages == 1
    assert kit.run_record.escalation_tokens_estimated == result.escalation_tokens_estimated


def test_escalated_claim_citing_a_passage_it_was_not_sent_fails_the_quote_check(tmp_path, source_paths):
    kit = HarnessKit(make_openbook_config(tmp_path / "cache", {"escalation_token_budget": 1}))
    kit.client.script("mid-model", openbook_draft([THRESHOLD_CLAIM, FABRICATED_CLAIM]))
    kit.client.script("ground-model", grounding_report("yes", "supported"))
    kit.client.script("frontier-model", openbook_draft([CODE_CLAIM]))

    result = kit.harness.answer(ask(source_paths))

    assert result.abstained is True
    assert any("unknown_passage" in note for note in result.grounding_notes)


def test_escalated_abstention_gets_the_best_ranked_passages(tmp_path, source_paths):
    kit = HarnessKit(make_openbook_config(tmp_path / "cache", {"escalation_token_budget": 1}))
    kit.client.script("mid-model", unanswerable_draft())
    kit.client.script("ground-model", grounding_report("yes"))
    kit.client.script("frontier-model", unanswerable_draft())

    result = kit.harness.answer(ask(source_paths))

    assert result.escalation_passages == 1
    assert kit.client.calls[-1].messages[1]["content"].count("] source: ") == 1


def test_zero_escalation_budget_sends_what_the_drafter_saw(tmp_path, source_paths):
    kit = HarnessKit(make_openbook_config(tmp_path / "cache", {"escalation_token_budget": 0}))
    kit.client.script("mid-model", openbook_draft([THRESHOLD_CLAIM, FABRICATED_CLAIM]))
    kit.client.script("ground-model", grounding_report("yes", "supported"))
    kit.client.script("frontier-model", openbook_draft())

    result = kit.harness.answer(ask(source_paths))

    assert kit.client.calls[-1].messages == kit.client.calls[0].messages
    assert result.escalation_passages == result.passages_retrieved


def test_run_without_an_escalation_reports_no_escalation_input(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft())
    ob_kit.client.script("ground-model", grounding_report("yes", "supported", "supported"))

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.escalation_passages is None
    assert result.escalation_tokens_estimated is None


# --- refusals are final ---


def refused_result() -> CallResult:
    return CallResult(
        status="error",
        error_type="refused",
        error_message="the model refused the request (finish_reason=content_filter)",
        latency_ms=5.0,
        finish_reason="content_filter",
        native_finish_reason="refusal",
    )


def test_refused_escalation_is_final_and_keeps_the_checked_first_draft(ob_kit, source_paths):
    ob_kit.client.script("mid-model", openbook_draft([THRESHOLD_CLAIM, FABRICATED_CLAIM]))
    ob_kit.client.script("ground-model", grounding_report("yes", "supported"))
    ob_kit.client.script("frontier-model", refused_result())

    result = ob_kit.harness.answer(ask(source_paths))

    assert ob_kit.client.models_called() == ["mid-model", "ground-model", "frontier-model"]
    assert result.status == "ok"
    assert result.repaired is False
    assert result.repair_error_type == "refused"
    assert result.grounding_status == "partly_grounded"
    assert result.answer == GOOD_ANSWER
    assert [citation.passage_id for citation in result.citations] == ["S1-P1"]
    assert ob_kit.call_records[-1].error_type == "refused"
    assert ob_kit.call_records[-1].finish_reason == "content_filter"


def test_refused_escalation_of_an_abstention_abstains(ob_kit, source_paths):
    ob_kit.client.script("mid-model", unanswerable_draft())
    ob_kit.client.script("ground-model", grounding_report("yes"))
    ob_kit.client.script("frontier-model", refused_result())

    result = ob_kit.harness.answer(ask(source_paths))

    assert result.abstained is True
    assert result.repair_error_type == "refused"


def test_refused_draft_takes_no_fallback_hop(ob_kit, source_paths):
    ob_kit.client.script("mid-model", refused_result())

    result = ob_kit.harness.answer(ask(source_paths))

    assert ob_kit.client.models_called() == ["mid-model"]
    assert result.status == "error"
    assert result.error_type == "refused"
    assert result.fallback_used is False
