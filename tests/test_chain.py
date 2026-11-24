"""Chain mode: draft, schema gate, verify, at most one repair. Fake clients, no network."""

from __future__ import annotations

import copy
import json

import httpx
import pytest
from pydantic import ValidationError

from inference_arbiter import chain
from inference_arbiter.cli import format_trace, main
from inference_arbiter.clients import OpenRouterDecisionsClient
from inference_arbiter.config import HarnessConfig, load_config
from inference_arbiter.router import ChainNotConfiguredError, Harness, UnknownTierError
from inference_arbiter.schemas import AskRequest, DraftAnswer
from inference_arbiter.telemetry import Telemetry

from conftest import (
    BASE_CONFIG,
    CHAIN_VERIFIER_TIER,
    REAL_CONFIG_PATH,
    FakeClient,
    FakeTelemetry,
    HarnessKit,
    draft_json,
    error_result,
    make_chain_config,
    ok_result,
    triage_json,
    verdict_json,
)

QUESTION = "How many legs do three spiders have?"


def chain_request(**overrides) -> AskRequest:
    return AskRequest.from_question(QUESTION, mode="chain", **overrides)


def call_trail(kit: HarnessKit) -> list[tuple[str, str, str]]:
    return [(record.role, record.tier, record.status) for record in kit.call_records]


# --- supported ---


def test_supported_draft_is_returned_without_repair(chain_kit):
    chain_kit.client.script("triage-model", triage_json("local", 0.95))
    chain_kit.client.script("local-model", draft_json("Three spiders have 24 legs."))
    chain_kit.client.script("verify-model", verdict_json("supported", "correct"))

    result = chain_kit.harness.answer(chain_request())

    assert result.status == "ok"
    assert result.mode == "chain"
    assert result.answer == "Three spiders have 24 legs."
    assert result.verification_status == "supported"
    assert result.verification_critique == "correct"
    assert result.repaired is False
    assert result.schema_escalated is False
    assert result.verifier_tier == "verify"
    assert result.selected_tier == "local"
    assert result.final_tier == "local"
    assert result.final_model == "local-model"
    assert chain_kit.client.models_called() == ["triage-model", "local-model", "verify-model"]
    assert [(call.seq, call.role, call.tier) for call in result.calls] == [
        (1, "triage", "triage"),
        (2, "draft", "local"),
        (3, "verify", "verify"),
    ]


def test_draft_call_asks_for_the_draft_schema_and_keeps_the_question(chain_kit):
    chain_kit.client.script("local-model", draft_json())
    chain_kit.client.script("verify-model", verdict_json("supported"))

    chain_kit.harness.answer(chain_request(tier="local"))

    draft_call = chain_kit.client.calls[0]
    assert draft_call.json_schema == chain.draft_json_schema()
    assert draft_call.messages == [
        {"role": "system", "content": chain.DRAFT_SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION},
    ]


def test_forced_tier_forces_the_draft_tier_and_skips_triage(chain_kit):
    chain_kit.client.script("mid-model", draft_json("from mid"))
    chain_kit.client.script("verify-model", verdict_json("supported"))

    result = chain_kit.harness.answer(chain_request(tier="mid"))

    assert result.triage_status == "skipped"
    assert result.route_reason == "forced"
    assert result.selected_tier == "mid"
    assert result.answer == "from mid"
    assert chain_kit.client.models_called() == ["mid-model", "verify-model"]


# --- unsupported / uncertain then repaired ---


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_rejected_draft_is_repaired_once_by_the_repair_tier(chain_kit, verdict):
    chain_kit.client.script("local-model", draft_json("Three spiders have 18 legs.", ["A spider has 6 legs."]))
    chain_kit.client.script(
        "verify-model", verdict_json(verdict, "A spider has 8 legs, so 24.", ["A spider has 6 legs."])
    )
    chain_kit.client.script("frontier-model", draft_json("Three spiders have 24 legs."))

    result = chain_kit.harness.answer(chain_request(tier="local"))

    assert result.status == "ok"
    assert result.answer == "Three spiders have 24 legs."
    assert result.verification_status == verdict
    assert result.verification_critique == "A spider has 8 legs, so 24."
    assert result.repaired is True
    assert result.repair_error_type is None
    assert result.selected_tier == "local"
    assert result.final_tier == "frontier"
    assert result.final_model == "frontier-model"
    # One repair, no second verification.
    assert chain_kit.client.models_called() == ["local-model", "verify-model", "frontier-model"]
    assert call_trail(chain_kit) == [
        ("draft", "local", "ok"),
        ("verify", "verify", "ok"),
        ("repair", "frontier", "ok"),
    ]


def test_repair_call_gets_question_draft_and_specific_critique(chain_kit):
    chain_kit.client.script("local-model", draft_json("Three spiders have 18 legs.", ["A spider has 6 legs."]))
    chain_kit.client.script(
        "verify-model", verdict_json("unsupported", "A spider has 8 legs, so 24.", ["A spider has 6 legs."])
    )
    chain_kit.client.script("frontier-model", draft_json("24"))

    chain_kit.harness.answer(chain_request(tier="local"))

    repair_call = chain_kit.client.calls[2]
    assert repair_call.json_schema == chain.draft_json_schema()
    assert repair_call.messages[0] == {"role": "system", "content": chain.REPAIR_SYSTEM_PROMPT}
    repair_text = repair_call.messages[1]["content"]
    assert QUESTION in repair_text
    assert "Three spiders have 18 legs." in repair_text
    assert "A spider has 8 legs, so 24." in repair_text
    assert "- A spider has 6 legs." in repair_text


def test_max_repairs_zero_returns_the_rejected_draft_unrepaired():
    kit = HarnessKit(make_chain_config({"max_repairs": 0}))
    kit.client.script("local-model", draft_json("18"))
    kit.client.script("verify-model", verdict_json("unsupported", "wrong"))

    result = kit.harness.answer(chain_request(tier="local"))

    assert result.status == "ok"
    assert result.answer == "18"
    assert result.verification_status == "unsupported"
    assert result.repaired is False
    assert kit.client.models_called() == ["local-model", "verify-model"]


# --- verifier error ---


@pytest.mark.parametrize(
    ("verifier_result", "expected_call_status"),
    [
        (error_result("http_5xx"), "error"),
        (error_result("timeout"), "timeout"),
        (ok_result("this is not a verdict"), "invalid"),
    ],
)
def test_verifier_error_returns_the_draft_unverified_by_default(
    chain_kit, verifier_result, expected_call_status
):
    chain_kit.client.script("local-model", draft_json("the draft"))
    chain_kit.client.script("verify-model", verifier_result)

    result = chain_kit.harness.answer(chain_request(tier="local"))

    assert result.status == "ok"
    assert result.answer == "the draft"
    assert result.verification_status == "unavailable"
    assert result.verification_critique is None
    assert result.repaired is False
    assert result.final_tier == "local"
    assert chain_kit.client.models_called() == ["local-model", "verify-model"]
    assert call_trail(chain_kit)[-1] == ("verify", "verify", expected_call_status)


def test_verifier_error_triggers_repair_when_configured():
    kit = HarnessKit(make_chain_config({"on_verifier_error": "repair"}))
    kit.client.script("local-model", draft_json("the draft"))
    kit.client.script("verify-model", error_result("timeout"))
    kit.client.script("frontier-model", draft_json("the repaired answer"))

    result = kit.harness.answer(chain_request(tier="local"))

    assert result.status == "ok"
    assert result.answer == "the repaired answer"
    assert result.verification_status == "unavailable"
    assert result.repaired is True
    assert result.final_tier == "frontier"
    repair_text = kit.client.calls[2].messages[1]["content"]
    assert chain.NO_VERDICT_NOTE in repair_text


# --- schema gate ---


@pytest.mark.parametrize(
    "bad_draft_text",
    [
        "Three spiders have 24 legs.",
        json.dumps({"answer": "24"}),
        json.dumps({"answer": "", "key_claims": [], "assumptions": []}),
        json.dumps({"answer": "24", "key_claims": "8 legs each", "assumptions": []}),
    ],
)
def test_invalid_draft_escalates_once_per_the_escalation_map(chain_kit, bad_draft_text):
    chain_kit.client.script("local-model", ok_result(bad_draft_text))
    chain_kit.client.script("mid-model", draft_json("from mid"))
    chain_kit.client.script("verify-model", verdict_json("supported"))

    result = chain_kit.harness.answer(chain_request(tier="local"))

    assert result.status == "ok"
    assert result.answer == "from mid"
    assert result.schema_escalated is True
    assert result.fallback_used is False
    assert result.selected_tier == "local"
    assert result.final_tier == "mid"
    assert call_trail(chain_kit) == [
        ("draft", "local", "invalid"),
        ("draft", "mid", "ok"),
        ("verify", "verify", "ok"),
    ]
    assert chain_kit.call_records[0].error_type == chain.ERROR_SCHEMA_INVALID
    # The verifier saw the valid draft, never the malformed one.
    assert "from mid" in chain_kit.client.calls[2].messages[1]["content"]


def test_invalid_draft_without_escalation_entry_uses_fallback_tier(chain_kit):
    chain_kit.client.script("heavy-model", ok_result("not json"))
    chain_kit.client.script("mid-model", draft_json("from mid"))
    chain_kit.client.script("verify-model", verdict_json("supported"))

    result = chain_kit.harness.answer(chain_request(tier="local_heavy"))

    assert result.schema_escalated is True
    assert result.final_tier == "mid"


def test_double_schema_failure_fails_the_run_with_an_explicit_error(chain_kit):
    chain_kit.client.script("local-model", ok_result("not json"))
    chain_kit.client.script("mid-model", ok_result("still not json"))

    result = chain_kit.harness.answer(chain_request(tier="local"))

    assert result.status == "error"
    assert result.answer is None
    assert result.error_type == chain.ERROR_DRAFT_SCHEMA_INVALID
    assert "mid" in result.error_message
    assert result.schema_escalated is True
    assert result.verification_status == "skipped"
    assert result.repaired is False
    # Nothing malformed went downstream: no verifier call, no repair call, no third draft.
    assert chain_kit.client.models_called() == ["local-model", "mid-model"]
    assert call_trail(chain_kit) == [("draft", "local", "invalid"), ("draft", "mid", "invalid")]
    assert chain_kit.run_record.status == "error"


# --- draft call failure (transport level) ---


def test_failed_draft_call_takes_the_single_on_error_hop(chain_kit):
    chain_kit.client.script("local-model", error_result("http_5xx"))
    chain_kit.client.script("mid-model", draft_json("from mid"))
    chain_kit.client.script("verify-model", verdict_json("supported"))

    result = chain_kit.harness.answer(chain_request(tier="local"))

    assert result.status == "ok"
    assert result.answer == "from mid"
    assert result.fallback_used is True
    assert result.schema_escalated is False
    assert result.final_tier == "mid"


def test_failed_draft_then_invalid_hop_fails_without_a_third_attempt(chain_kit):
    chain_kit.client.script("local-model", error_result("http_5xx"))
    chain_kit.client.script("mid-model", ok_result("not json"))

    result = chain_kit.harness.answer(chain_request(tier="local"))

    assert result.status == "error"
    assert result.answer is None
    assert result.error_type == chain.ERROR_DRAFT_SCHEMA_INVALID
    assert chain_kit.client.models_called() == ["local-model", "mid-model"]


def test_failed_draft_with_cloud_fallback_disabled_fails_the_run():
    kit = HarnessKit(make_chain_config(allow_cloud_fallback=False))
    kit.client.script("local-model", error_result("timeout"))

    result = kit.harness.answer(chain_request(tier="local"))

    assert result.status == "error"
    assert result.error_type == "timeout"
    assert result.fallback_skipped_reason == "cloud_fallback_disabled"
    assert kit.client.models_called() == ["local-model"]


# --- repair failure ---


@pytest.mark.parametrize(
    ("repair_result", "expected_error_type", "expected_call_status"),
    [
        (error_result("http_5xx"), "http_5xx", "error"),
        (error_result("missing_api_key"), "missing_api_key", "error"),
        (ok_result("a repair that is not JSON"), chain.ERROR_SCHEMA_INVALID, "invalid"),
    ],
)
def test_failed_repair_returns_the_draft_and_records_the_failure(
    chain_kit, repair_result, expected_error_type, expected_call_status
):
    chain_kit.client.script("local-model", draft_json("the draft"))
    chain_kit.client.script("verify-model", verdict_json("unsupported", "wrong"))
    chain_kit.client.script("frontier-model", repair_result)

    result = chain_kit.harness.answer(chain_request(tier="local"))

    assert result.status == "ok"
    assert result.answer == "the draft"
    assert result.verification_status == "unsupported"
    assert result.repaired is False
    assert result.repair_error_type == expected_error_type
    # The tier that actually wrote the final answer is reported, not the repair tier.
    assert result.final_tier == "local"
    assert result.final_model == "local-model"
    assert result.fallback_used is False
    # frontier's on_error target (mid) is never called to stand in for the repair.
    assert chain_kit.client.models_called() == ["local-model", "verify-model", "frontier-model"]
    assert call_trail(chain_kit)[-1] == ("repair", "frontier", expected_call_status)
    assert chain_kit.run_record.final_tier == "local"
    assert chain_kit.run_record.repaired is False


# --- telemetry ---


def test_chain_run_record_and_call_rows(chain_kit):
    chain_kit.client.script("triage-model", triage_json("local", 0.95))
    chain_kit.client.script("local-model", draft_json("18"))
    chain_kit.client.script("verify-model", verdict_json("unsupported", "wrong count"))
    chain_kit.client.script("frontier-model", draft_json("24"))

    chain_kit.harness.answer(chain_request())

    run = chain_kit.run_record
    assert run.mode == "chain"
    assert run.verification_status == "unsupported"
    assert run.verification_critique == "wrong count"
    assert run.repaired is True
    assert run.schema_escalated is False
    assert run.verifier_tier == "verify"
    assert run.selected_tier == "local"
    assert run.final_tier == "frontier"
    assert [(record.seq, record.role, record.tier, record.model) for record in chain_kit.call_records] == [
        (1, "triage", "triage", "triage-model"),
        (2, "draft", "local", "local-model"),
        (3, "verify", "verify", "verify-model"),
        (4, "repair", "frontier", "frontier-model"),
    ]
    for record in chain_kit.call_records:
        assert record.latency_ms == 12.0
        assert record.prompt_tokens == 10
        assert record.completion_tokens == 5
        assert record.cost_usd == 0.0


def test_chain_run_lands_in_a_real_database(tmp_path):
    db_path = tmp_path / "telemetry.db"
    client = FakeClient()
    client.script("local-model", ok_result("not json"))
    client.script("mid-model", draft_json("the draft"))
    client.script("verify-model", verdict_json("uncertain", "cannot tell"))
    client.script("frontier-model", draft_json("the repair"))
    telemetry = Telemetry(db_path)
    harness = Harness(make_chain_config(), {"ollama": client, "openrouter": client}, telemetry)

    result = harness.answer(chain_request(tier="local"))
    telemetry.close()

    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        run_row = conn.execute("SELECT * FROM runs WHERE trace_id = ?", (result.trace_id,)).fetchone()
        call_rows = conn.execute(
            "SELECT role, tier, status FROM calls WHERE trace_id = ? ORDER BY seq", (result.trace_id,)
        ).fetchall()
    finally:
        conn.close()
    assert run_row["mode"] == "chain"
    assert run_row["verification_status"] == "uncertain"
    assert run_row["verification_critique"] == "cannot tell"
    assert run_row["repaired"] == 1
    assert run_row["schema_escalated"] == 1
    assert run_row["verifier_tier"] == "verify"
    assert [tuple(row) for row in call_rows] == [
        ("draft", "local", "invalid"),
        ("draft", "mid", "ok"),
        ("verify", "verify", "ok"),
        ("repair", "frontier", "ok"),
    ]


# --- route mode is unchanged ---


def test_route_mode_makes_no_chain_calls(chain_kit):
    chain_kit.client.script("triage-model", triage_json("local", 0.95))
    chain_kit.client.script("local-model", ok_result("plain text answer"))

    request = AskRequest.from_question(QUESTION, mode="route")
    result = chain_kit.harness.answer(request)

    assert request.mode == "route"
    assert result.mode == "route"
    assert result.answer == "plain text answer"
    assert result.verification_status == "skipped"
    assert result.verification_critique is None
    assert result.repaired is False
    assert result.schema_escalated is False
    assert result.verifier_tier is None
    assert chain_kit.client.models_called() == ["triage-model", "local-model"]
    answer_call = chain_kit.client.calls[1]
    assert answer_call.json_schema is None
    assert answer_call.messages == [{"role": "user", "content": QUESTION}]
    assert [(record.seq, record.role) for record in chain_kit.call_records] == [(1, "triage"), (2, "answer")]
    assert chain_kit.run_record.mode == "route"
    assert chain_kit.run_record.verification_status == "skipped"


def test_route_mode_works_without_a_chain_block(kit):
    kit.client.script("local-model", ok_result("plain"))

    result = kit.harness.answer(AskRequest.from_question(QUESTION, tier="local", mode="route"))

    assert result.status == "ok"
    assert result.mode == "route"
    assert result.verification_status == "skipped"


# --- request and config validation ---


def test_chain_mode_without_a_chain_block_is_a_usage_error(kit):
    with pytest.raises(ChainNotConfiguredError):
        kit.harness.answer(chain_request(tier="local"))
    assert kit.client.calls == []


def test_verifier_tier_can_be_overridden_per_request_but_must_be_a_verifier():
    raw_tiers = copy.deepcopy(BASE_CONFIG["tiers"])
    raw_tiers["verify"] = copy.deepcopy(CHAIN_VERIFIER_TIER)
    raw_tiers["verify_b"] = {**CHAIN_VERIFIER_TIER, "model": "verify-b-model"}
    kit = HarnessKit(make_chain_config(tiers=raw_tiers))
    kit.client.script("local-model", draft_json())
    kit.client.script("verify-b-model", verdict_json("supported"))

    result = kit.harness.answer(chain_request(tier="local", verifier_tier="verify_b"))
    assert result.verifier_tier == "verify_b"
    assert kit.run_record.verifier_tier == "verify_b"

    with pytest.raises(UnknownTierError, match="unknown verifier tier"):
        kit.harness.answer(chain_request(tier="local", verifier_tier="mid"))


def test_verifier_tier_is_not_an_answer_tier(chain_kit):
    assert "verify" not in chain_kit.config.answer_tier_names()
    with pytest.raises(UnknownTierError):
        chain_kit.harness.answer(chain_request(tier="verify"))
    with pytest.raises(UnknownTierError):
        chain_kit.harness.answer(AskRequest.from_question(QUESTION, tier="verify"))


def test_chain_only_request_options_are_rejected_in_route_mode():
    with pytest.raises(ValidationError, match="needs mode 'chain'"):
        AskRequest.from_question(QUESTION, simulate="verifier-error")
    with pytest.raises(ValidationError, match="needs mode 'chain'"):
        AskRequest.from_question(QUESTION, simulate="draft-invalid")
    with pytest.raises(ValidationError, match="needs mode 'chain'"):
        AskRequest.from_question(QUESTION, verifier_tier="verify")


@pytest.mark.parametrize(
    ("chain_block", "expected_message"),
    [
        ({"verifier_tier": "mid", "repair_tier": "frontier"}, "is not a verifier tier"),
        ({"verifier_tier": "verify", "repair_tier": "verify"}, "is not an answer tier"),
        ({"verifier_tier": "verify", "repair_tier": "triage"}, "is not an answer tier"),
        ({"verifier_tier": "verify", "repair_tier": "frontier", "max_repairs": 2}, "max_repairs"),
        (
            {"verifier_tier": "verify", "repair_tier": "frontier", "on_verifier_error": "retry"},
            "on_verifier_error",
        ),
    ],
)
def test_chain_block_is_validated(chain_block, expected_message):
    raw_config = copy.deepcopy(BASE_CONFIG)
    raw_config["tiers"]["verify"] = copy.deepcopy(CHAIN_VERIFIER_TIER)
    raw_config["chain"] = chain_block
    with pytest.raises(ValidationError, match=expected_message):
        HarnessConfig.model_validate(raw_config)


def test_verifier_tier_cannot_be_a_fallback_or_escalation_target():
    for field_name, value in (("fallback_tier", "verify"), ("escalation", {"local": "verify"})):
        with pytest.raises(ValidationError, match="is not an answer tier"):
            make_chain_config(**{field_name: value})

    raw_tiers = copy.deepcopy(BASE_CONFIG["tiers"])
    raw_tiers["verify"] = copy.deepcopy(CHAIN_VERIFIER_TIER)
    raw_tiers["local"]["on_error"] = "verify"
    with pytest.raises(ValidationError, match="must be an answer tier"):
        make_chain_config(tiers=raw_tiers)


def test_shipped_config_has_a_chain_block_and_two_verifier_tiers():
    config = load_config(REAL_CONFIG_PATH)

    assert config.chain is not None
    assert config.chain.verifier_tier == "verify"
    assert config.chain.repair_tier == "frontier"
    assert config.chain.on_verifier_error == "return_unverified"
    assert config.chain.max_repairs == 1
    assert {"verify", "verify_jev"} <= set(config.verifier_tier_names())
    assert config.providers[config.tiers["verify_jev"].provider].kind == "openrouter_decisions"
    assert config.tiers["verify"].model != config.tiers["local"].model
    assert "verify" not in config.answer_tier_names()
    assert "verify_jev" not in config.answer_tier_names()


# --- Jev verifier inside the chain ---


def test_chain_with_a_decisions_verifier_repairs_on_its_verdict(monkeypatch):
    monkeypatch.setenv("TEST_CHAIN_DECISIONS_KEY", "test-key-not-a-real-secret")
    raw_config = copy.deepcopy(BASE_CONFIG)
    raw_config["providers"]["decisions"] = {
        "kind": "openrouter_decisions",
        "base_url": "http://decisions.invalid/api",
        "api_key_env": "TEST_CHAIN_DECISIONS_KEY",
    }
    raw_config["tiers"]["verify_jev"] = {
        "model": "typesafe/jev-test",
        "provider": "decisions",
        "role": "verifier",
        "cloud": True,
        "timeout_s": 5,
    }
    raw_config["chain"] = {
        "verifier_tier": "verify_jev",
        "repair_tier": "frontier",
        "repair_style": "repair",
    }
    config = HarnessConfig.model_validate(raw_config)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answers": {
                    "answers_question": {"noul": 0.95},
                    "unflagged_false_premise": {"noul": 0.88},
                    "self_contradiction": {"noul": 0.02},
                },
                "usage": {"input_tokens": 150, "output_tokens": 12, "cost": 0.000004},
            },
        )

    decisions_client = OpenRouterDecisionsClient(
        "http://decisions.invalid/api", "TEST_CHAIN_DECISIONS_KEY", transport=httpx.MockTransport(handler)
    )
    chat_client = FakeClient()
    chat_client.script("local-model", draft_json("the draft"))
    chat_client.script("frontier-model", draft_json("the repair"))
    telemetry = FakeTelemetry()
    harness = Harness(
        config, {"ollama": chat_client, "openrouter": chat_client, "decisions": decisions_client}, telemetry
    )

    result = harness.answer(chain_request(tier="local"))

    assert result.verification_status == "unsupported"
    assert "false premise" in result.verification_critique
    assert result.repaired is True
    assert result.answer == "the repair"
    verify_record = telemetry.recorded[-1][1][1]
    assert (verify_record.role, verify_record.tier, verify_record.provider) == ("verify", "verify_jev", "decisions")
    assert verify_record.cost_usd == 0.000004
    assert verify_record.prompt_tokens == 150


# --- simulate and CLI ---


def test_simulated_invalid_draft_exercises_the_schema_gate_without_calling_the_drafter(chain_kit):
    chain_kit.client.script("mid-model", draft_json("from mid"))
    chain_kit.client.script("verify-model", verdict_json("supported"))

    result = chain_kit.harness.answer(chain_request(tier="local", simulate="draft-invalid"))

    assert result.schema_escalated is True
    assert result.final_tier == "mid"
    assert chain_kit.client.models_called() == ["mid-model", "verify-model"]
    assert chain_kit.call_records[0].error_type == chain.ERROR_SIMULATED_INVALID
    assert chain_kit.call_records[0].status == "invalid"


def test_simulated_verifier_error_exercises_on_verifier_error_without_calling_the_verifier(chain_kit):
    chain_kit.client.script("local-model", draft_json("the draft"))

    result = chain_kit.harness.answer(chain_request(tier="local", simulate="verifier-error"))

    assert result.answer == "the draft"
    assert result.verification_status == "unavailable"
    assert chain_kit.client.models_called() == ["local-model"]
    assert chain_kit.call_records[1].error_type == "simulated_error"


def test_simulated_answer_error_in_chain_mode_takes_the_on_error_hop(chain_kit):
    chain_kit.client.script("mid-model", draft_json("from mid"))
    chain_kit.client.script("verify-model", verdict_json("supported"))

    result = chain_kit.harness.answer(chain_request(tier="local", simulate="answer-error"))

    assert result.fallback_used is True
    assert result.final_tier == "mid"


def test_trace_shows_each_chain_step_and_the_verdict(chain_kit):
    chain_kit.client.script("local-model", draft_json("18"))
    chain_kit.client.script("verify-model", verdict_json("unsupported", "A spider has 8 legs."))
    chain_kit.client.script("frontier-model", error_result("missing_api_key"))

    trace = format_trace(chain_kit.harness.answer(chain_request(tier="local")))

    assert "verifier verify, verdict unsupported, repaired: False" in trace
    assert "critique      A spider has 8 legs." in trace
    assert "repair        failed (missing_api_key)" in trace
    for role in ("draft", "verify", "repair"):
        assert f"{role:<6}" in trace


def test_route_trace_has_no_chain_lines(chain_kit):
    chain_kit.client.script("local-model", ok_result("plain"))

    trace = format_trace(chain_kit.harness.answer(AskRequest.from_question(QUESTION, tier="local")))

    assert "chain " not in trace
    assert "critique" not in trace


def test_cli_rejects_chain_only_flags_without_chain(tmp_path, capsys, monkeypatch):
    # The CLI loads ./.env; run from an empty directory so no real key file is read.
    monkeypatch.chdir(tmp_path)
    db_path = str(tmp_path / "telemetry.db")
    config_path = str(REAL_CONFIG_PATH)

    exit_code = main(["ask", "q", "--simulate", "verifier-error", "--config", config_path, "--db", db_path])
    assert exit_code == 2
    assert "needs mode 'chain'" in capsys.readouterr().err

    exit_code = main(["ask", "q", "--verifier-tier", "verify", "--config", config_path, "--db", db_path])
    assert exit_code == 2

    exit_code = main(
        ["ask", "q", "--chain", "--verifier-tier", "mid", "--tier", "local", "--config", config_path, "--db", db_path]
    )
    assert exit_code == 2
    assert "unknown verifier tier" in capsys.readouterr().err


def test_draft_schema_puts_working_first_and_requires_it():
    schema = chain.draft_json_schema()
    assert list(schema["properties"])[0] == "working"
    assert schema["required"][0] == "working"
    # Older stored drafts without `working` still validate.
    assert DraftAnswer(answer="a", key_claims=[], assumptions=[]).working == ""


# --- clean escalation (the shipped default) ---


def escalate_kit(chain_overrides: dict | None = None, tiers: dict | None = None) -> HarnessKit:
    overrides = {"repair_style": "escalate", **(chain_overrides or {})}
    if tiers is not None:
        return HarnessKit(make_chain_config(overrides, tiers=tiers))
    return HarnessKit(make_chain_config(overrides))


def test_escalation_sends_only_the_original_question_as_a_plain_call():
    kit = escalate_kit()
    kit.client.script("local-model", draft_json("Three spiders have 18 legs."))
    kit.client.script("verify-model", verdict_json("unsupported", "spiders have eight legs"))
    kit.client.script("frontier-model", ok_result("Three spiders have 24 legs."))

    result = kit.harness.answer(chain_request(tier="local"))

    assert result.repaired is True
    assert result.answer == "Three spiders have 24 legs."
    assert result.final_tier == "frontier"
    escalation_call = kit.client.calls[-1]
    assert escalation_call.model == "frontier-model"
    assert escalation_call.json_schema is None
    sent_text = json.dumps(escalation_call.messages)
    assert QUESTION in sent_text
    assert "18 legs" not in sent_text  # the rejected draft is not sent
    assert "spiders have eight legs" not in sent_text  # nor the critique


def test_failed_escalation_returns_the_draft():
    kit = escalate_kit()
    kit.client.script("local-model", draft_json("the draft"))
    kit.client.script("verify-model", verdict_json("unsupported"))
    kit.client.script("frontier-model", error_result("http_4xx"))

    result = kit.harness.answer(chain_request(tier="local"))

    assert result.status == "ok"
    assert result.repaired is False
    assert result.answer == "the draft"
    assert result.final_tier == "local"


# --- a model never verifies its own draft ---


def same_model_tiers() -> dict:
    raw_tiers = copy.deepcopy(BASE_CONFIG["tiers"])
    raw_tiers["verify"] = {**CHAIN_VERIFIER_TIER, "model": "mid-model"}
    raw_tiers["verify_b"] = {**CHAIN_VERIFIER_TIER, "model": "verify-b-model"}
    return raw_tiers


def test_same_model_draft_is_reported_unverified_without_calling_the_verifier():
    kit = escalate_kit(tiers=same_model_tiers())
    kit.client.script("mid-model", draft_json("mid draft"))

    result = kit.harness.answer(chain_request(tier="mid"))

    assert result.verification_status == "unavailable"
    assert "same model" in result.verification_critique
    assert result.answer == "mid draft"
    assert kit.client.models_called() == ["mid-model"]  # one draft call, no verifier call


def test_same_model_draft_uses_the_alternate_verifier_when_configured():
    kit = escalate_kit({"alt_verifier_tier": "verify_b"}, tiers=same_model_tiers())
    kit.client.script("mid-model", draft_json("mid draft"))
    kit.client.script("verify-b-model", verdict_json("supported"))

    result = kit.harness.answer(chain_request(tier="mid"))

    assert result.verification_status == "supported"
    assert result.verifier_tier == "verify_b"
    assert kit.client.models_called() == ["mid-model", "verify-b-model"]


def test_other_tiers_keep_the_main_verifier_when_an_alternate_is_configured():
    kit = escalate_kit({"alt_verifier_tier": "verify_b"}, tiers=same_model_tiers())
    kit.client.script("local-model", draft_json("local draft"))
    kit.client.script("mid-model", verdict_json("supported"))

    result = kit.harness.answer(chain_request(tier="local"))

    assert result.verifier_tier == "verify"
    assert result.verification_status == "supported"


def test_alternate_verifier_must_run_a_different_model():
    raw_tiers = copy.deepcopy(BASE_CONFIG["tiers"])
    raw_tiers["verify"] = copy.deepcopy(CHAIN_VERIFIER_TIER)
    raw_tiers["verify_b"] = copy.deepcopy(CHAIN_VERIFIER_TIER)
    with pytest.raises(ValidationError, match="different model"):
        make_chain_config({"alt_verifier_tier": "verify_b"}, tiers=raw_tiers)
