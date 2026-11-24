from __future__ import annotations

import hashlib

import pytest

from inference_arbiter.router import UnknownTierError, choose_route
from inference_arbiter.schemas import AskRequest, Message, TriageDecision, TriageOutcome

from conftest import HarnessKit, error_result, make_config, ok_result, triage_json


def ask(question: str = "what is a nonce?", **overrides) -> AskRequest:
    overrides.setdefault("mode", "route")
    return AskRequest.from_question(question, **overrides)


def test_forced_tier_skips_triage(kit):
    kit.client.script("code-model", ok_result("forced answer"))

    result = kit.harness.answer(ask(tier="code"))

    assert result.status == "ok"
    assert result.answer == "forced answer"
    assert result.route_reason == "forced"
    assert result.triage_status == "skipped"
    assert result.selected_tier == result.final_tier == "code"
    assert kit.client.models_called() == ["code-model"]


@pytest.mark.parametrize(
    ("triage_result", "expected_reason", "expected_triage_status"),
    [
        (error_result("timeout"), "triage_timeout", "timeout"),
        (error_result("connection"), "triage_error", "error"),
        (ok_result("definitely not json"), "triage_invalid", "invalid"),
    ],
)
def test_failed_triage_routes_to_fallback_tier(
    kit, triage_result, expected_reason, expected_triage_status
):
    kit.client.script("triage-model", triage_result)
    kit.client.script("mid-model", ok_result("mid answer"))

    result = kit.harness.answer(ask())

    assert result.status == "ok"
    assert result.route_reason == expected_reason
    assert result.triage_status == expected_triage_status
    assert result.triage_label is None
    assert result.selected_tier == "mid"
    assert kit.client.models_called() == ["triage-model", "mid-model"]
    assert kit.call_records[0].status == expected_triage_status


@pytest.mark.parametrize(
    ("label", "expected_tier"),
    [("local", "mid"), ("code", "mid"), ("mid", "frontier"), ("frontier", "frontier")],
)
def test_low_confidence_escalates_one_step(kit, label, expected_tier):
    kit.client.script("triage-model", triage_json(label, 0.79))
    kit.client.script(f"{expected_tier}-model", ok_result())

    result = kit.harness.answer(ask())

    assert result.route_reason == "low_confidence"
    assert result.triage_label == label
    assert result.selected_tier == expected_tier


@pytest.mark.parametrize("label", ["local", "code", "mid", "frontier"])
def test_confident_label_routes_to_its_tier(kit, label):
    kit.client.script("triage-model", triage_json(label, 0.80))
    kit.client.script(f"{label}-model", ok_result())

    result = kit.harness.answer(ask())

    assert result.route_reason == "triage_label"
    assert result.selected_tier == result.final_tier == label
    assert result.fallback_used is False


def test_local_heavy_is_never_auto_routed():
    config = make_config()
    possible_outcomes = [
        TriageOutcome(status="timeout"),
        TriageOutcome(status="error"),
        TriageOutcome(status="invalid"),
    ]
    for label in ("local", "code", "mid", "frontier"):
        for confidence in (0.1, 0.95):
            decision = TriageDecision(label=label, confidence=confidence, reason="r")
            possible_outcomes.append(TriageOutcome(status="ok", decision=decision))

    for outcome in possible_outcomes:
        selected_tier, _ = choose_route(config, None, outcome)
        assert selected_tier != "local_heavy"


def test_local_heavy_is_reachable_when_forced_and_sends_effort_as_think(kit):
    kit.client.script("heavy-model", ok_result())

    result = kit.harness.answer(ask(tier="local_heavy", effort="high"))

    assert result.final_tier == "local_heavy"
    assert result.effort == "high"
    assert kit.client.calls[0].think == "high"


def test_unknown_or_triage_forced_tier_is_rejected(kit):
    with pytest.raises(UnknownTierError):
        kit.harness.answer(ask(tier="nope"))
    with pytest.raises(UnknownTierError):
        kit.harness.answer(ask(tier="triage"))
    assert kit.client.calls == []


def test_effort_override_reaches_the_answer_call(kit):
    kit.client.script("frontier-model", ok_result())

    result = kit.harness.answer(ask(tier="frontier", effort="low"))

    assert kit.client.calls[0].effort == "low"
    assert result.effort == "low"
    assert kit.run_record.effort == "low"


def test_answer_call_carries_tier_settings_from_config():
    kit = HarnessKit(make_config())
    kit.config.tiers["code"].num_ctx = 32768
    kit.config.tiers["code"].keep_alive = "10m"
    kit.client.script("code-model", ok_result())

    kit.harness.answer(ask(tier="code"))

    answer_call = kit.client.calls[0]
    assert answer_call.num_ctx == 32768
    assert answer_call.keep_alive == "10m"
    assert answer_call.timeout_s == 30
    assert answer_call.json_schema is None


def test_answer_call_receives_text_only_messages(kit):
    kit.client.script("local-model", ok_result())
    image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZZZ"}}
    request = AskRequest(
        messages=[Message(role="user", content=[{"type": "text", "text": "describe"}, image_part])],
        tier="local",
    )

    kit.harness.answer(request)

    assert kit.client.calls[0].messages == [{"role": "user", "content": "describe"}]


def test_telemetry_gets_one_run_and_one_record_per_call(kit):
    kit.client.script("triage-model", triage_json("local", 0.91, "simple fact"))
    kit.client.script("local-model", ok_result("42"))

    result = kit.harness.answer(ask("what is six times seven?"))

    assert len(kit.telemetry.recorded) == 1
    run_record = kit.run_record
    assert run_record.trace_id == result.trace_id
    assert run_record.prompt_text == "what is six times seven?"
    assert run_record.prompt_sha256 == hashlib.sha256(b"what is six times seven?").hexdigest()
    assert run_record.config_sha256 == "test-config-sha"
    assert run_record.triage_status == "ok"
    assert run_record.triage_label == "local"
    assert run_record.triage_confidence == 0.91
    assert run_record.triage_reason == "simple fact"
    assert run_record.route_reason == "triage_label"
    assert run_record.selected_tier == run_record.final_tier == "local"
    assert run_record.final_model == "local-model"
    assert run_record.fallback_used is False
    assert run_record.status == "ok"
    assert run_record.error_type is None

    assert [(record.seq, record.role, record.tier) for record in kit.call_records] == [
        (1, "triage", "triage"),
        (2, "answer", "local"),
    ]
    answer_record = kit.call_records[1]
    assert answer_record.provider == "ollama"
    assert answer_record.model == "local-model"
    assert answer_record.timeout_s == 30
    assert answer_record.cost_source == "local_zero"
    assert answer_record.ollama_load_duration_ms == 3.0


def test_prompt_text_is_omitted_when_storage_is_disabled():
    kit = HarnessKit(make_config(store_prompt_text=False))
    kit.client.script("local-model", ok_result())

    kit.harness.answer(ask("a private question", tier="local"))

    assert kit.run_record.prompt_text is None
    assert kit.run_record.prompt_sha256 == hashlib.sha256(b"a private question").hexdigest()


def test_threshold_comes_from_config():
    kit = HarnessKit(make_config(confidence_threshold=0.50))
    kit.client.script("triage-model", triage_json("local", 0.55))
    kit.client.script("local-model", ok_result())

    result = kit.harness.answer(ask())

    assert result.route_reason == "triage_label"
    assert result.selected_tier == "local"
