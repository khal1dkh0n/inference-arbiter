"""The computation check: the call, the fail-closed reading, the verdict table, its config
and the accuracy script. Fakes only."""

from __future__ import annotations

import itertools
import json
import sys

import pytest
import yaml
from pydantic import ValidationError

from inference_arbiter import computation_check
from inference_arbiter.clients import ERROR_TIMEOUT
from inference_arbiter.config import load_config
from inference_arbiter.schemas import ComputationCheckDecision, ComputationCheckOutcome, Message

from conftest import BASE_CONFIG, REPO_ROOT, FakeClient, error_result, make_config, make_execute_config, ok_result

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_computation  # noqa: E402

THRESHOLD = 0.8


def check_json(computable: bool, confidence: float, reason: str = "test reason"):
    return ok_result(json.dumps({"reason": reason, "computable": computable, "confidence": confidence}))


def question(text: str = "What is 4637 multiplied by 8219?") -> list[Message]:
    return [Message(role="user", content=text)]


# --- the call ---


def test_the_call_asks_the_check_tier_with_the_schema_and_no_thinking():
    config = make_config()
    client = FakeClient()
    client.script("triage-model", check_json(True, 0.9))

    outcome = computation_check.run_computation_check(config, client, question())

    call = client.calls[0]
    assert call.model == "triage-model"
    assert call.messages[0] == {"role": "system", "content": computation_check.CHECK_SYSTEM_PROMPT}
    assert call.messages[1] == {"role": "user", "content": "What is 4637 multiplied by 8219?"}
    assert call.json_schema == ComputationCheckDecision.model_json_schema()
    assert call.think is False
    assert call.temperature == 0.0
    assert call.max_output_tokens == 200
    assert outcome.status == "ok"
    assert (outcome.decision.computable, outcome.decision.confidence) == (True, 0.9)


def test_a_tier_without_a_think_setting_gets_none_and_keeps_its_temperature():
    config = make_config(auto={"computation_check_tier": "local"})
    config.tiers["local"].temperature = 0.3

    call = computation_check.build_check_call(config.tiers["local"], "q")

    assert call.model == "local-model"
    assert call.think is None
    assert call.temperature == 0.3


def test_only_the_last_user_message_is_sent_and_it_is_truncated():
    config = make_config()
    client = FakeClient()
    client.script("triage-model", check_json(False, 0.9))
    messages = [
        Message(role="user", content="an earlier question"),
        Message(role="assistant", content="an earlier answer"),
        Message(role="user", content="H" * 500 + "T" * 500),
    ]

    computation_check.run_computation_check(config, client, messages)

    sent = client.calls[0].messages[1]["content"]
    assert "earlier" not in sent
    assert sent.startswith("H" * 100) and sent.endswith("T" * 100)
    assert len(sent) < 1000


def test_the_reason_is_declared_before_the_verdict():
    assert list(ComputationCheckDecision.model_json_schema()["properties"]) == ["reason", "computable", "confidence"]


def test_the_prompt_names_both_answers_and_the_json_fields():
    prompt = computation_check.CHECK_SYSTEM_PROMPT
    for needed in ("Do not answer", "standard library", "computable: true", "computable: false", "confidence"):
        assert needed in prompt
    assert prompt.isascii()


# --- fail-closed reading of the call ---


@pytest.mark.parametrize(
    ("call_result", "status"),
    [
        (error_result(ERROR_TIMEOUT), "timeout"),
        (error_result("http_5xx"), "error"),
        (error_result("connection_error"), "error"),
        (ok_result("not json"), "invalid"),
        (ok_result(json.dumps({"reason": "r", "computable": True})), "invalid"),
        (ok_result(json.dumps({"reason": "r", "computable": True, "confidence": 1.5})), "invalid"),
        (ok_result(json.dumps({"reason": "r", "computable": "maybe", "confidence": 0.9})), "invalid"),
        (ok_result(json.dumps({"reason": "r", "computable": True, "confidence": 0.9, "extra": 1})), "invalid"),
        (ok_result(""), "invalid"),
    ],
)
def test_a_failed_or_invalid_call_is_never_a_yes(call_result, status):
    outcome = computation_check.interpret_check_result(call_result)

    assert outcome.status == status
    assert outcome.decision is None
    assert outcome.call is call_result
    assert computation_check.verdict_of(outcome, THRESHOLD) == "failed"


def test_a_skipped_check_is_not_run():
    assert computation_check.verdict_of(ComputationCheckOutcome(status="skipped"), THRESHOLD) == "not_run"


# --- the verdict table ---


def expected_verdict(status, computable, confidence):
    if status == "skipped":
        return "not_run"
    if status in ("timeout", "error", "invalid"):
        return "failed"
    if computable is None:
        return "failed"
    if computable is False:
        return "no"
    if confidence is None:
        return "failed"
    return "yes" if confidence >= THRESHOLD else "low_confidence"


@pytest.mark.parametrize(
    ("status", "computable", "confidence"),
    list(
        itertools.product(
            ["ok", "timeout", "error", "invalid", "skipped"],
            [None, True, False],
            [None, 0.0, 0.79, 0.8, 0.81, 1.0],
        )
    ),
)
def test_verdict_table(status, computable, confidence):
    verdict = computation_check.decide(status, computable, confidence, THRESHOLD)

    assert verdict == expected_verdict(status, computable, confidence)


def test_yes_needs_an_ok_call_a_true_answer_and_enough_confidence():
    combinations = itertools.product(
        ["ok", "timeout", "error", "invalid", "skipped"], [None, True, False], [None, 0.0, 0.79, 0.8, 1.0]
    )
    yes_rows = [
        row for row in combinations if computation_check.decide(*row, THRESHOLD) == "yes"
    ]

    assert yes_rows == [("ok", True, 0.8), ("ok", True, 1.0)]


@pytest.mark.parametrize(
    ("threshold", "confidence", "verdict"),
    [(0.0, 0.0, "yes"), (1.0, 0.99, "low_confidence"), (1.0, 1.0, "yes"), (0.5, 0.49, "low_confidence")],
)
def test_the_threshold_itself_is_a_yes(threshold, confidence, verdict):
    assert computation_check.decide("ok", True, confidence, threshold) == verdict


# --- config ---


def test_config_defaults_leave_the_check_off():
    config = make_config()

    assert config.auto.computation_check is False
    assert config.auto.computation_check_tier == "triage"
    assert config.auto.computation_check_threshold == 0.8
    assert config.auto.computation_cross_check is True


def test_shipped_config_documents_and_sets_the_check():
    config = load_config("config/models.yaml")
    text = (REPO_ROOT / "config" / "models.yaml").read_text(encoding="utf-8")

    assert config.auto.computation_check is True
    assert config.auto.computation_check_tier == "local"
    assert config.tiers["local"].cloud is False
    assert config.auto.computation_check_threshold == 0.8
    assert config.auto.computation_cross_check is True
    assert config.execute is not None
    for field in (
        "computation_check ", "computation_check_tier", "computation_check_threshold", "computation_cross_check"
    ):
        assert f"#   {field}" in text
    assert "Auto mode never picks execute mode" not in text


def test_the_check_tier_may_be_an_answer_tier():
    assert make_config(auto={"computation_check_tier": "local"}).auto.computation_check_tier == "local"


def test_the_check_tier_must_exist_and_be_a_chat_tier():
    with pytest.raises(ValidationError, match="auto.computation_check_tier 'nope' is not a triage tier"):
        make_config(auto={"computation_check_tier": "nope"})

    raw_tiers = dict(BASE_CONFIG["tiers"])
    raw_tiers["verify"] = {"model": "v", "provider": "ollama", "role": "verifier", "timeout_s": 5}
    with pytest.raises(ValidationError, match="auto.computation_check_tier 'verify' is not a triage tier"):
        make_config(tiers=raw_tiers, auto={"computation_check_tier": "verify"})

    providers = dict(BASE_CONFIG["providers"])
    providers["decisions"] = {"kind": "openrouter_decisions", "base_url": "http://d.invalid", "api_key_env": "K"}
    raw_tiers = dict(BASE_CONFIG["tiers"])
    raw_tiers["triage_typed"] = {"model": "t", "provider": "decisions", "role": "triage", "timeout_s": 5}
    with pytest.raises(ValidationError, match="uses a decisions provider"):
        make_config(providers=providers, tiers=raw_tiers, auto={"computation_check_tier": "triage_typed"})


def test_the_check_on_needs_an_execute_block():
    with pytest.raises(ValidationError, match="auto.computation_check needs an `execute:` block"):
        make_config(auto={"computation_check": True})
    assert make_execute_config(auto={"computation_check": True}).auto.computation_check is True


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_the_threshold_is_a_probability(threshold):
    with pytest.raises(ValidationError):
        make_config(auto={"computation_check_threshold": threshold})


# --- labels in the question files ---


@pytest.mark.parametrize(
    ("path", "cases", "computable"),
    [("eval/golden_questions.yaml", 36, 15), ("eval/hard_questions.yaml", 26, 26)],
)
def test_every_case_carries_a_computable_label(path, cases, computable):
    loaded = yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8"))["cases"]

    assert len(loaded) == cases
    assert all(isinstance(case.get("computable"), bool) for case in loaded)
    assert sum(case["computable"] for case in loaded) == computable


def test_golden_labels_follow_the_stated_rule():
    cases = yaml.safe_load((REPO_ROOT / "eval/golden_questions.yaml").read_text(encoding="utf-8"))["cases"]
    labels = {case["id"]: case["computable"] for case in cases}

    for case in cases:
        if case["category"] in ("lookup", "domain", "trap"):
            assert labels[case["id"]] is False
        if case["category"] == "reasoning":
            assert labels[case["id"]] is True
    # Code cases: Python and SQL snippets run in the standard library; JavaScript does not.
    assert {case_id for case_id, label in labels.items() if case_id.startswith("CD") and not label} == {
        "CD-05", "CD-07"
    }


# --- the accuracy script ---


def check_record(case_id, expected, verdict, latency_ms=100.0, category="reasoning"):
    return {
        "case_id": case_id, "category": category, "expected_computable": expected, "verdict": verdict,
        "latency_ms": latency_ms,
    }


def test_summarise_counts_precision_and_recall():
    records = [
        check_record("A", True, "yes", 100.0),
        check_record("B", True, "yes", 200.0),
        check_record("C", True, "no", 300.0),
        check_record("D", True, "low_confidence", 400.0),
        check_record("E", True, "failed", None),
        check_record("F", False, "yes", 500.0, "lookup"),
        check_record("G", False, "no", 600.0, "lookup"),
        check_record("H", False, "failed", 700.0, "domain"),
    ]

    summary = check_computation.summarise(records)

    assert (summary["true_positive"], summary["false_positive"]) == (2, 1)
    assert (summary["false_negative"], summary["true_negative"]) == (3, 2)
    assert summary["precision"] == pytest.approx(2 / 3)
    assert summary["recall"] == pytest.approx(2 / 5)
    assert summary["false_positive_cases"] == ["F"]
    assert summary["false_negative_cases"] == ["C", "D", "E"]
    assert summary["false_positive_by_category"] == {"lookup": 1}
    assert summary["verdicts"] == {"yes": 3, "no": 2, "low_confidence": 1, "failed": 2}
    assert summary["latency_median_ms"] == 400.0


def test_summarise_without_a_yes_has_no_precision():
    summary = check_computation.summarise([check_record("A", False, "no")])

    assert summary["precision"] is None
    assert summary["recall"] is None


def test_check_case_records_the_verdict_against_the_label():
    config = make_config()
    client = FakeClient()
    client.script("triage-model", check_json(True, 0.5, "arithmetic"), error_result("http_5xx"))
    clients = {"ollama": client}
    case = {"id": "RS-01", "question": "q", "category": "reasoning", "computable": True, "question_file": "f"}

    low = check_computation.check_case(config, clients, case)
    failed = check_computation.check_case(config, clients, case)

    assert (low["verdict"], low["computable"], low["confidence"], low["reason"]) == (
        "low_confidence", True, 0.5, "arithmetic"
    )
    assert low["expected_computable"] is True and low["latency_ms"] == 12.0
    assert (failed["verdict"], failed["status"], failed["error_type"]) == ("failed", "error", "http_5xx")


def test_load_labelled_cases_skips_cases_without_a_label(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(
        yaml.safe_dump({"cases": [{"id": "A", "computable": True}, {"id": "B"}, {"id": "C", "computable": False}]}),
        encoding="utf-8",
    )

    cases = check_computation.load_labelled_cases([path])

    assert [(case["id"], case["computable"], case["question_file"]) for case in cases] == [
        ("A", True, str(path)), ("C", False, str(path))
    ]
