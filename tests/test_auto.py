"""Auto mode: the decision table, the harness wiring and the CLI flags. Fakes only."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from inference_arbiter import auto, cli, openbook
from inference_arbiter.config import HarnessConfig
from inference_arbiter.router import OpenbookNotConfiguredError
from inference_arbiter.schemas import AskRequest

from conftest import (
    BASE_CONFIG,
    HarnessKit,
    error_result,
    execute_draft_json,
    make_config,
    make_execute_config,
    make_openbook_config,
    ok_result,
    program_output,
    triage_json,
)

QUESTION = "What confidence value triggers escalation?"
GUIDE_TEXT = "# Router Guide\n\nThe router escalates when confidence is below 0.80.\n"
QUOTE = "The router escalates when confidence is below 0.80."


# --- the decision table ---


@pytest.mark.parametrize(
    ("has_sources", "estimated", "window", "ratio", "require_citations", "mode", "reason"),
    [
        # row 1: no sources
        (False, 0, None, 0.8, False, "route", "auto: no sources, one closed-book call"),
        (False, 0, 1_000_000, 0.8, False, "route", "auto: no sources, one closed-book call"),
        # row 2: citations required wins over a fit
        (True, 100, 1_000_000, 0.8, True, "openbook", "auto: citations required"),
        (True, 100, None, 0.8, True, "openbook", "auto: citations required"),
        # row 3: unknown window is treated as not fitting
        (
            True,
            100,
            None,
            0.8,
            False,
            "openbook",
            "auto: the answering tier has no context_window_tokens, treated as not fitting",
        ),
        # row 4: fits, the limit itself included
        (True, 17_000, 1_048_576, 0.8, False, "stuffed", "auto: sources fit, 17k of 839k tokens"),
        (True, 800, 1000, 0.8, False, "stuffed", "auto: sources fit, 800 of 800 tokens"),
        # row 5: does not fit, the safety margin included
        (True, 801, 1000, 0.8, False, "openbook", "auto: sources exceed context window, 801 of 800 tokens"),
        (
            True,
            3_400_000,
            1_048_576,
            0.8,
            False,
            "openbook",
            "auto: sources exceed context window, 3.4M of 839k tokens",
        ),
        (True, 900, 1000, 1.0, False, "stuffed", "auto: sources fit, 900 of 1k tokens"),
    ],
)
def test_decision_table(has_sources, estimated, window, ratio, require_citations, mode, reason):
    decision = auto.decide_mode(has_sources, estimated, window, ratio, require_citations)

    assert (decision.mode, decision.reason) == (mode, reason)


VERDICTS = ["not_run", "yes", "no", "low_confidence", "failed"]


@pytest.mark.parametrize(
    ("computation", "mode", "reason"),
    [
        ("yes", "execute", "auto: no sources, the computation check said a program can compute the answer"),
        ("not_run", "route", "auto: no sources, one closed-book call"),
        ("no", "route", "auto: no sources, the computation check said no, one closed-book call"),
        (
            "low_confidence",
            "route",
            "auto: no sources, the computation check said yes below the confidence threshold, "
            "one closed-book call",
        ),
        ("failed", "route", "auto: no sources, the computation check failed, one closed-book call"),
    ],
)
def test_decision_table_without_sources_reads_the_computation_check(computation, mode, reason):
    decision = auto.decide_mode(False, 0, None, 0.8, False, computation)

    assert (decision.mode, decision.reason) == (mode, reason)


@pytest.mark.parametrize("computation", VERDICTS)
@pytest.mark.parametrize(
    ("estimated", "window", "require_citations", "mode"),
    [
        (100, 1_000_000, True, "openbook"),
        (100, None, False, "openbook"),
        (800, 1000, False, "stuffed"),
        (801, 1000, False, "openbook"),
    ],
)
def test_decision_table_with_sources_never_reads_the_computation_check(
    computation, estimated, window, require_citations, mode
):
    with_verdict = auto.decide_mode(True, estimated, window, 0.8, require_citations, computation)
    without = auto.decide_mode(True, estimated, window, 0.8, require_citations)

    assert with_verdict == without
    assert with_verdict.mode == mode


def test_only_a_yes_without_sources_reaches_execute():
    executing = [
        (has_sources, computation)
        for has_sources in (False, True)
        for computation in VERDICTS
        for window in (None, 1000)
        for require_citations in (False, True)
        if auto.decide_mode(has_sources, 100, window, 0.8, require_citations, computation).mode == "execute"
    ]

    assert set(executing) == {(False, "yes")}


def test_the_check_is_described_for_the_auto_reason():
    assert auto.describe_check("triage", "yes", 0.9) == "check on triage: yes, confidence 0.90"
    assert auto.describe_check("local", "failed", None) == "check on local: failed"
    assert auto.execute_fallback_reason("execute_no_answer") == (
        "auto: no sources, the computation check said a program can compute the answer; "
        "execution failed (execute_no_answer), one closed-book call, not verified"
    )


# --- request contract ---


def test_auto_is_the_default_mode_and_accepts_sources():
    assert AskRequest.from_question(QUESTION).mode == "auto"
    assert AskRequest.from_question(QUESTION, sources=["a.md"]).mode == "auto"


def test_require_citations_without_sources_is_refused():
    with pytest.raises(ValidationError, match="nothing to cite"):
        AskRequest.from_question(QUESTION, require_citations=True)


@pytest.mark.parametrize("mode", ["stuffed", "openbook"])
def test_require_citations_only_fits_auto_and_openbook(mode):
    if mode == "openbook":
        assert AskRequest.from_question(QUESTION, mode=mode, sources=["a.md"], require_citations=True)
        return
    with pytest.raises(ValidationError, match="require_citations needs mode"):
        AskRequest.from_question(QUESTION, mode=mode, sources=["a.md"], require_citations=True)


def test_triage_timeout_simulation_is_refused_in_auto_mode():
    with pytest.raises(ValidationError, match="needs a mode with a triage call"):
        AskRequest.from_question(QUESTION, simulate="triage-timeout")


def test_verifier_tier_in_auto_needs_sources():
    assert AskRequest.from_question(QUESTION, sources=["a.md"], verifier_tier="ground_chat")
    with pytest.raises(ValidationError, match="verifier_tier needs"):
        AskRequest.from_question(QUESTION, verifier_tier="ground_chat")


@pytest.mark.parametrize("value", [True, False])
def test_computation_check_only_fits_auto_without_sources_or_a_forced_tier(value):
    assert AskRequest.from_question(QUESTION, computation_check=value).computation_check is value
    for overrides in (
        {"mode": "route"},
        {"mode": "chain"},
        {"mode": "execute"},
        {"sources": ["a.md"]},
        {"mode": "stuffed", "sources": ["a.md"]},
        {"tier": "local"},
    ):
        with pytest.raises(ValidationError, match="computation_check needs mode 'auto'"):
            AskRequest.from_question(QUESTION, computation_check=value, **overrides)


# --- config ---


def test_closed_book_tier_defaults_to_the_fallback_tier():
    config = make_config()

    assert config.auto.closed_book_tier is None
    assert config.auto_closed_book_tier() == "mid"
    assert config.auto.stuffed_fill_ratio == 0.8


def test_closed_book_tier_must_be_an_answer_tier():
    with pytest.raises(ValidationError, match="auto.closed_book_tier 'triage' is not an answer tier"):
        make_config(auto={"closed_book_tier": "triage"})


def test_shipped_config_names_the_closed_book_tier():
    from inference_arbiter.config import load_config

    config = load_config("config/models.yaml")

    assert config.auto.closed_book_tier == "mid"
    assert config.tiers["mid"].context_window_tokens is not None


# --- harness: no sources ---


def test_no_sources_is_one_call_to_the_closed_book_tier_without_triage():
    kit = HarnessKit(make_config(auto={"closed_book_tier": "local"}))
    kit.client.script("local-model", ok_result("a nonce is used once"))

    result = kit.harness.answer(AskRequest.from_question("what is a nonce?"))

    assert kit.client.models_called() == ["local-model"]
    assert result.mode == "route"
    assert result.auto_reason == "auto: no sources, one closed-book call"
    assert result.triage_status == "skipped"
    assert (result.selected_tier, result.route_reason) == ("local", "auto")
    assert [(call.role, call.tier) for call in result.calls] == [("answer", "local")]
    assert result.answer == "a nonce is used once"
    assert kit.run_record.mode == "route"
    assert kit.run_record.auto_reason == result.auto_reason


def test_no_sources_keeps_the_on_error_fallback_hop():
    kit = HarnessKit(make_config())
    kit.client.script("mid-model", error_result("http_5xx"))
    kit.client.script("local-model", ok_result("from the fallback"))

    result = kit.harness.answer(AskRequest.from_question("what is a nonce?"))

    assert kit.client.models_called() == ["mid-model", "local-model"]
    assert result.fallback_used is True
    assert result.final_tier == "local"
    assert result.answer == "from the fallback"


def test_forced_tier_wins_over_the_closed_book_tier():
    kit = HarnessKit(make_config())
    kit.client.script("local-model", ok_result("forced"))

    result = kit.harness.answer(AskRequest.from_question("what is a nonce?", tier="local"))

    assert kit.client.models_called() == ["local-model"]
    assert (result.selected_tier, result.route_reason) == ("local", "forced")
    assert result.auto_reason == "auto: no sources, one closed-book call"


def test_explicit_route_mode_still_triages_and_reports_no_auto_reason():
    kit = HarnessKit(make_config())
    kit.client.script("triage-model", triage_json("local", 0.95))
    kit.client.script("local-model", ok_result("routed"))

    result = kit.harness.answer(AskRequest.from_question("what is a nonce?", mode="route"))

    assert kit.client.models_called() == ["triage-model", "local-model"]
    assert result.route_reason == "triage_label"
    assert result.auto_reason is None
    assert kit.run_record.auto_reason is None


# --- harness: the computation check ---

COMPUTE_QUESTION = "What is 6 times 7?"


def check_config(auto_overrides: dict | None = None, **execute_overrides):
    """The check on (asked of the triage tier); execute on mid with program B on code, one
    program after the other so scripted results line up."""
    auto_settings = {"computation_check": True}
    auto_settings.update(auto_overrides or {})
    execute_settings = {"cross_check_tier": "code", "cross_check_parallel": False}
    execute_settings.update(execute_overrides)
    return make_execute_config(execute_settings, auto=auto_settings)


def check_json(computable: bool, confidence: float, reason: str = "arithmetic"):
    return ok_result(json.dumps({"reason": reason, "computable": computable, "confidence": confidence}))


def program_draft(program: str = "print(42)"):
    return execute_draft_json(program, None)


def test_check_yes_runs_execute_with_the_cross_check():
    kit = HarnessKit(check_config())
    kit.client.script("triage-model", check_json(True, 0.95))
    kit.client.script("mid-model", program_draft())
    kit.client.script("code-model", program_draft("print(6 * 7)"))
    kit.sandbox_runner.script("run", program_output("42\n"), program_output("42\n"))

    result = kit.harness.answer(AskRequest.from_question(COMPUTE_QUESTION))

    assert kit.client.models_called() == ["triage-model", "mid-model", "code-model"]
    assert kit.client.calls[0].messages[1]["content"] == COMPUTE_QUESTION
    assert result.mode == "execute"
    assert (result.execution_status, result.cross_check, result.cross_check_tier) == ("verified", True, "code")
    assert result.answer.endswith("Answer: 42")
    assert (result.selected_tier, result.route_reason) == ("mid", "configured")
    assert result.triage_status == "skipped"
    assert result.auto_reason == (
        "auto: no sources, the computation check said a program can compute the answer "
        "[check on triage: yes, confidence 0.95]"
    )
    assert (result.computation_verdict, result.computation_check_tier) == ("yes", "triage")
    assert (result.computation_check_confidence, result.computation_check_reason) == (0.95, "arithmetic")
    assert [(call.role, call.tier) for call in result.calls] == [
        ("check", "triage"), ("draft", "mid"), ("draft_b", "code")
    ]
    assert [record.seq for record in kit.call_records] == [1, 2, 3]
    assert (kit.call_records[0].role, kit.call_records[0].status, kit.call_records[0].latency_ms) == (
        "check", "ok", 12.0
    )
    assert kit.run_record.mode == "execute"
    assert kit.run_record.auto_reason == result.auto_reason
    assert kit.run_record.execution_status == "verified"
    assert len(kit.sandbox_records) == 2


def test_check_yes_without_the_cross_check_runs_one_program():
    kit = HarnessKit(check_config({"computation_cross_check": False}, cross_check=True))
    kit.client.script("triage-model", check_json(True, 0.9))
    kit.client.script("mid-model", program_draft())
    kit.sandbox_runner.script("run", program_output("42\n"))

    result = kit.harness.answer(AskRequest.from_question(COMPUTE_QUESTION))

    assert kit.client.models_called() == ["triage-model", "mid-model"]
    assert (result.mode, result.execution_status, result.cross_check) == ("execute", "executed", False)


def test_auto_turns_the_cross_check_on_even_when_execute_has_it_off():
    config = check_config(cross_check=False)
    assert config.execute.cross_check is False
    kit = HarnessKit(config)
    kit.client.script("triage-model", check_json(True, 0.9))
    kit.client.script("mid-model", program_draft())
    kit.client.script("code-model", program_draft())
    kit.sandbox_runner.script("run", program_output("42\n"), program_output("42\n"))

    result = kit.harness.answer(AskRequest.from_question(COMPUTE_QUESTION))

    assert (result.execution_status, result.cross_check) == ("verified", True)


@pytest.mark.parametrize(
    ("check_result", "verdict", "reason_start", "confidence", "call_status"),
    [
        (check_json(False, 0.95, "a fact"), "no", auto.REASON_NOT_COMPUTABLE, 0.95, "ok"),
        (check_json(True, 0.79), "low_confidence", auto.REASON_CHECK_LOW_CONFIDENCE, 0.79, "ok"),
        (error_result("timeout"), "failed", auto.REASON_CHECK_FAILED, None, "timeout"),
        (error_result("http_5xx"), "failed", auto.REASON_CHECK_FAILED, None, "error"),
        (ok_result("not json"), "failed", auto.REASON_CHECK_FAILED, None, "invalid"),
        (ok_result(json.dumps({"computable": True})), "failed", auto.REASON_CHECK_FAILED, None, "invalid"),
    ],
)
def test_anything_but_a_confident_yes_is_the_closed_book_call(
    check_result, verdict, reason_start, confidence, call_status
):
    kit = HarnessKit(check_config())
    kit.client.script("triage-model", check_result)
    kit.client.script("mid-model", ok_result("a nonce is used once"))

    result = kit.harness.answer(AskRequest.from_question("what is a nonce?"))

    assert kit.client.models_called() == ["triage-model", "mid-model"]
    assert kit.sandbox_runner.commands == []
    assert result.mode == "route"
    assert result.answer == "a nonce is used once"
    assert (result.selected_tier, result.route_reason) == ("mid", "auto")
    assert result.execution_status == "skipped"
    assert result.computation_verdict == verdict
    assert result.computation_check_confidence == confidence
    assert result.auto_reason.startswith(reason_start + " [check on triage: " + verdict)
    assert [(call.role, call.status) for call in result.calls] == [("check", call_status), ("answer", "ok")]
    assert kit.run_record.auto_reason == result.auto_reason
    assert kit.run_record.mode == "route"


def test_the_threshold_comes_from_the_config():
    kit = HarnessKit(check_config({"computation_check_threshold": 0.5}))
    kit.client.script("triage-model", check_json(True, 0.5))
    kit.client.script("mid-model", program_draft())
    kit.client.script("code-model", program_draft())
    kit.sandbox_runner.script("run", program_output("42\n"), program_output("42\n"))

    result = kit.harness.answer(AskRequest.from_question(COMPUTE_QUESTION))

    assert (result.mode, result.computation_verdict) == ("execute", "yes")


def test_the_check_tier_comes_from_the_config():
    kit = HarnessKit(check_config({"computation_check_tier": "local"}))
    kit.client.script("local-model", check_json(False, 0.9))
    kit.client.script("mid-model", ok_result("closed book"))

    result = kit.harness.answer(AskRequest.from_question("what is a nonce?"))

    assert kit.client.models_called() == ["local-model", "mid-model"]
    assert result.computation_check_tier == "local"
    assert (result.calls[0].role, result.calls[0].tier) == ("check", "local")


def test_the_check_is_off_by_default_and_nothing_extra_is_called():
    kit = HarnessKit(check_config({"computation_check": False}))
    kit.client.script("mid-model", ok_result("a nonce is used once"))

    result = kit.harness.answer(AskRequest.from_question("what is a nonce?"))

    assert kit.client.models_called() == ["mid-model"]
    assert result.auto_reason == auto.REASON_NO_SOURCES
    assert (result.computation_verdict, result.computation_check_tier) == ("not_run", None)


def test_the_request_switches_the_check_on_and_off():
    off_in_config = HarnessKit(check_config({"computation_check": False}))
    off_in_config.client.script("triage-model", check_json(False, 0.9))
    off_in_config.client.script("mid-model", ok_result("closed book"))
    on_in_config = HarnessKit(check_config())
    on_in_config.client.script("mid-model", ok_result("closed book"))

    switched_on = off_in_config.harness.answer(AskRequest.from_question("q?", computation_check=True))
    switched_off = on_in_config.harness.answer(AskRequest.from_question("q?", computation_check=False))

    assert off_in_config.client.models_called() == ["triage-model", "mid-model"]
    assert switched_on.computation_verdict == "no"
    assert on_in_config.client.models_called() == ["mid-model"]
    assert (switched_off.computation_verdict, switched_off.auto_reason) == ("not_run", auto.REASON_NO_SOURCES)


def test_the_request_flag_without_an_execute_block_is_a_usage_error():
    from inference_arbiter.router import ExecuteNotConfiguredError

    kit = HarnessKit(make_config())

    with pytest.raises(ExecuteNotConfiguredError, match="computation check needs an `execute:` block"):
        kit.harness.answer(AskRequest.from_question("q?", computation_check=True))
    assert kit.client.models_called() == []


def fallback_kit() -> HarnessKit:
    kit = HarnessKit(check_config())
    kit.client.script("triage-model", check_json(True, 0.95))
    return kit


def assert_fell_back(kit: HarnessKit, result, closed_book_answer: str) -> None:
    assert result.status == "ok"
    assert result.mode == "route"
    assert result.execution_status == "failed"
    assert result.answer.startswith("NOT VERIFIED: a program was meant to compute this answer")
    assert result.answer.endswith(closed_book_answer)
    assert "NOT VERIFIED" in result.execution_note and "Neither program produced one" in result.execution_note
    assert result.auto_reason == (
        "auto: no sources, the computation check said a program can compute the answer; execution "
        "failed (execute_no_answer), one closed-book call, not verified [check on triage: yes, confidence 0.95]"
    )
    assert (result.selected_tier, result.route_reason, result.final_tier) == ("mid", "auto", "mid")
    assert result.calls[-1].role == "answer"
    assert kit.run_record.mode == "route"
    assert kit.run_record.execution_status == "failed"
    assert kit.run_record.auto_reason == result.auto_reason
    assert [record.seq for record in kit.call_records] == list(range(1, len(kit.call_records) + 1))


def test_sbx_missing_falls_back_to_the_closed_book_call_marked_not_verified():
    kit = fallback_kit()
    kit.client.script("mid-model", program_draft(), ok_result("It is 42."))
    kit.client.script("code-model", program_draft())
    kit.sandbox_runner.script("create", FileNotFoundError("sbx"), FileNotFoundError("sbx"))

    result = kit.harness.answer(AskRequest.from_question(COMPUTE_QUESTION))

    assert_fell_back(kit, result, "It is 42.")
    assert [(call.role, call.tier) for call in result.calls] == [
        ("check", "triage"), ("draft", "mid"), ("draft_b", "code"), ("answer", "mid")
    ]
    assert "sandbox_unavailable" in result.execution_note
    assert [run.outcome for run in result.sandbox_runs] == ["infra_failed", "infra_failed"]
    assert len(kit.sandbox_records) == 2


def test_both_drafts_failing_falls_back_to_the_closed_book_call():
    kit = fallback_kit()
    kit.client.script("mid-model", error_result("http_5xx"), ok_result("It is 42."))
    kit.client.script("code-model", ok_result("not a draft"))

    result = kit.harness.answer(AskRequest.from_question(COMPUTE_QUESTION))

    assert_fell_back(kit, result, "It is 42.")
    assert [(call.role, call.status) for call in result.calls] == [
        ("check", "ok"), ("draft", "error"), ("draft_b", "invalid"), ("answer", "ok")
    ]
    assert result.sandbox_runs == []


def test_both_programs_failing_after_repair_falls_back():
    kit = fallback_kit()
    kit.client.script("mid-model", program_draft("boom"), program_draft("boom"), ok_result("It is 42."))
    kit.client.script("code-model", program_draft("boom"), program_draft("boom"))
    failed_run = program_output("", "Traceback", exit_code=1)
    kit.sandbox_runner.script("run", failed_run, failed_run, failed_run, failed_run)

    result = kit.harness.answer(AskRequest.from_question(COMPUTE_QUESTION))

    assert_fell_back(kit, result, "It is 42.")
    assert [call.role for call in result.calls] == ["check", "draft", "repair", "draft_b", "repair_b", "answer"]
    assert len(result.sandbox_runs) == 4


def test_one_program_failing_outright_falls_back_too():
    kit = HarnessKit(check_config({"computation_cross_check": False}))
    kit.client.script("triage-model", check_json(True, 0.95))
    kit.client.script("mid-model", error_result("http_5xx"), ok_result("It is 42."))

    result = kit.harness.answer(AskRequest.from_question(COMPUTE_QUESTION))

    assert (result.status, result.mode, result.execution_status) == ("ok", "route", "failed")
    assert result.answer.startswith("NOT VERIFIED") and result.answer.endswith("It is 42.")
    assert result.cross_check is False


def test_a_fallback_whose_closed_book_call_fails_too_is_an_error_with_the_usual_hop():
    kit = fallback_kit()
    kit.client.script("mid-model", error_result("http_5xx"), error_result("http_5xx"))
    kit.client.script("code-model", error_result("http_5xx"))
    kit.client.script("local-model", error_result("timeout"))

    result = kit.harness.answer(AskRequest.from_question(COMPUTE_QUESTION))

    assert (result.status, result.answer, result.error_type) == ("error", None, "timeout")
    assert (result.mode, result.execution_status, result.fallback_used) == ("route", "failed", True)
    assert [(call.role, call.tier) for call in result.calls][-2:] == [("answer", "mid"), ("answer", "local")]


def test_disputed_is_returned_as_it_is():
    kit = fallback_kit()
    kit.client.script("mid-model", program_draft("print(41)"))
    kit.client.script("code-model", program_draft("print(42)"))
    kit.sandbox_runner.script("run", program_output("41\n"), program_output("42\n"))

    result = kit.harness.answer(AskRequest.from_question(COMPUTE_QUESTION))

    assert kit.client.models_called() == ["triage-model", "mid-model", "code-model"]
    assert (result.mode, result.status, result.execution_status) == ("execute", "ok", "disputed")
    assert result.answer.startswith("DISPUTED")
    assert not result.auto_reason.startswith(auto.execute_fallback_reason("execute_no_answer"))


def test_one_sided_is_returned_as_it_is():
    kit = fallback_kit()
    kit.client.script("mid-model", program_draft("print(42)"))
    kit.client.script("code-model", error_result("http_5xx"))
    kit.sandbox_runner.script("run", program_output("42\n"))

    result = kit.harness.answer(AskRequest.from_question(COMPUTE_QUESTION))

    assert kit.client.models_called() == ["triage-model", "mid-model", "code-model"]
    assert (result.mode, result.status, result.execution_status) == ("execute", "ok", "one_sided")
    assert result.answer.startswith("NOT CROSS-CHECKED")


@pytest.mark.parametrize(
    ("overrides", "scripted"),
    [
        ({"mode": "route"}, [("triage-model", triage_json("local", 0.95)), ("local-model", ok_result("x"))]),
        ({"tier": "local"}, [("local-model", ok_result("x"))]),
        ({"mode": "route", "tier": "local"}, [("local-model", ok_result("x"))]),
        (
            {"mode": "execute"},
            [("mid-model", execute_draft_json("print(42)", "42"))],
        ),
        (
            {"mode": "execute", "cross_check": True},
            [("mid-model", execute_draft_json("print(42)", None)), ("code-model", execute_draft_json("print(42)", None))],
        ),
    ],
)
def test_explicit_choices_never_run_the_check(overrides, scripted):
    kit = HarnessKit(check_config())
    for model, scripted_result in scripted:
        kit.client.script(model, scripted_result)
    kit.sandbox_runner.script("run", program_output("42\n"), program_output("42\n"))

    result = kit.harness.answer(AskRequest.from_question(COMPUTE_QUESTION, **overrides))

    assert "check" not in [call.role for call in result.calls]
    assert kit.client.models_called() == [model for model, _ in scripted]
    assert result.computation_verdict == "not_run"
    assert result.computation_check_tier is None


def test_chain_mode_never_runs_the_check():
    from conftest import CHAIN_VERIFIER_TIER, draft_json, verdict_json

    tiers = dict(BASE_CONFIG["tiers"])
    tiers["verify"] = dict(CHAIN_VERIFIER_TIER)
    config = make_execute_config(
        tiers=tiers, chain={"verifier_tier": "verify"}, auto={"computation_check": True}
    )
    kit = HarnessKit(config)
    kit.client.script("local-model", draft_json("42"))
    kit.client.script("verify-model", verdict_json("supported"))

    result = kit.harness.answer(AskRequest.from_question(COMPUTE_QUESTION, mode="chain", tier="local"))

    assert [call.role for call in result.calls] == ["draft", "verify"]
    assert result.computation_verdict == "not_run"


def test_sources_never_run_the_check(tmp_path, guide_path):
    config = windowed_config(tmp_path, 10_000)
    raw = config.model_dump(exclude={"sha256", "source_path"})
    raw["execute"] = {"tier": "mid"}
    raw["auto"] = {"computation_check": True}
    kit = HarnessKit(HarnessConfig.model_validate(raw))
    kit.client.script("mid-model", ok_result("Below 0.80."))

    result = kit.harness.answer(AskRequest.from_question(QUESTION, sources=[guide_path]))

    assert kit.client.models_called() == ["mid-model"]
    assert result.mode == "stuffed"
    assert result.computation_verdict == "not_run"
    assert "check" not in result.auto_reason


# --- harness: sources ---


def windowed_config(tmp_path, window: int | None, **openbook_overrides) -> HarnessConfig:
    config = make_openbook_config(tmp_path / "cache", openbook_overrides)
    config.tiers["mid"].context_window_tokens = window
    return config


@pytest.fixture
def guide_path(tmp_path) -> str:
    path = tmp_path / "guide.md"
    path.write_text(GUIDE_TEXT, encoding="utf-8")
    return str(path)


def cited_draft() -> object:
    claim = {"text": "Escalation is below 0.80.", "passage_id": "S1-P1", "quote": QUOTE}
    return ok_result(
        json.dumps({"working": "read", "answerable": True, "answer": "Below 0.80.", "claims": [claim]})
    )


def grounded_report() -> object:
    return ok_result(
        json.dumps(
            {"claims": [{"claim_number": 1, "reason": "stated", "verdict": "supported"}], "answerable": "yes"}
        )
    )


def test_sources_that_fit_go_stuffed_in_one_call(tmp_path, guide_path):
    kit = HarnessKit(windowed_config(tmp_path, 10_000))
    kit.client.script("mid-model", ok_result("Below 0.80."))

    result = kit.harness.answer(AskRequest.from_question(QUESTION, sources=[guide_path]))

    assert kit.client.models_called() == ["mid-model"]
    assert result.mode == "stuffed"
    assert result.auto_reason.startswith("auto: sources fit, ")
    assert result.auto_reason.endswith(" of 8k tokens")
    assert (result.selected_tier, result.route_reason) == ("mid", "auto")
    assert result.triage_status == "skipped"
    assert result.citations == []
    assert GUIDE_TEXT.strip() in kit.client.calls[0].messages[1]["content"]
    assert kit.run_record.mode == "stuffed"
    assert kit.run_record.auto_reason == result.auto_reason


def test_sources_that_do_not_fit_go_through_the_pipeline(tmp_path, guide_path):
    kit = HarnessKit(windowed_config(tmp_path, 20))
    kit.client.script("mid-model", cited_draft())
    kit.client.script("ground-model", grounded_report())

    result = kit.harness.answer(AskRequest.from_question(QUESTION, sources=[guide_path]))

    assert kit.client.models_called() == ["mid-model", "ground-model"]
    assert result.mode == "openbook"
    assert result.auto_reason.startswith("auto: sources exceed context window, ")
    assert result.route_reason == "configured"
    assert result.grounding_status == "grounded"
    assert [citation.passage_id for citation in result.citations] == ["S1-P1"]


def test_unknown_context_window_is_treated_as_not_fitting(tmp_path, guide_path):
    kit = HarnessKit(windowed_config(tmp_path, None))
    kit.client.script("mid-model", cited_draft())
    kit.client.script("ground-model", grounded_report())

    result = kit.harness.answer(AskRequest.from_question(QUESTION, sources=[guide_path]))

    assert result.mode == "openbook"
    assert result.auto_reason == auto.REASON_NO_WINDOW


def test_require_citations_takes_the_pipeline_even_when_the_sources_fit(tmp_path, guide_path):
    kit = HarnessKit(windowed_config(tmp_path, 10_000))
    kit.client.script("mid-model", cited_draft())
    kit.client.script("ground-model", grounded_report())

    result = kit.harness.answer(
        AskRequest.from_question(QUESTION, sources=[guide_path], require_citations=True)
    )

    assert result.mode == "openbook"
    assert result.auto_reason == "auto: citations required"
    assert len(result.citations) == 1


def test_the_window_checked_is_the_forced_tiers(tmp_path, guide_path):
    config = windowed_config(tmp_path, 20)
    config.tiers["local"].context_window_tokens = 10_000
    kit = HarnessKit(config)
    kit.client.script("local-model", ok_result("Below 0.80."))

    result = kit.harness.answer(AskRequest.from_question(QUESTION, sources=[guide_path], tier="local"))

    assert kit.client.models_called() == ["local-model"]
    assert result.mode == "stuffed"
    assert result.route_reason == "forced"


def test_sources_are_loaded_once_and_reused(tmp_path, guide_path, monkeypatch):
    load_calls = []
    real_load_sources = openbook.load_sources

    def counting_load_sources(sources, **settings):
        load_calls.append(list(sources))
        return real_load_sources(sources, **settings)

    monkeypatch.setattr(openbook, "load_sources", counting_load_sources)
    stuffed_kit = HarnessKit(windowed_config(tmp_path, 10_000))
    stuffed_kit.client.script("mid-model", ok_result("Below 0.80."))
    pipeline_kit = HarnessKit(windowed_config(tmp_path, 20))
    pipeline_kit.client.script("mid-model", cited_draft())
    pipeline_kit.client.script("ground-model", grounded_report())

    stuffed_kit.harness.answer(AskRequest.from_question(QUESTION, sources=[guide_path]))
    pipeline_kit.harness.answer(AskRequest.from_question(QUESTION, sources=[guide_path]))

    assert load_calls == [[guide_path], [guide_path]]


def test_a_source_that_cannot_be_loaded_fails_the_run_before_any_model_call(tmp_path):
    kit = HarnessKit(windowed_config(tmp_path, 10_000))

    result = kit.harness.answer(AskRequest.from_question(QUESTION, sources=[str(tmp_path / "absent.md")]))

    assert kit.client.models_called() == []
    assert result.status == "error"
    assert result.error_type == "source_load_failed"
    assert result.auto_reason == auto.REASON_LOAD_FAILED


def test_sources_without_an_openbook_block_are_a_usage_error():
    kit = HarnessKit(HarnessConfig.model_validate(BASE_CONFIG))

    with pytest.raises(OpenbookNotConfiguredError, match="openbook"):
        kit.harness.answer(AskRequest.from_question(QUESTION, sources=["a.md"]))


def test_explicit_openbook_and_stuffed_modes_report_no_auto_reason(tmp_path, guide_path):
    kit = HarnessKit(windowed_config(tmp_path, 10_000))
    kit.client.script("mid-model", cited_draft())
    kit.client.script("ground-model", grounded_report())
    kit.client.script("mid-model", ok_result("stuffed"))

    pipeline = kit.harness.answer(AskRequest.from_question(QUESTION, mode="openbook", sources=[guide_path]))
    stuffed = kit.harness.answer(
        AskRequest.from_question(QUESTION, mode="stuffed", sources=[guide_path], tier="mid")
    )

    assert (pipeline.mode, pipeline.auto_reason) == ("openbook", None)
    assert (stuffed.mode, stuffed.auto_reason) == ("stuffed", None)


# --- CLI ---


def run_cli(monkeypatch, kit: HarnessKit, arguments: list[str]) -> int:
    monkeypatch.setattr(cli, "build_harness", lambda *args, **kwargs: kit.harness)
    return cli.main(arguments)


def test_cli_default_is_auto_and_the_trace_shows_the_choice(monkeypatch, capsys):
    kit = HarnessKit(make_config())
    kit.client.script("mid-model", ok_result("a nonce is used once"))

    exit_code = run_cli(monkeypatch, kit, ["ask", "what is a nonce?"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "a nonce is used once"
    assert "mode          route (auto: no sources, one closed-book call)" in captured.err
    assert "route         mid (reason: auto)" in captured.err
    assert "triage        skipped" in captured.err


def test_cli_route_flag_triages(monkeypatch, capsys):
    kit = HarnessKit(make_config())
    kit.client.script("triage-model", triage_json("local", 0.95))
    kit.client.script("local-model", ok_result("routed"))

    exit_code = run_cli(monkeypatch, kit, ["ask", "what is a nonce?", "--route"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert kit.client.models_called() == ["triage-model", "local-model"]
    assert "mode          route\n" in captured.err
    assert "auto:" not in captured.err


@pytest.mark.parametrize("extra", [["--triage-tier", "triage"], ["--simulate", "triage-timeout"]])
def test_cli_triage_flags_imply_route_mode(monkeypatch, capsys, extra):
    kit = HarnessKit(make_config())
    kit.client.script("triage-model", triage_json("local", 0.95))
    kit.client.script("local-model", ok_result("routed"))
    kit.client.script("mid-model", ok_result("after the triage timeout"))

    exit_code = run_cli(monkeypatch, kit, ["ask", "what is a nonce?", *extra])

    assert exit_code == 0
    assert kit.run_record.auto_reason is None
    assert kit.run_record.route_reason in ("triage_label", "triage_timeout")


def test_cli_tier_flag_alone_is_one_forced_call(monkeypatch, capsys):
    kit = HarnessKit(make_config())
    kit.client.script("local-model", ok_result("forced"))

    exit_code = run_cli(monkeypatch, kit, ["ask", "what is a nonce?", "--tier", "local"])

    assert exit_code == 0
    assert kit.client.models_called() == ["local-model"]
    assert kit.run_record.route_reason == "forced"


def test_cli_sources_go_stuffed_when_they_fit(monkeypatch, capsys, tmp_path, guide_path):
    kit = HarnessKit(windowed_config(tmp_path, 10_000))
    kit.client.script("mid-model", ok_result("Below 0.80."))

    exit_code = run_cli(monkeypatch, kit, ["ask", QUESTION, "--sources", guide_path])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "Below 0.80."
    assert "mode          stuffed (auto: sources fit, " in captured.err


def test_cli_require_citations_runs_the_pipeline(monkeypatch, capsys, tmp_path, guide_path):
    kit = HarnessKit(windowed_config(tmp_path, 10_000))
    kit.client.script("mid-model", cited_draft())
    kit.client.script("ground-model", grounded_report())

    exit_code = run_cli(
        monkeypatch,
        kit,
        ["ask", QUESTION, "--sources", guide_path, "--require-citations", "--verifier-tier", "ground_chat"],
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "mode          openbook (auto: citations required)" in captured.err
    assert "Citations:" in captured.out


def test_cli_pipeline_only_simulation_selects_the_pipeline(monkeypatch, capsys, tmp_path, guide_path):
    kit = HarnessKit(windowed_config(tmp_path, 10_000))
    kit.client.script("mid-model", cited_draft())

    exit_code = run_cli(
        monkeypatch, kit, ["ask", QUESTION, "--sources", guide_path, "--simulate", "grounding-error"]
    )

    assert exit_code == 0
    assert kit.run_record.mode == "openbook"
    assert kit.run_record.auto_reason is None


def test_cli_usage_errors_for_the_new_flags(monkeypatch, capsys, tmp_path, guide_path):
    kit = HarnessKit(windowed_config(tmp_path, 10_000))

    assert run_cli(monkeypatch, kit, ["ask", QUESTION, "--require-citations"]) == 2
    assert "nothing to cite" in capsys.readouterr().err

    assert run_cli(monkeypatch, kit, ["ask", QUESTION, "--route", "--sources", guide_path]) == 2
    assert "--route cannot be combined" in capsys.readouterr().err

    assert run_cli(monkeypatch, kit, ["ask", QUESTION, "--route", "--chain"]) == 2
    assert "--route cannot be combined" in capsys.readouterr().err

    assert (
        run_cli(monkeypatch, kit, ["ask", QUESTION, "--sources", guide_path, "--stuffed", "--require-citations"])
        == 2
    )
    assert "require_citations needs mode" in capsys.readouterr().err


def test_cli_computation_check_flag_sends_a_computable_question_to_execute(monkeypatch, capsys):
    kit = HarnessKit(check_config({"computation_check": False}))
    kit.client.script("triage-model", check_json(True, 0.95, "simple arithmetic"))
    kit.client.script("mid-model", program_draft())
    kit.client.script("code-model", program_draft())
    kit.sandbox_runner.script("run", program_output("42\n"), program_output("42\n"))

    exit_code = run_cli(monkeypatch, kit, ["ask", COMPUTE_QUESTION, "--computation-check"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip().endswith("Answer: 42")
    assert "check         yes on triage (confidence 0.95) - simple arithmetic" in captured.err
    assert "mode          execute (auto: no sources, the computation check said a program can compute" in captured.err
    assert "execution     verified" in captured.err
    assert "call 1        check    triage" in captured.err


def test_cli_no_computation_check_flag_wins_over_the_config(monkeypatch, capsys):
    kit = HarnessKit(check_config())
    kit.client.script("mid-model", ok_result("a nonce is used once"))

    exit_code = run_cli(monkeypatch, kit, ["ask", "what is a nonce?", "--no-computation-check"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert kit.client.models_called() == ["mid-model"]
    assert "check   " not in captured.err
    assert "mode          route (auto: no sources, one closed-book call)" in captured.err


def test_cli_trace_shows_a_fallback_as_not_verified(monkeypatch, capsys):
    kit = fallback_kit()
    kit.client.script("mid-model", error_result("http_5xx"), ok_result("It is 42."))
    kit.client.script("code-model", error_result("http_5xx"))

    exit_code = run_cli(monkeypatch, kit, ["ask", COMPUTE_QUESTION])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("NOT VERIFIED: ")
    assert captured.out.strip().endswith("It is 42.")
    assert "execution failed (execute_no_answer), one closed-book call, not verified" in captured.err
    assert "execution     failed; the answer is one closed-book call, NOT VERIFIED" in captured.err


def test_cli_check_flags_are_usage_errors_with_explicit_choices(monkeypatch, capsys, tmp_path, guide_path):
    kit = HarnessKit(check_config())

    assert run_cli(monkeypatch, kit, ["ask", QUESTION, "--computation-check", "--no-computation-check"]) == 2
    assert "cannot be combined" in capsys.readouterr().err

    for extra in (["--route"], ["--chain"], ["--execute"], ["--cross-check"], ["--tier", "local"],
                  ["--sources", guide_path], ["--sources", guide_path, "--stuffed"]):
        for flag in ("--computation-check", "--no-computation-check"):
            assert run_cli(monkeypatch, kit, ["ask", QUESTION, flag, *extra]) == 2
            assert "computation_check needs mode 'auto'" in capsys.readouterr().err
    assert kit.client.models_called() == []
