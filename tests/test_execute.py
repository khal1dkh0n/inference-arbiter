"""Execute mode: the comparison, the decision table and the control flow. Fakes only."""

from __future__ import annotations

import itertools
import json

import pytest
from pydantic import ValidationError

from inference_arbiter import cli, execute, sandbox
from inference_arbiter.config import HarnessConfig
from inference_arbiter.router import ExecuteNotConfiguredError
from inference_arbiter.sandbox import CommandResult
from inference_arbiter.schemas import AskRequest

from conftest import (
    BASE_CONFIG,
    HarnessKit,
    error_result,
    execute_draft_json,
    make_config,
    make_execute_config,
    ok_result,
    program_output,
)

MID = "mid-model"


@pytest.fixture
def execute_kit() -> HarnessKit:
    return HarnessKit(make_execute_config())


def ask(kit: HarnessKit, question: str = "What is 6 times 7?", **overrides):
    return kit.harness.answer(AskRequest.from_question(question, mode="execute", **overrides))


def sandbox_result(error_type: str | None = None, stdout: str = "") -> sandbox.SandboxResult:
    return sandbox.SandboxResult(
        status="ok" if error_type is None else "error", error_type=error_type, error_message=None,
        runner="python", sandbox_name="inference-arbiter-test", exit_code=0 if error_type is None else 1,
        stdout=stdout, stderr="", stdout_truncated=False, stderr_truncated=False, timed_out=False,
        duration_ms=1.0, total_ms=2.0, network_policy="deny_all", environment_names=("PATH",),
        sandbox_removed=True,
    )


# ---- agreement is decided in code ----


@pytest.mark.parametrize(
    "claimed, executed",
    [
        ("42", "42"),
        ("42", " 42 "),
        ("Answer: 42", "42"),
        ("**42**", "42"),
        ("`42`", "42"),
        ('"hello world"', "Hello   World"),
        ("Paris.", "paris"),
        ("42", "42.0"),
        ("42.0", "42"),
        ("1,234,567", "1234567"),
        ("0.3", "0.30000000000000004"),
        ("24 legs", "24"),
        ("$1,250.50", "1250.5"),
        ("15%", "15"),
        ("-7", "-7.0"),
        ("1e3", "1000"),
        ("[1, 2, 3]", "[1, 2, 3]"),
        ("\uff46\uff55\uff4c\uff4c\uff57\uff49\uff44\uff54\uff48", "fullwidth"),
    ],
)
def test_answers_that_agree(claimed, executed):
    assert execute.answers_agree(claimed, executed)


@pytest.mark.parametrize(
    "claimed, executed",
    [
        ("42", "43"),
        ("42", "42.001"),
        ("3.14", "3.14159"),
        ("24 legs", "24 arms"),
        ("5 apples", "5 oranges"),
        ("yes", "no"),
        ("the answer is 42", "it prints 42"),
        ("[1, 2, 3]", "[1, 2, 4]"),
        ("1 2", "1"),
        ("42", "4 2"),
        ("42", "error"),
        ("true", "1"),
        ("", "42"),
        ("1000", "1001"),
        ("1e9", "1000000002"),
    ],
)
def test_answers_that_disagree(claimed, executed):
    assert not execute.answers_agree(claimed, executed)


def test_the_executed_answer_is_the_last_non_empty_line():
    assert execute.executed_answer("debug\n\n  42  \n\n") == "42"
    assert execute.executed_answer("42") == "42"
    assert execute.executed_answer("") is None
    assert execute.executed_answer(" \n\t\n") is None


@pytest.mark.parametrize(
    "result, claimed, expected",
    [
        (sandbox_result(stdout="42\n"), "42", ("agree", None)),
        (sandbox_result(stdout="working...\n42\n"), "42", ("agree", None)),
        (sandbox_result(stdout="43\n"), "42", ("disagree", None)),
        (sandbox_result(stdout="42\n"), None, ("no_claim", None)),
        (sandbox_result(stdout=""), None, ("program_failed", "no_output")),
        (sandbox_result("timeout"), None, ("program_failed", "timeout")),
        (sandbox_result("sandbox_unavailable"), None, ("infra_failed", "sandbox_unavailable")),
        (sandbox_result(stdout=""), "42", ("program_failed", "no_output")),
        (sandbox_result(stdout="  \n"), "42", ("program_failed", "no_output")),
        (sandbox_result("timeout"), "42", ("program_failed", "timeout")),
        (sandbox_result("nonzero_exit", stdout="42\n"), "42", ("program_failed", "nonzero_exit")),
        (sandbox_result("output_too_large", stdout="42\n"), "42", ("program_failed", "output_too_large")),
        (sandbox_result("sandbox_unavailable"), "42", ("infra_failed", "sandbox_unavailable")),
        (sandbox_result("network_policy_not_applied"), "42", ("infra_failed", "network_policy_not_applied")),
        (sandbox_result("runner_not_allowed"), "42", ("infra_failed", "runner_not_allowed")),
        (sandbox_result("program_too_large"), "42", ("infra_failed", "program_too_large")),
        (sandbox_result("some_future_error"), "42", ("infra_failed", "some_future_error")),
    ],
)
def test_classify_attempt(result, claimed, expected):
    assert execute.classify_attempt(result, claimed) == expected


# ---- decide: the full truth table, every combination ----

FIRST_CLASSES = ("agree", "disagree", "no_claim", "program_failed", "infra_failed")
REPAIR_STATES = (
    "not_attempted", "agree", "disagree", "no_claim", "program_failed", "infra_failed", "draft_failed"
)
RETURN = "return"
FAILED = (RETURN, "failed", None)
ALL_COMBINATIONS = list(
    itertools.product(FIRST_CLASSES, REPAIR_STATES, (True, False), (True, False), (True, False))
)


def expected_decision(
    first: str, repair: str, repairs_allowed: bool, first_has_claim: bool, repair_has_claim: bool
) -> tuple:
    """The table written out by hand, independently of execute.decide."""
    first_claim = (RETURN, "unverified", "claim_first") if first_has_claim else FAILED
    if repair_has_claim:
        latest_claim = (RETURN, "unverified", "claim_repair")
    else:
        latest_claim = first_claim
    if first == "agree":
        return (RETURN, "claim_matched", "executed_first")
    if first == "disagree":
        return (RETURN, "corrected", "executed_first")
    if first == "no_claim":
        return (RETURN, "executed", "executed_first")
    if first == "infra_failed":
        return first_claim
    return {
        ("not_attempted", True): ("repair", None, None),
        ("not_attempted", False): first_claim,
        ("agree", True): (RETURN, "claim_matched", "executed_repair"),
        ("agree", False): (RETURN, "claim_matched", "executed_repair"),
        ("disagree", True): (RETURN, "corrected", "executed_repair"),
        ("disagree", False): (RETURN, "corrected", "executed_repair"),
        ("no_claim", True): (RETURN, "executed", "executed_repair"),
        ("no_claim", False): (RETURN, "executed", "executed_repair"),
        ("program_failed", True): latest_claim,
        ("program_failed", False): latest_claim,
        ("infra_failed", True): latest_claim,
        ("infra_failed", False): latest_claim,
        ("draft_failed", True): first_claim,
        ("draft_failed", False): first_claim,
    }[(repair, repairs_allowed)]


@pytest.mark.parametrize("first, repair, repairs_allowed, first_has_claim, repair_has_claim", ALL_COMBINATIONS)
def test_decide_truth_table(first, repair, repairs_allowed, first_has_claim, repair_has_claim):
    decision = execute.decide(first, repair, repairs_allowed, first_has_claim, repair_has_claim)
    assert (decision.action, decision.status, decision.answer_from) == expected_decision(
        first, repair, repairs_allowed, first_has_claim, repair_has_claim
    )


def test_decide_table_covers_all_280_combinations():
    assert len(ALL_COMBINATIONS) == 5 * 7 * 2 * 2 * 2 == 280


def test_decide_defaults_to_both_claims_present():
    for first, repair, allowed in itertools.product(FIRST_CLASSES, REPAIR_STATES, (True, False)):
        assert execute.decide(first, repair, allowed) == execute.decide(first, repair, allowed, True, True)


def test_only_a_failed_program_is_ever_repaired():
    for first, repair, allowed, first_claim, repair_claim in ALL_COMBINATIONS:
        if execute.decide(first, repair, allowed, first_claim, repair_claim).action == "repair":
            assert (first, repair, allowed) == ("program_failed", "not_attempted", True)


def test_each_status_returns_from_the_right_place():
    for combination in ALL_COMBINATIONS:
        decision = execute.decide(*combination)
        if decision.status in ("claim_matched", "corrected", "executed"):
            assert decision.answer_from.startswith("executed_")
        if decision.status == "unverified":
            assert decision.answer_from.startswith("claim_")
        if decision.status == "failed":
            assert decision.answer_from is None


def test_a_run_with_no_claim_is_never_claim_matched_or_corrected():
    """claim_matched and corrected keep their meaning: both need a claim that was compared."""
    for first, repair, allowed, first_claim, repair_claim in ALL_COMBINATIONS:
        decision = execute.decide(first, repair, allowed, first_claim, repair_claim)
        if decision.status in ("claim_matched", "corrected"):
            compared = first if decision.answer_from == "executed_first" else repair
            assert compared in ("agree", "disagree")
        if decision.status == "executed":
            compared = first if decision.answer_from == "executed_first" else repair
            assert compared == "no_claim"


def test_an_unverified_answer_is_always_a_claim_that_exists():
    for first, repair, allowed, first_claim, repair_claim in ALL_COMBINATIONS:
        decision = execute.decide(first, repair, allowed, first_claim, repair_claim)
        if decision.answer_from == "claim_first":
            assert first_claim
        if decision.answer_from == "claim_repair":
            assert repair_claim


# ---- control flow through the harness ----


def test_claim_matched_when_the_output_agrees_with_the_claim(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("print(6 * 7)", "42"))
    execute_kit.sandbox_runner.script("run", program_output("42\n"))

    result = ask(execute_kit)

    assert (result.status, result.mode, result.execution_status) == ("ok", "execute", "claim_matched")
    assert result.answer.splitlines()[-1] == "Answer: 42"
    assert result.answer.startswith("Matches the model's claim")
    assert (result.claimed_answer, result.executed_answer, result.repaired) == ("42", "42", False)
    assert (result.selected_tier, result.final_tier, result.route_reason) == ("mid", "mid", "configured")
    assert result.triage_status == "skipped"
    assert [call.role for call in result.calls] == ["draft"]
    assert execute_kit.sandbox_runner.stdin_of("stage") == "print(6 * 7)"
    assert len(result.sandbox_runs) == 1
    assert result.sandbox_runs[0].outcome == "agree" and result.sandbox_runs[0].sandbox_removed


def test_the_model_is_asked_for_a_schema_constrained_program(execute_kit):
    execute_kit.client.script(MID, execute_draft_json())
    execute_kit.sandbox_runner.script("run", program_output("42\n"))

    ask(execute_kit)

    call = execute_kit.client.calls[0]
    assert call.json_schema["required"] == ["working", "program", "claimed_answer"]
    assert call.messages[0]["role"] == "system" and "Standard library only" in call.messages[0]["content"]
    assert call.messages[-1] == {"role": "user", "content": "What is 6 times 7?"}


def test_corrected_when_the_output_disagrees_and_the_executed_output_wins(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("print(17 * 23)", "381"))
    execute_kit.sandbox_runner.script("run", program_output("391\n"))

    result = ask(execute_kit)

    assert (result.status, result.execution_status) == ("ok", "corrected")
    assert result.answer.splitlines()[-1] == "Answer: 391"
    assert 'the model claimed "381"' in result.answer
    assert (result.claimed_answer, result.executed_answer) == ("381", "391")
    assert execute_kit.client.models_called() == [MID], "a disagreement is settled by the output, not by another call"
    assert execute_kit.sandbox_runner.verbs().count("create") == 1


def test_a_failed_program_gets_one_repair_with_the_stderr_fed_back(execute_kit):
    execute_kit.client.script(
        MID, execute_draft_json("print(1/0)", "42"), execute_draft_json("print(6 * 7)", "42")
    )
    execute_kit.sandbox_runner.script(
        "run",
        program_output("", "Traceback (most recent call last):\nZeroDivisionError: division by zero\n", exit_code=1),
        program_output("42\n"),
    )

    result = ask(execute_kit)

    assert (result.status, result.execution_status, result.repaired) == ("ok", "claim_matched", True)
    assert result.answer.splitlines()[-1] == "Answer: 42"
    assert "repaired program" in result.answer
    assert [call.role for call in result.calls] == ["draft", "repair"]
    repair_prompt = execute_kit.client.calls[1].messages[-1]["content"]
    assert "ZeroDivisionError: division by zero" in repair_prompt
    assert "print(1/0)" in repair_prompt and "What is 6 times 7?" in repair_prompt
    assert "exited with code 1" in repair_prompt
    assert [run.attempt for run in result.sandbox_runs] == ["first", "repair"]
    assert [run.outcome for run in result.sandbox_runs] == ["program_failed", "agree"]
    assert len({run.sandbox_name for run in result.sandbox_runs}) == 2, "each program gets a fresh sandbox"
    assert execute_kit.sandbox_runner.existing == set()


def test_a_repair_that_disagrees_is_corrected_by_its_own_output(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("boom", "42"), execute_draft_json("print(41)", "42"))
    execute_kit.sandbox_runner.script("run", program_output("", "SyntaxError", exit_code=1), program_output("41\n"))

    result = ask(execute_kit)

    assert (result.execution_status, result.repaired) == ("corrected", True)
    assert result.answer.splitlines()[-1] == "Answer: 41"


def test_unverified_when_the_repair_fails_too_and_there_is_no_third_attempt(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("bad one", "41"), execute_draft_json("bad two", "42"))
    execute_kit.sandbox_runner.script(
        "run", program_output("", "NameError", exit_code=1), program_output("", "", exit_code=124)
    )

    result = ask(execute_kit)

    assert (result.status, result.execution_status, result.repaired) == ("ok", "unverified", False)
    assert result.answer.startswith("UNVERIFIED")
    assert result.answer.splitlines()[-1] == "Answer: 42", "the latest claim is returned, marked unverified"
    assert "nonzero_exit" in result.answer and "timeout" in result.answer
    assert result.executed_answer is None
    assert execute_kit.client.models_called() == [MID, MID]
    assert execute_kit.sandbox_runner.verbs().count("run") == 2
    assert [run.outcome for run in result.sandbox_runs] == ["program_failed", "program_failed"]


def test_a_timeout_is_repaired_and_the_prompt_says_it_timed_out(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("while True: pass", "42"), execute_draft_json())
    execute_kit.sandbox_runner.script(
        "run", CommandResult(exit_code=None, stdout="", stderr="", timed_out=True), program_output("42\n")
    )

    result = ask(execute_kit)

    assert result.execution_status == "claim_matched" and result.repaired
    assert result.sandbox_runs[0].timed_out and result.sandbox_runs[0].error_type == "timeout"
    assert "did not finish within the time limit" in execute_kit.client.calls[1].messages[-1]["content"]


def test_a_program_that_prints_nothing_is_a_failed_program(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("x = 42", "42"), execute_draft_json())
    execute_kit.sandbox_runner.script("run", program_output(""), program_output("42\n"))

    result = ask(execute_kit)

    assert result.execution_status == "claim_matched" and result.repaired
    assert "printed nothing" in execute_kit.client.calls[1].messages[-1]["content"]


def test_sandbox_unavailable_returns_the_claim_unverified_without_a_repair_call(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("print(6 * 7)", "42"))
    execute_kit.sandbox_runner.script("create", CommandResult(exit_code=1, stdout="", stderr="daemon down"))

    result = ask(execute_kit)

    assert (result.status, result.execution_status) == ("ok", "unverified")
    assert result.answer.startswith("UNVERIFIED") and "sandbox_unavailable" in result.answer
    assert result.answer.splitlines()[-1] == "Answer: 42"
    assert execute_kit.client.models_called() == [MID], "the program was not at fault, so it is not rewritten"
    assert result.sandbox_runs[0].outcome == "infra_failed"


def test_a_network_policy_that_was_not_confirmed_never_runs_the_program(execute_kit):
    execute_kit.client.script(MID, execute_draft_json())
    execute_kit.sandbox_runner.script("check", CommandResult(exit_code=0, stdout='{"allowed": true}', stderr=""))

    result = ask(execute_kit)

    assert result.execution_status == "unverified"
    assert "run" not in execute_kit.sandbox_runner.verbs()
    assert result.sandbox_runs[0].error_type == "network_policy_not_applied"


def test_no_repair_when_max_repairs_is_zero():
    kit = HarnessKit(make_execute_config({"max_repairs": 0}))
    kit.client.script(MID, execute_draft_json("print(1/0)", "42"))
    kit.sandbox_runner.script("run", program_output("", "ZeroDivisionError", exit_code=1))

    result = ask(kit)

    assert result.execution_status == "unverified"
    assert kit.client.models_called() == [MID]


def test_a_failed_repair_call_returns_the_first_claim_unverified(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("print(1/0)", "42"), error_result("http_5xx"))
    execute_kit.sandbox_runner.script("run", program_output("", "ZeroDivisionError", exit_code=1))

    result = ask(execute_kit)

    assert (result.status, result.execution_status) == ("ok", "unverified")
    assert result.error_type is None
    assert result.repair_error_type == "http_5xx"
    assert result.answer.splitlines()[-1] == "Answer: 42"
    assert "the repair call failed (http_5xx)" in result.answer
    assert execute_kit.sandbox_runner.verbs().count("run") == 1


def test_an_invalid_repair_draft_returns_the_first_claim_unverified(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("print(1/0)", "42"), ok_result("not json"))
    execute_kit.sandbox_runner.script("run", program_output("", "ZeroDivisionError", exit_code=1))

    result = ask(execute_kit)

    assert result.execution_status == "unverified"
    assert result.repair_error_type == "draft_schema_invalid"
    assert [call.status for call in result.calls] == ["ok", "invalid"]


def test_the_repair_never_hops_to_another_tier(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("print(1/0)", "42"), error_result("timeout"))
    execute_kit.sandbox_runner.script("run", program_output("", "ZeroDivisionError", exit_code=1))

    result = ask(execute_kit)

    assert set(execute_kit.client.models_called()) == {MID}
    assert result.fallback_used is False and result.final_tier == "mid"


@pytest.mark.parametrize(
    "draft, error_type",
    [
        (ok_result("I think the answer is 42"), "draft_schema_invalid"),
        (ok_result(json.dumps({"program": "", "claimed_answer": "42"})), "draft_schema_invalid"),
        (ok_result(json.dumps({"program": "print(42)", "claimed_answer": 42})), "draft_schema_invalid"),
        (ok_result(json.dumps({"program": "print(42)", "claimed_answer": "42", "extra": 1})), "draft_schema_invalid"),
        (error_result("http_5xx"), "http_5xx"),
    ],
)
def test_a_first_draft_that_fails_fails_the_run_and_nothing_is_executed(execute_kit, draft, error_type):
    execute_kit.client.script(MID, draft)

    result = ask(execute_kit)

    assert (result.status, result.answer, result.error_type) == ("error", None, error_type)
    assert result.execution_status == "skipped"
    assert execute_kit.sandbox_runner.commands == []
    assert execute_kit.client.models_called() == [MID], "no fallback hop and no second draft"


# ---- the explanation the program prints, and the single answer label ----


@pytest.mark.parametrize(
    "stdout, expected",
    [
        ("42\n", None),
        ("\n42\n\n", None),
        ("", None),
        ("count is 14, total is 5770.49\nthe totals differ\n0.00\n", "count is 14, total is 5770.49\nthe totals differ"),
        ("  step one\n\n  step two  \n42\n\n", "step one\n\n  step two"),
    ],
)
def test_the_explanation_is_what_was_printed_before_the_answer_line(stdout, expected):
    assert execute.executed_explanation(stdout) == expected


def test_a_long_explanation_is_capped_and_says_so():
    stdout = "x" * (execute.EXPLANATION_MAX_CHARS + 500) + "\n42\n"

    explanation = execute.executed_explanation(stdout)

    assert explanation.endswith(execute.EXPLANATION_TRUNCATED_MARK)
    assert len(explanation) <= execute.EXPLANATION_MAX_CHARS + len(execute.EXPLANATION_TRUNCATED_MARK) + 1


@pytest.mark.parametrize(
    "answer, expected_last_line",
    [
        ("42", "Answer: 42"),
        ("Answer: 41925", "Answer: 41925"),
        ("answer = 41925", "Answer: 41925"),
        ("Final answer: No", "Answer: No"),
        ("Answer:", "Answer: Answer:"),
        ("Answering machines: 3", "Answer: Answering machines: 3"),
    ],
)
def test_the_answer_line_carries_exactly_one_label(answer, expected_last_line):
    assert execute.format_answer(answer, "note").splitlines()[-1] == expected_last_line


def test_format_answer_puts_the_note_then_the_explanation_then_the_answer():
    text = execute.format_answer("0.00", "NOTE", "trailer total 5770.49 differs from 5406.34")

    assert text == (
        "NOTE\n\nExplanation printed by the program:\ntrailer total 5770.49 differs from 5406.34\n\nAnswer: 0.00"
    )
    assert execute.format_answer("0.00", "NOTE") == "NOTE\n\nAnswer: 0.00"


@pytest.mark.parametrize("claim, status", [("0.00", "claim_matched"), ("3.50", "corrected"), (None, "executed")])
def test_the_printed_explanation_is_in_the_answer_for_every_executed_status(execute_kit, claim, status):
    execute_kit.client.script(MID, execute_draft_json("print('...')", claim))
    execute_kit.sandbox_runner.script(
        "run", program_output("computed total 5406.34 differs from trailer 5770.49\nfile rejected, no fee\nAnswer: 0.00\n")
    )

    result = ask(execute_kit)

    assert result.execution_status == status
    assert result.executed_explanation == "computed total 5406.34 differs from trailer 5770.49\nfile rejected, no fee"
    assert result.executed_answer == "Answer: 0.00", "the stored answer is the line as printed"
    assert result.answer.splitlines()[-1] == "Answer: 0.00", "one label, final position"
    assert "computed total 5406.34 differs from trailer 5770.49" in result.answer
    assert result.answer.index(result.execution_note) < result.answer.index("Explanation printed by the program:")
    assert execute_kit.client.models_called() == [MID], "no extra model call writes the explanation"


def test_a_repaired_program_supplies_the_explanation_and_the_failed_one_does_not(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("print(1/0)", None), execute_draft_json("print(42)", None))
    execute_kit.sandbox_runner.script(
        "run",
        program_output("half-finished reasoning\n", exit_code=1, stderr="ZeroDivisionError"),
        program_output("six times seven\n42\n"),
    )

    result = ask(execute_kit)

    assert (result.execution_status, result.executed_explanation) == ("executed", "six times seven")
    assert "half-finished reasoning" not in result.answer


def test_an_unverified_claim_carries_no_executed_explanation(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("print(42)", "42"))
    execute_kit.sandbox_runner.script("create", program_output(exit_code=1, stderr="no sbx"))

    result = ask(execute_kit)

    assert (result.execution_status, result.executed_explanation) == ("unverified", None)
    assert "Explanation printed by the program" not in result.answer


def test_the_prompts_ask_the_program_to_print_its_reasoning_from_computed_values():
    for prompt in (execute.EXECUTE_SYSTEM_PROMPT, execute.REPAIR_SYSTEM_PROMPT):
        assert "it must print the explanation the question needs" in prompt
        assert "from values the program computed" in prompt
        assert "Its last line of output must be the final answer and nothing else" in prompt


# ---- the model may decline to claim an answer ----


@pytest.mark.parametrize("claim_fields", [{"claimed_answer": None}, {"claimed_answer": ""}, {"claimed_answer": "  "}, {}])
def test_executed_when_the_program_runs_cleanly_and_there_is_no_claim(execute_kit, claim_fields):
    draft = ok_result(json.dumps({"working": "multiply", "program": "print(17 * 23)", **claim_fields}))
    execute_kit.client.script(MID, draft)
    execute_kit.sandbox_runner.script("run", program_output("391\n"))

    result = ask(execute_kit)

    assert (result.status, result.execution_status) == ("ok", "executed")
    assert (result.claimed_answer, result.executed_answer, result.repaired) == (None, "391", False)
    assert result.answer.splitlines()[-1] == "Answer: 391"
    assert "not cross-checked" in result.execution_note and "Verified" not in result.execution_note
    assert [run.outcome for run in result.sandbox_runs] == ["no_claim"]
    assert result.sandbox_runs[0].claimed_answer is None
    assert execute_kit.client.models_called() == [MID]


def test_the_prompts_tell_the_model_not_to_work_the_answer_out_by_hand():
    for prompt in (execute.EXECUTE_SYSTEM_PROMPT, execute.REPAIR_SYSTEM_PROMPT):
        assert "claimed_answer: null, unless" in prompt
        assert "Never compute, trace or simulate the program by hand" in prompt


def test_the_schema_requires_the_claim_field_and_lets_it_be_null():
    schema = execute.draft_json_schema()
    assert "claimed_answer" in schema["required"]
    assert {"type": "null"} in schema["properties"]["claimed_answer"]["anyOf"]


def test_a_failed_program_with_no_claim_is_repaired_and_the_repair_is_executed(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("print(1/0)", None), execute_draft_json("print(42)", None))
    execute_kit.sandbox_runner.script("run", program_output(exit_code=1, stderr="ZeroDivisionError"), program_output("42\n"))

    result = ask(execute_kit)

    assert (result.status, result.execution_status, result.repaired) == ("ok", "executed", True)
    assert (result.claimed_answer, result.executed_answer) == (None, "42")
    assert "repaired program" in result.execution_note


def test_no_clean_run_and_no_claim_fails_the_run_with_no_answer(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("print(1/0)", None), execute_draft_json("print(2/0)", None))
    failure = program_output(exit_code=1, stderr="ZeroDivisionError")
    execute_kit.sandbox_runner.script("run", failure, failure)

    result = ask(execute_kit)

    assert (result.status, result.answer, result.error_type) == ("error", None, "execute_no_answer")
    assert result.execution_status == "failed"
    assert result.claimed_answer is None and result.executed_answer is None
    assert len(result.sandbox_runs) == 2, "both failed runs are still in the audit trail"
    assert len(execute_kit.telemetry.sandbox_recorded[-1]) == 2


def test_a_failed_repair_without_a_claim_falls_back_to_the_first_claim(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("print(1/0)", "42"), execute_draft_json("print(2/0)", None))
    failure = program_output(exit_code=1, stderr="ZeroDivisionError")
    execute_kit.sandbox_runner.script("run", failure, failure)

    result = ask(execute_kit)

    assert (result.status, result.execution_status, result.claimed_answer) == ("ok", "unverified", "42")
    assert result.answer.splitlines()[-1] == "Answer: 42"


def test_sandbox_unavailable_with_no_claim_fails_closed(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("print(42)", None))
    execute_kit.sandbox_runner.script("create", program_output(exit_code=1, stderr="no sbx"))

    result = ask(execute_kit)

    assert (result.status, result.answer, result.execution_status) == ("error", None, "failed")
    assert result.error_type == "execute_no_answer"
    assert execute_kit.client.models_called() == [MID], "no repair call for a sandbox failure"


def test_simulated_answer_error_calls_no_model_and_no_sandbox(execute_kit):
    result = ask(execute_kit, simulate="answer-error")

    assert (result.status, result.error_type) == ("error", "simulated_error")
    assert execute_kit.client.calls == [] and execute_kit.sandbox_runner.commands == []


def test_tier_flag_overrides_the_configured_tier(execute_kit):
    execute_kit.client.script("local-model", execute_draft_json())
    execute_kit.sandbox_runner.script("run", program_output("42\n"))

    result = ask(execute_kit, tier="local")

    assert (result.selected_tier, result.route_reason, result.execution_status) == ("local", "forced", "claim_matched")


def test_limits_come_from_the_execute_block():
    kit = HarnessKit(make_execute_config({"timeout_s": 5, "memory": "512m", "cpus": 1, "sbx_path": "/opt/sbx"}))
    kit.client.script(MID, execute_draft_json())
    kit.sandbox_runner.script("run", program_output("42\n"))

    ask(kit)

    create = kit.sandbox_runner.argv_of("create")
    assert create[0] == "/opt/sbx" and create[create.index("--memory") + 1] == "512m"
    run = kit.sandbox_runner.argv_of("run")
    assert run[run.index("timeout") + 3] == "5"


@pytest.mark.parametrize("think", ["low", "high", False, True])
def test_execute_think_is_sent_with_the_draft_and_the_repair_call(think):
    kit = HarnessKit(make_execute_config({"think": think}))
    kit.client.script(MID, execute_draft_json("print(1/0)"), execute_draft_json("print(42)"))
    kit.sandbox_runner.script("run", program_output(exit_code=1, stderr="ZeroDivisionError"), program_output("42\n"))

    ask(kit)

    assert [call.think for call in kit.client.calls] == [think, think]


def test_without_execute_think_the_tier_setting_is_used(execute_kit):
    execute_kit.client.script(MID, execute_draft_json())
    execute_kit.sandbox_runner.script("run", program_output("42\n"))

    ask(execute_kit)

    assert execute_kit.client.calls[0].think == execute_kit.config.tiers["mid"].think


def test_execute_think_does_not_reach_other_modes():
    kit = HarnessKit(make_execute_config({"think": "low"}))
    kit.client.script(MID, ok_result("plain answer"))

    kit.harness.answer(AskRequest.from_question("q", tier="mid"))

    assert kit.client.calls[-1].think == kit.config.tiers["mid"].think


def test_execute_think_rejects_an_unknown_level():
    with pytest.raises(ValidationError):
        make_execute_config({"think": "enormous"})


# ---- telemetry hand-off ----


def test_run_and_audit_rows_are_handed_to_telemetry(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("print(1/0)", "41"), execute_draft_json("print(6 * 7)", "42"))
    execute_kit.sandbox_runner.script(
        "run", program_output("", "ZeroDivisionError\n", exit_code=1), program_output("42\n")
    )

    result = ask(execute_kit)

    run = execute_kit.run_record
    assert (run.mode, run.execution_status, run.claimed_answer, run.executed_answer) == (
        "execute", "claim_matched", "42", "42"
    )
    assert run.repaired is True and run.trace_id == result.trace_id
    first, repair = execute_kit.sandbox_records
    assert (first.seq, first.attempt, first.program, first.outcome) == (1, "first", "print(1/0)", "program_failed")
    assert (first.exit_code, first.error_type, first.stderr, first.claimed_answer) == (
        1, "nonzero_exit", "ZeroDivisionError\n", "41"
    )
    assert (repair.seq, repair.attempt, repair.program, repair.outcome) == (2, "repair", "print(6 * 7)", "agree")
    assert (repair.exit_code, repair.stdout, repair.executed_answer, repair.status) == (0, "42\n", "42", "ok")
    for record in (first, repair):
        assert record.network_policy == "deny_all"
        assert json.loads(record.environment_names) == ["PATH", "HOME", "LANG"]
        assert record.sandbox_removed is True and record.sandbox_name.startswith("inference-arbiter-")
        assert record.runner == "python"


def test_other_modes_report_execution_as_skipped_and_write_no_audit_rows(kit):
    kit.client.script(MID, ok_result("plain answer"))

    result = kit.harness.answer(AskRequest.from_question("q"))

    assert result.execution_status == "skipped" and result.sandbox_runs == []
    assert result.executed_explanation is None
    assert kit.run_record.execution_status == "skipped"
    assert kit.sandbox_records == [] and kit.sandbox_runner.commands == []


# ---- config, request and CLI ----


def test_execute_mode_needs_an_execute_block(kit):
    with pytest.raises(ExecuteNotConfiguredError):
        kit.harness.answer(AskRequest.from_question("q", mode="execute"))


def test_execute_tier_must_be_an_answer_tier():
    raw_config = {**BASE_CONFIG, "execute": {"tier": "triage"}}
    with pytest.raises(ValidationError, match="execute.tier"):
        HarnessConfig.model_validate(raw_config)


def test_execute_block_is_optional_and_has_shipped_defaults():
    assert make_config().execute is None
    execute_config = make_execute_config().execute
    assert (execute_config.tier, execute_config.max_repairs, execute_config.timeout_s) == ("mid", 1, 20)
    with pytest.raises(ValidationError):
        make_execute_config({"max_repairs": 2})


def test_the_shipped_config_has_an_execute_block_on_the_free_mid_tier():
    from inference_arbiter.config import load_config
    from conftest import REAL_CONFIG_PATH

    shipped = load_config(REAL_CONFIG_PATH)
    assert shipped.execute is not None and shipped.execute.tier == "mid"
    assert shipped.execute.think == "medium"


@pytest.mark.parametrize(
    "overrides",
    [
        {"sources": ["notes.md"]},
        {"verifier_tier": "verify"},
        {"simulate": "triage-timeout"},
        {"simulate": "verifier-error"},
        {"simulate": "quote-mismatch"},
    ],
)
def test_request_options_that_do_not_belong_to_execute_mode_are_refused(overrides):
    with pytest.raises(ValidationError):
        AskRequest.from_question("q", mode="execute", **overrides)


def test_auto_mode_never_picks_execute(execute_kit):
    execute_kit.client.script(MID, ok_result("42"))

    result = execute_kit.harness.answer(AskRequest.from_question("What is 6 times 7?"))

    assert result.mode == "route" and execute_kit.sandbox_runner.commands == []


def test_cli_execute_flag_selects_execute_mode():
    arguments = cli.build_parser().parse_args(["ask", "q", "--execute"])
    assert cli._mode_from_arguments(arguments) == "execute"
    assert cli._mode_from_arguments(cli.build_parser().parse_args(["ask", "q"])) == "auto"


@pytest.mark.parametrize(
    "extra", [["--chain"], ["--route"], ["--sources", "a.md"], ["--triage-tier", "triage"], ["--verifier-tier", "v"]]
)
def test_cli_refuses_execute_with_another_mode(extra, capsys):
    assert cli.main(["ask", "q", "--execute", *extra]) == cli.EXIT_USAGE
    assert "--execute and --cross-check cannot be combined" in capsys.readouterr().err


def test_cli_trace_shows_what_was_executed(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("print(17 * 23)", "381"))
    execute_kit.sandbox_runner.script("run", program_output("391\n"))

    trace = cli.format_trace(ask(execute_kit))

    assert "execution     corrected, claimed '381', executed '391'" in trace
    assert "network deny_all" in trace and "env PATH,HOME,LANG" in trace and "removed: True" in trace


def test_cli_trace_says_when_the_model_declined_to_claim(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("print(17 * 23)", None))
    execute_kit.sandbox_runner.script("run", program_output("391\n"))

    trace = cli.format_trace(ask(execute_kit))

    assert "execution     executed, claimed nothing (declined), executed '391'" in trace
    assert "explanation   none printed by the program" in trace


def test_cli_trace_says_how_much_explanation_the_program_printed(execute_kit):
    execute_kit.client.script(MID, execute_draft_json("print('...')", None))
    execute_kit.sandbox_runner.script("run", program_output("17 times 23\n391\n"))

    assert "explanation   11 characters printed by the program" in cli.format_trace(ask(execute_kit))
