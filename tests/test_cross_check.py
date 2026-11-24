"""Execute mode's cross-check: two programs written independently, compared in code. Fakes only."""

from __future__ import annotations

import inspect
import itertools
import json

import pytest
from pydantic import ValidationError

from inference_arbiter import cli, execute, sandbox
from inference_arbiter.config import HarnessConfig
from inference_arbiter.router import UnknownTierError
from inference_arbiter.schemas import AskRequest

from conftest import (
    BASE_CONFIG,
    HarnessKit,
    error_result,
    execute_draft_json,
    make_execute_config,
    ok_result,
    program_output,
)

MID = "mid-model"
CODE = "code-model"
QUESTION = "What is 6 times 7?"


def cross_config(**execute_overrides) -> HarnessConfig:
    """A on mid, B on code (a different model), one after the other so scripted results line up."""
    settings = {"cross_check": True, "cross_check_tier": "code", "cross_check_parallel": False}
    settings.update(execute_overrides)
    return make_execute_config(settings)


@pytest.fixture
def cross_kit() -> HarnessKit:
    return HarnessKit(cross_config())


def ask(kit: HarnessKit, question: str = QUESTION, **overrides):
    return kit.harness.answer(AskRequest.from_question(question, mode="execute", **overrides))


def draft(program: str = "print(6 * 7)"):
    return execute_draft_json(program, None)


def sandbox_result(error_type: str | None = None, stdout: str = "") -> sandbox.SandboxResult:
    return sandbox.SandboxResult(
        status="ok" if error_type is None else "error", error_type=error_type, error_message=None,
        runner="python", sandbox_name="inference-arbiter-test", exit_code=0 if error_type is None else 1,
        stdout=stdout, stderr="", stdout_truncated=False, stderr_truncated=False, timed_out=False,
        duration_ms=1.0, total_ms=2.0, network_policy="deny_all", environment_names=("PATH",),
        sandbox_removed=True,
    )


# ---- values_match: the comparison is code, and it fails closed ----


@pytest.mark.parametrize(
    "answer_a, answer_b, rule",
    [
        ("42", "42", "integer"),
        ("42", " 42 ", "integer"),
        ("Answer: 25775", "25775", "integer"),
        ("25,775", "25775", "integer"),
        ("-7", "-7", "integer"),
        ("+7", "7", "integer"),
        ("**42**", "`42`", "integer"),
        ("42.", "42", "integer"),
        ("0xA3C5", "0xa3c5", "hex"),
        ("0xA3C5", "41925", "hex"),
        ("$0.00", "0.00", "decimal"),
        ("42", "42.0", "decimal"),
        ("1580.17", "1580.17", "decimal"),
        ("1580.17", "1580.1687", "decimal"),
        ("1580.1687", "1580.17", "decimal"),
        ("0.3", "0.30000000000000004", "decimal"),
        ("0.1234567890123", "0.1234567890124", "decimal"),
        ("1e3", "1000", "decimal"),
        ("$1,250.50", "1250.5", "decimal"),
        ("[-3333, -3333, -3333]", "-3333 -3333 -3333", "number_list"),
        ("9592, 192139", "(9592; 192139)", "number_list"),
        ("[1.5, 2]", "[1.50, 2]", "number_list"),
        ("No", "no.", "string"),
        ("None", "none", "string"),
        ("Yes", "Answer: yes", "string"),
        ('"Hello  World"', "hello world", "string"),
        ("No amount should be posted", "no amount should be posted", "string"),
        ("5%", "5%", "string"),
        ("\uff46\uff55\uff4c\uff4c\uff57\uff49\uff44\uff54\uff48", "fullwidth", "string"),
    ],
)
def test_values_that_match(answer_a, answer_b, rule):
    assert execute.values_match(answer_a, answer_b) == execute.ValueComparison(True, rule)
    assert execute.values_match(answer_b, answer_a) == execute.ValueComparison(True, rule), "the rule is symmetric"


@pytest.mark.parametrize(
    "answer_a, answer_b, rule",
    [
        ("25775", "7466", "integer"),
        ("42", "43", "integer"),
        ("1000000000000000000000", "1000000000000000000001", "integer"),
        ("0xA3C5", "0xA3C6", "hex"),
        ("0x10", "10", "hex"),
        ("1580.17", "1580.18", "decimal"),
        ("1580", "1580.17", "decimal"),
        ("42", "42.4", "decimal"),
        ("3.14", "3.15159", "decimal"),
        ("0.30", "0.36", "decimal"),
        ("1.0", "1.1", "decimal"),
        ("[1, 2, 3]", "[1, 2, 4]", "number_list"),
        ("[1, 2, 3]", "[1, 2]", "number_list"),
        ("1 2", "1", "string"),
        ("5%", "5", "string"),
        ("24 legs", "24", "string"),
        ("yes", "no", "string"),
        ("No", "No amount should be posted", "string"),
        ("No", "Yes", "string"),
        ("No", "No, unless the total is positive", "string"),
        ("No", "No: it returns [-3333, -3333, -3333]", "string"),
        ("Yes", "Yes, if the remainder is zero", "string"),
        ("None", "No", "string"),
        ("None", "0", "string"),
        ("None", "Malformed message", "string"),
        ("No amount posted: malformed message", "Malformed message", "string"),
        ("TXN-0042", "TXN-0043", "string"),
        ("TXN-0042", "TXN 0042", "string"),
        ("2028-05-19", "2028-05-20", "string"),
        ("0001875000978A", "0001875000978B", "string"),
        ("true", "1", "string"),
        ("the answer is 42", "it prints 42", "string"),
        ("0xZZ", "0", "string"),
        ("42", "error", "string"),
        ("", "", "string"),
        ("", "42", "string"),
    ],
)
def test_values_that_do_not_match(answer_a, answer_b, rule):
    assert execute.values_match(answer_a, answer_b) == execute.ValueComparison(False, rule)
    assert not execute.values_match(answer_b, answer_a).match


def test_the_decimal_tolerance_is_the_stated_one():
    assert execute.CROSS_CHECK_RELATIVE_TOLERANCE == execute.Decimal("1e-9")
    assert execute.values_match("1000.0000001", "1000.0000002").match  # 1e-10 relative
    assert not execute.values_match("1000.00001", "1000.00002").match  # 1e-8 relative
    assert execute.values_match("2.5", "2.45").match, "half a unit in the last place of the rounded side"
    assert execute.values_match("2.5", "2.55").match
    assert not execute.values_match("2.5", "2.56").match


def test_the_cross_check_comparison_is_stricter_than_the_claim_comparison():
    """`answers_agree` lets a claim of "24 legs" agree with an output of "24". Two programs do not get that."""
    assert execute.answers_agree("24 legs", "24")
    assert not execute.values_match("24 legs", "24").match


# ---- classify_run ----


@pytest.mark.parametrize(
    "result, expected",
    [
        (sandbox_result(stdout="42\n"), ("ran", None)),
        (sandbox_result(stdout="why\n42\n"), ("ran", None)),
        (sandbox_result(stdout=""), ("program_failed", "no_output")),
        (sandbox_result(stdout=" \n"), ("program_failed", "no_output")),
        (sandbox_result("timeout"), ("program_failed", "timeout")),
        (sandbox_result("nonzero_exit", stdout="42\n"), ("program_failed", "nonzero_exit")),
        (sandbox_result("output_too_large"), ("program_failed", "output_too_large")),
        (sandbox_result("sandbox_unavailable"), ("infra_failed", "sandbox_unavailable")),
        (sandbox_result("network_policy_not_applied"), ("infra_failed", "network_policy_not_applied")),
        (sandbox_result("some_future_error"), ("infra_failed", "some_future_error")),
    ],
)
def test_classify_run(result, expected):
    assert execute.classify_run(result) == expected


# ---- decide_track: one program's own course, every combination ----

RUN_STATES = ("ran", "program_failed", "infra_failed")
TRACK_REPAIR_STATES = ("not_attempted", "ran", "program_failed", "infra_failed", "draft_failed")
TRACK_COMBINATIONS = list(itertools.product(RUN_STATES, TRACK_REPAIR_STATES, (True, False)))


def expected_track(first: str, repair: str, repairs_allowed: bool) -> tuple:
    """The table written out again, independently of decide_track."""
    if first == "ran":
        return ("return", "ran", "first")
    if first == "infra_failed":
        return ("return", "infra_failed", None)
    return {
        ("not_attempted", True): ("repair", None, None),
        ("not_attempted", False): ("return", "program_failed", None),
        ("ran", True): ("return", "ran", "repair"),
        ("ran", False): ("return", "ran", "repair"),
        ("program_failed", True): ("return", "program_failed", None),
        ("program_failed", False): ("return", "program_failed", None),
        ("infra_failed", True): ("return", "infra_failed", None),
        ("infra_failed", False): ("return", "infra_failed", None),
        ("draft_failed", True): ("return", "program_failed", None),
        ("draft_failed", False): ("return", "program_failed", None),
    }[(repair, repairs_allowed)]


@pytest.mark.parametrize("first, repair, repairs_allowed", TRACK_COMBINATIONS)
def test_decide_track_truth_table(first, repair, repairs_allowed):
    decision = execute.decide_track(first, repair, repairs_allowed)
    assert (decision.action, decision.state, decision.use) == expected_track(first, repair, repairs_allowed)


def test_decide_track_covers_all_30_combinations_and_only_repairs_a_failed_program():
    assert len(TRACK_COMBINATIONS) == 30
    for first, repair, repairs_allowed in TRACK_COMBINATIONS:
        decision = execute.decide_track(first, repair, repairs_allowed)
        if decision.action == "repair":
            assert (first, repair, repairs_allowed) == ("program_failed", "not_attempted", True)
        if decision.state == "ran":
            assert decision.use in ("first", "repair"), "a program that ran says which run it was"


# ---- decide_cross_check: the verdict, every combination ----

TRACK_STATES = ("ran", "program_failed", "infra_failed", "draft_failed")
CROSS_COMBINATIONS = list(itertools.product(TRACK_STATES, TRACK_STATES, (True, False, None)))


def expected_verdict(a: str, b: str, match: bool | None) -> tuple:
    """The table written out again, independently of decide_cross_check."""
    a_ran, b_ran = a == "ran", b == "ran"
    if a_ran and b_ran:
        return ("verified", "a") if match is True else ("disputed", None)
    if a_ran:
        return ("one_sided", "a")
    if b_ran:
        return ("one_sided", "b")
    return ("failed", None)


@pytest.mark.parametrize("a, b, match", CROSS_COMBINATIONS)
def test_decide_cross_check_truth_table(a, b, match):
    decision = execute.decide_cross_check(a, b, match)
    assert (decision.status, decision.answer_from) == expected_verdict(a, b, match)


def test_decide_cross_check_covers_all_48_combinations():
    assert len(CROSS_COMBINATIONS) == 48
    statuses = {execute.decide_cross_check(a, b, match).status for a, b, match in CROSS_COMBINATIONS}
    assert statuses == {"verified", "disputed", "one_sided", "failed"}


def test_verified_needs_both_programs_to_run_and_a_positive_match():
    verified = [
        (a, b, match) for a, b, match in CROSS_COMBINATIONS
        if execute.decide_cross_check(a, b, match).status == "verified"
    ]
    assert verified == [("ran", "ran", True)]


def test_an_unknown_comparison_is_never_agreement():
    assert execute.decide_cross_check("ran", "ran", None).status == "disputed"


def test_only_a_program_that_ran_supplies_an_answer():
    for a, b, match in CROSS_COMBINATIONS:
        decision = execute.decide_cross_check(a, b, match)
        if decision.answer_from == "a":
            assert a == "ran"
        if decision.answer_from == "b":
            assert b == "ran" and a != "ran"


def test_the_claim_table_never_returns_verified():
    """One word, one meaning: `verified` comes from two programs, never from a claim."""
    classes = ("agree", "disagree", "no_claim", "program_failed", "infra_failed")
    repairs = ("not_attempted", "agree", "disagree", "no_claim", "program_failed", "infra_failed", "draft_failed")
    for first, repair, allowed, first_claim, repair_claim in itertools.product(
        classes, repairs, (True, False), (True, False), (True, False)
    ):
        decision = execute.decide(first, repair, allowed, first_claim, repair_claim)
        assert decision.status not in ("verified", "disputed", "one_sided")


# ---- control flow through the harness ----


def test_verified_when_both_programs_run_and_print_the_same_value(cross_kit):
    cross_kit.client.script(MID, draft("print('six sevens'); print(6 * 7)"))
    cross_kit.client.script(CODE, draft("print(sum([7] * 6))"))
    cross_kit.sandbox_runner.script("run", program_output("six sevens\n42\n"), program_output("42\n"))

    result = ask(cross_kit)

    assert (result.status, result.mode, result.execution_status) == ("ok", "execute", "verified")
    assert result.answer.splitlines()[-1] == "Answer: 42"
    assert result.answer.startswith("Verified by two programs")
    assert "tier mid, model mid-model" in result.answer and "tier code, model code-model" in result.answer
    assert "compared as integer" in result.answer
    assert (result.cross_check, result.cross_check_tier, result.cross_check_rule) == (True, "code", "integer")
    assert (result.executed_answer, result.executed_answer_b, result.repaired) == ("42", "42", False)
    assert (result.selected_tier, result.final_tier) == ("mid", "mid")
    assert [(call.role, call.tier) for call in result.calls] == [("draft", "mid"), ("draft_b", "code")]
    assert [(run.program_label, run.attempt, run.outcome) for run in result.sandbox_runs] == [
        ("a", "first", "ran"), ("b", "first", "ran")
    ]
    assert len({run.sandbox_name for run in result.sandbox_runs}) == 2, "each program gets its own fresh sandbox"
    assert cross_kit.sandbox_runner.existing == set()
    assert result.executed_explanation == "six sevens", "program A's explanation is the one shown"


def test_disputed_when_both_run_and_print_different_values(cross_kit):
    cross_kit.client.script(MID, draft("print('A reasons'); print(25775)"))
    cross_kit.client.script(CODE, draft("print('B reasons'); print(7466)"))
    cross_kit.sandbox_runner.script("run", program_output("A reasons\n25775\n"), program_output("B reasons\n7466\n"))

    result = ask(cross_kit)

    assert (result.status, result.execution_status) == ("ok", "disputed")
    assert result.answer.startswith("DISPUTED: no answer is confirmed.")
    assert "Program A (tier mid, model mid-model) printed: 25775" in result.answer
    assert "Program B (tier code, model code-model) printed: 7466" in result.answer
    assert result.answer.splitlines()[-1] == execute.DISPUTED_LAST_LINE
    assert not any(line.startswith("Answer:") for line in result.answer.splitlines()), "no line reads as the answer"
    assert "Explanation printed by program A:\nA reasons" in result.answer
    assert "Explanation printed by program B:\nB reasons" in result.answer
    assert (result.executed_answer, result.executed_answer_b, result.cross_check_rule) == ("25775", "7466", "integer")
    assert cross_kit.client.models_called() == [MID, CODE], "a dispute is not settled by another model call"


def test_a_dispute_writes_both_outcomes_and_both_outputs_to_telemetry(cross_kit):
    cross_kit.client.script(MID, draft("print(25775)"))
    cross_kit.client.script(CODE, draft("print(7466)"))
    cross_kit.sandbox_runner.script("run", program_output("25775\n"), program_output("because\n7466\n"))

    result = ask(cross_kit)

    run = cross_kit.run_record
    assert (run.mode, run.execution_status, run.cross_check, run.cross_check_tier) == (
        "execute", "disputed", True, "code"
    )
    assert (run.executed_answer, run.executed_answer_b, run.cross_check_rule) == ("25775", "7466", "integer")
    assert run.trace_id == result.trace_id
    first, second = cross_kit.sandbox_records
    assert (first.seq, first.program_label, first.program, first.stdout, first.executed_answer, first.outcome) == (
        1, "a", "print(25775)", "25775\n", "25775", "ran"
    )
    assert (second.seq, second.program_label, second.program, second.stdout, second.executed_answer) == (
        2, "b", "print(7466)", "because\n7466\n", "7466"
    )
    assert [(call.role, call.tier, call.model) for call in cross_kit.call_records] == [
        ("draft", "mid", MID), ("draft_b", "code", CODE)
    ]


def test_one_sided_when_program_b_fails_and_its_repair_fails_too(cross_kit):
    cross_kit.client.script(MID, draft("print(42)"))
    cross_kit.client.script(CODE, draft("boom"), draft("boom again"))
    cross_kit.sandbox_runner.script(
        "run",
        program_output("why\n42\n"),
        program_output("", "NameError: boom\n", exit_code=1),
        program_output("", "NameError: again\n", exit_code=1),
    )

    result = ask(cross_kit)

    assert (result.status, result.execution_status) == ("ok", "one_sided")
    assert result.answer.startswith("NOT CROSS-CHECKED: only one of the two programs produced an answer.")
    assert "Program A (tier mid, model mid-model) computed this" in result.answer
    assert "Program B (tier code, model code-model) produced nothing to compare it with" in result.answer
    assert "nonzero_exit" in result.answer and "Nothing independent agreed" in result.answer
    assert result.answer.splitlines()[-1] == "Answer: 42"
    assert (result.executed_answer, result.executed_answer_b, result.cross_check_rule) == ("42", None, None)
    assert [(call.role, call.tier) for call in result.calls] == [
        ("draft", "mid"), ("draft_b", "code"), ("repair_b", "code")
    ]
    assert [(run.program_label, run.attempt, run.outcome) for run in result.sandbox_runs] == [
        ("a", "first", "ran"), ("b", "first", "program_failed"), ("b", "repair", "program_failed")
    ]
    assert [record.stderr for record in cross_kit.sandbox_records] == ["", "NameError: boom\n", "NameError: again\n"]
    assert cross_kit.run_record.execution_status == "one_sided"


def test_one_sided_returns_program_b_when_only_b_ran(cross_kit):
    cross_kit.client.script(MID, error_result("timeout"))
    cross_kit.client.script(CODE, draft("print(42)"))
    cross_kit.sandbox_runner.script("run", program_output("B says\n42\n"))

    result = ask(cross_kit)

    assert (result.status, result.execution_status) == ("ok", "one_sided")
    assert "Program B (tier code, model code-model) computed this" in result.answer
    assert "the call that should have written it failed (timeout)" in result.answer
    assert result.answer.splitlines()[-1] == "Answer: 42"
    assert (result.executed_answer, result.executed_answer_b) == (None, "42")
    assert result.executed_explanation == "B says"
    assert result.error_type is None


def test_failed_when_neither_program_produces_an_answer(cross_kit):
    cross_kit.client.script(MID, draft("bad"), draft("bad"))
    cross_kit.client.script(CODE, ok_result("not json"))
    cross_kit.sandbox_runner.script(
        "run", program_output("", "E1", exit_code=1), program_output("", "E2", exit_code=1)
    )

    result = ask(cross_kit)

    assert (result.status, result.answer, result.execution_status) == ("error", None, "failed")
    assert result.error_type == execute.ERROR_NO_ANSWER
    assert "Neither program produced one" in result.error_message
    assert "draft_schema_invalid" in result.error_message
    assert cross_kit.run_record.status == "error"


def test_each_program_gets_its_own_repair_and_a_repaired_pair_can_be_verified(cross_kit):
    cross_kit.client.script(MID, draft("print(1/0)"), draft("print(42)"))
    cross_kit.client.script(CODE, draft("import nope"), draft("print(40 + 2)"))
    cross_kit.sandbox_runner.script(
        "run",
        program_output("", "ZeroDivisionError: A-ONLY\n", exit_code=1),
        program_output("42\n"),
        program_output("", "ModuleNotFoundError: B-ONLY\n", exit_code=1),
        program_output("42\n"),
    )

    result = ask(cross_kit)

    assert (result.execution_status, result.repaired) == ("verified", True)
    assert "Repaired after a first failure, each from its own error only: program A and program B." in result.answer
    assert [(call.role, call.tier) for call in result.calls] == [
        ("draft", "mid"), ("repair", "mid"), ("draft_b", "code"), ("repair_b", "code")
    ]
    repair_a, repair_b = cross_kit.client.calls[1], cross_kit.client.calls[3]
    assert "A-ONLY" in repair_a.messages[-1]["content"] and "print(1/0)" in repair_a.messages[-1]["content"]
    assert "B-ONLY" in repair_b.messages[-1]["content"] and "import nope" in repair_b.messages[-1]["content"]
    assert len({run.sandbox_name for run in result.sandbox_runs}) == 4


def test_no_repairs_when_max_repairs_is_zero():
    kit = HarnessKit(cross_config(max_repairs=0))
    kit.client.script(MID, draft("print(42)"))
    kit.client.script(CODE, draft("boom"))
    kit.sandbox_runner.script("run", program_output("42\n"), program_output("", "E", exit_code=1))

    result = ask(kit)

    assert result.execution_status == "one_sided"
    assert [call.role for call in result.calls] == ["draft", "draft_b"]


def test_sandbox_unavailable_for_one_program_is_one_sided_without_a_repair(cross_kit):
    cross_kit.client.script(MID, draft("print(42)"))
    cross_kit.client.script(CODE, draft("print(42)"))
    cross_kit.sandbox_runner.script("create", program_output(), FileNotFoundError("sbx"))
    cross_kit.sandbox_runner.script("run", program_output("42\n"))

    result = ask(cross_kit)

    assert result.execution_status == "one_sided"
    assert "could not be run in the sandbox (sandbox_unavailable)" in result.answer
    assert [call.role for call in result.calls] == ["draft", "draft_b"]


def test_a_claim_plays_no_part_in_a_cross_check(cross_kit):
    cross_kit.client.script(MID, execute_draft_json("print(42)", "41"))
    cross_kit.client.script(CODE, execute_draft_json("print(42)", "40"))
    cross_kit.sandbox_runner.script("run", program_output("42\n"), program_output("42\n"))

    result = ask(cross_kit)

    assert result.execution_status == "verified"
    assert [record.claimed_answer for record in cross_kit.sandbox_records] == ["41", "40"], "claims are still audited"


def test_the_labels_a_program_prints_are_ignored_by_the_comparison(cross_kit):
    cross_kit.client.script(MID, draft("a"))
    cross_kit.client.script(CODE, draft("b"))
    cross_kit.sandbox_runner.script("run", program_output("Answer: 18,750\n"), program_output("18750\n"))

    result = ask(cross_kit)

    assert result.execution_status == "verified"
    assert result.answer.splitlines()[-1] == "Answer: 18,750"


def test_simulated_answer_error_fails_program_a_only(cross_kit):
    cross_kit.client.script(CODE, draft("print(42)"))
    cross_kit.sandbox_runner.script("run", program_output("42\n"))

    result = ask(cross_kit, simulate="answer-error")

    assert result.execution_status == "one_sided" and result.executed_answer_b == "42"
    assert cross_kit.client.models_called() == [CODE]


# ---- strict independence: nothing of one program reaches the other program's prompts ----

SENTINEL_A = "SENTINEL-FROM-PROGRAM-A"
SENTINEL_B = "SENTINEL-FROM-PROGRAM-B"


def sentinel_draft(sentinel: str, part: str):
    return ok_result(json.dumps({
        "working": f"{sentinel} working {part}",
        "program": f"# {sentinel} program {part}\nprint(42)",
        "claimed_answer": f"{sentinel} claim {part}",
    }))


def sentinel_failure(sentinel: str):
    return program_output(f"{sentinel} stdout of a failed run\n", f"{sentinel} stderr traceback\n", exit_code=1)


def sentinel_success(sentinel: str, value: str = "42"):
    return program_output(f"{sentinel} explanation line\n{value}\n", f"{sentinel} stderr warning\n")


def messages_sent_to(kit: HarnessKit, model: str) -> str:
    return json.dumps([call.messages for call in kit.client.calls if call.model == model])


def assert_no_cross_talk(kit: HarnessKit) -> None:
    """Fails if any text that came from one program is in a message sent for the other."""
    to_a, to_b = messages_sent_to(kit, MID), messages_sent_to(kit, CODE)
    assert to_a != "[]" and to_b != "[]", "both programs were asked for"
    assert SENTINEL_A not in to_b, "program A's text reached a prompt for program B"
    assert SENTINEL_B not in to_a, "program B's text reached a prompt for program A"


INDEPENDENCE_SCENARIOS = {
    # name: (A's sandbox results, B's sandbox results); a failed first run triggers that program's repair
    "both_run": ([sentinel_success(SENTINEL_A)], [sentinel_success(SENTINEL_B)]),
    "both_dispute": ([sentinel_success(SENTINEL_A, "1")], [sentinel_success(SENTINEL_B, "2")]),
    "a_repaired": ([sentinel_failure(SENTINEL_A), sentinel_success(SENTINEL_A)], [sentinel_success(SENTINEL_B)]),
    "b_repaired": ([sentinel_success(SENTINEL_A)], [sentinel_failure(SENTINEL_B), sentinel_success(SENTINEL_B)]),
    "both_repaired": (
        [sentinel_failure(SENTINEL_A), sentinel_success(SENTINEL_A)],
        [sentinel_failure(SENTINEL_B), sentinel_success(SENTINEL_B)],
    ),
    "a_fails_twice": ([sentinel_failure(SENTINEL_A), sentinel_failure(SENTINEL_A)], [sentinel_success(SENTINEL_B)]),
    "b_fails_twice": ([sentinel_success(SENTINEL_A)], [sentinel_failure(SENTINEL_B), sentinel_failure(SENTINEL_B)]),
    "both_fail_twice": (
        [sentinel_failure(SENTINEL_A), sentinel_failure(SENTINEL_A)],
        [sentinel_failure(SENTINEL_B), sentinel_failure(SENTINEL_B)],
    ),
}


def run_independence_scenario(kit: HarnessKit, scenario: str) -> None:
    runs_a, runs_b = INDEPENDENCE_SCENARIOS[scenario]
    kit.client.script(MID, sentinel_draft(SENTINEL_A, "first"), sentinel_draft(SENTINEL_A, "repair"))
    kit.client.script(CODE, sentinel_draft(SENTINEL_B, "first"), sentinel_draft(SENTINEL_B, "repair"))
    kit.sandbox_runner.script("run", *runs_a, *runs_b)
    ask(kit)


@pytest.mark.parametrize("scenario", sorted(INDEPENDENCE_SCENARIOS))
def test_nothing_of_program_a_reaches_program_b_and_nothing_of_b_reaches_a(cross_kit, scenario):
    run_independence_scenario(cross_kit, scenario)

    assert_no_cross_talk(cross_kit)
    repairs_expected = sum(1 for runs in INDEPENDENCE_SCENARIOS[scenario] if len(runs) == 2)
    repair_calls = [call for call in cross_kit.client.calls if "failed when it was run" in call.messages[0]["content"]]
    assert len(repair_calls) == repairs_expected, "the scenario really went through the repair prompts"


def test_a_repair_prompt_does_carry_its_own_programs_failure(cross_kit):
    """The other half of the independence test: the sentinels are real, a repair sees its own program."""
    run_independence_scenario(cross_kit, "both_repaired")

    to_a, to_b = messages_sent_to(cross_kit, MID), messages_sent_to(cross_kit, CODE)
    for own_text in ("program first", "stderr traceback", "stdout of a failed run"):
        assert f"{SENTINEL_A} {own_text}" in to_a
        assert f"{SENTINEL_B} {own_text}" in to_b


def test_program_b_is_asked_with_the_question_and_nothing_else(cross_kit):
    run_independence_scenario(cross_kit, "a_repaired")

    b_draft = next(call for call in cross_kit.client.calls if call.model == CODE)
    assert b_draft.messages == [
        {"role": "system", "content": execute.EXECUTE_SYSTEM_PROMPT + execute.BARE_ANSWER_LINE_RULE},
        {"role": "user", "content": QUESTION},
    ]


def test_the_independence_check_catches_a_leak(cross_kit, monkeypatch):
    """The check above is not vacuous: a repair prompt that quotes the other program fails it."""
    seen: list[str] = []
    honest_builder = execute.build_repair_messages

    def leaky_builder(question, attempt, system_prompt):
        seen.append(attempt.program)
        messages = honest_builder(question, attempt, system_prompt)
        messages[-1]["content"] += "\n\nThe other program was:\n" + "\n".join(seen)
        return messages

    monkeypatch.setattr(execute, "build_repair_messages", leaky_builder)
    run_independence_scenario(cross_kit, "both_repaired")

    with pytest.raises(AssertionError, match="program A's text reached a prompt for program B"):
        assert_no_cross_talk(cross_kit)


def test_independence_holds_when_the_two_programs_run_at_the_same_time():
    kit = HarnessKit(cross_config(cross_check_parallel=True))
    kit.client.script(MID, sentinel_draft(SENTINEL_A, "first"))
    kit.client.script(CODE, sentinel_draft(SENTINEL_B, "first"))
    kit.sandbox_runner.script("run", program_output("same text\n42\n"), program_output("same text\n42\n"))

    result = ask(kit)

    assert result.execution_status == "verified"
    assert_no_cross_talk(kit)
    assert [(call.role, call.tier) for call in result.calls] == [("draft", "mid"), ("draft_b", "code")]
    assert [run.program_label for run in result.sandbox_runs] == ["a", "b"], "rows keep a fixed order"
    assert kit.sandbox_runner.existing == set()


def test_a_program_track_has_no_input_that_could_carry_the_other_program():
    """Structural: the function that builds every message of one program takes only these."""
    assert list(inspect.signature(execute.run_program_track).parameters) == [
        "label", "question", "conversation", "tier_name", "system_prompt", "max_repairs",
        "effort_override", "call_tier", "run_in_sandbox", "simulated_draft_failure",
    ]
    assert list(inspect.signature(execute.build_repair_messages).parameters) == [
        "question", "attempt", "system_prompt",
    ]
    assert list(inspect.signature(execute.track_system_prompt).parameters) == ["independent_method"]
    assert list(inspect.signature(execute.track_repair_system_prompt).parameters) == []


# ---- the answer line: both programs are told to print the bare value, in the same words ----


def test_the_bare_answer_line_rule_fixes_the_form_of_the_last_line():
    rule = execute.BARE_ANSWER_LINE_RULE
    assert "only the bare value the question asks for" in rule
    assert "exactly Yes or exactly No, and nothing else on the line" in rule
    assert "exactly None, and nothing else on the line" in rule
    assert "If the question states a format for the answer, follow that format exactly" in rule
    assert "never on the last line" in rule


def test_both_programs_get_the_answer_line_rule_on_draft_and_on_repair(cross_kit):
    run_independence_scenario(cross_kit, "both_repaired")

    system_prompts = {
        (call.model, call.messages[0]["content"].startswith(execute.REPAIR_SYSTEM_PROMPT)): call.messages[0]["content"]
        for call in cross_kit.client.calls
    }
    assert system_prompts == {
        (MID, False): execute.EXECUTE_SYSTEM_PROMPT + execute.BARE_ANSWER_LINE_RULE,
        (CODE, False): execute.EXECUTE_SYSTEM_PROMPT + execute.BARE_ANSWER_LINE_RULE,
        (MID, True): execute.REPAIR_SYSTEM_PROMPT + execute.BARE_ANSWER_LINE_RULE,
        (CODE, True): execute.REPAIR_SYSTEM_PROMPT + execute.BARE_ANSWER_LINE_RULE,
    }
    assert_no_cross_talk(cross_kit)


def test_the_answer_line_rule_carries_nothing_of_either_program():
    """The rule is a constant: the same text for A and for B, built from no run-time value."""
    assert execute.track_repair_system_prompt() == execute.REPAIR_SYSTEM_PROMPT + execute.BARE_ANSWER_LINE_RULE
    assert execute.track_system_prompt(False) == execute.EXECUTE_SYSTEM_PROMPT + execute.BARE_ANSWER_LINE_RULE
    assert execute.track_system_prompt(True) == (
        execute.EXECUTE_SYSTEM_PROMPT + execute.BARE_ANSWER_LINE_RULE + execute.INDEPENDENT_METHOD_RULE
    )


def test_single_program_execute_mode_keeps_its_prompts():
    assert execute.BARE_ANSWER_LINE_RULE not in execute.EXECUTE_SYSTEM_PROMPT
    assert execute.BARE_ANSWER_LINE_RULE not in execute.REPAIR_SYSTEM_PROMPT
    assert execute.build_draft_messages([])[0]["content"] == execute.EXECUTE_SYSTEM_PROMPT


def test_two_programs_that_follow_the_rule_verify_and_a_qualified_line_still_disputes(cross_kit):
    cross_kit.client.script(MID, draft(), draft())
    cross_kit.client.script(CODE, draft(), draft())
    cross_kit.sandbox_runner.script(
        "run",
        program_output("the loop never runs\nNo\n"), program_output("sum is -9999\nNo\n"),
        program_output("the loop never runs\nNo\n"), program_output("No, unless the total is positive\n"),
    )

    agreed = ask(cross_kit)
    qualified = ask(cross_kit)

    assert (agreed.execution_status, agreed.cross_check_rule) == ("verified", "string")
    assert (qualified.execution_status, qualified.cross_check_rule) == ("disputed", "string")
    assert qualified.answer.splitlines()[-1] == execute.DISPUTED_LAST_LINE


# ---- diversity: two tiers by preference, a different method when the model is the same ----


def test_two_different_models_get_the_same_plain_prompt(cross_kit):
    cross_kit.client.script(MID, draft())
    cross_kit.client.script(CODE, draft())
    cross_kit.sandbox_runner.script("run", program_output("42\n"), program_output("42\n"))

    ask(cross_kit)

    prompts = [call.messages[0]["content"] for call in cross_kit.client.calls]
    plain_prompt = execute.EXECUTE_SYSTEM_PROMPT + execute.BARE_ANSWER_LINE_RULE
    assert prompts == [plain_prompt, plain_prompt]


@pytest.mark.parametrize("settings", [{"cross_check_tier": None}, {"cross_check_tier": "mid"}])
def test_one_tier_for_both_asks_program_b_for_an_independent_method(settings):
    kit = HarnessKit(cross_config(**settings))
    kit.client.script(MID, draft("print(6 * 7)"), draft("print(sum([6] * 7))"))
    kit.sandbox_runner.script("run", program_output("42\n"), program_output("42\n"))

    result = ask(kit)

    prompt_a, prompt_b = [call.messages[0]["content"] for call in kit.client.calls]
    assert prompt_a == execute.EXECUTE_SYSTEM_PROMPT + execute.BARE_ANSWER_LINE_RULE
    assert prompt_b == prompt_a + execute.INDEPENDENT_METHOD_RULE
    assert "pick a different method" in prompt_b and "You are not shown the other program" in prompt_b
    assert (result.execution_status, result.cross_check_tier) == ("verified", "mid")
    assert "A by tier mid, model mid-model, B by tier mid, model mid-model" in result.answer


def test_two_tiers_that_run_one_model_count_as_one_model():
    raw_tiers = json.loads(json.dumps(BASE_CONFIG["tiers"]))
    raw_tiers["code"]["model"] = MID
    kit = HarnessKit(make_execute_config(
        {"cross_check": True, "cross_check_tier": "code", "cross_check_parallel": False}, tiers=raw_tiers
    ))
    kit.client.script(MID, draft(), draft())
    kit.sandbox_runner.script("run", program_output("42\n"), program_output("42\n"))

    ask(kit)

    assert kit.client.calls[1].messages[0]["content"].endswith(execute.INDEPENDENT_METHOD_RULE)


def test_execute_think_goes_to_program_a_and_not_to_a_different_b_tier():
    kit = HarnessKit(cross_config(think="medium"))
    kit.client.script(MID, draft())
    kit.client.script(CODE, draft())
    kit.sandbox_runner.script("run", program_output("42\n"), program_output("42\n"))

    ask(kit)

    assert [call.think for call in kit.client.calls] == ["medium", None]


def test_cross_check_think_sets_program_bs_thinking():
    kit = HarnessKit(cross_config(think="medium", cross_check_think=False))
    kit.client.script(MID, draft())
    kit.client.script(CODE, draft("x"), draft())
    kit.sandbox_runner.script("run", program_output("42\n"), program_output("", "E", exit_code=1), program_output("42\n"))

    ask(kit)

    assert [call.think for call in kit.client.calls] == ["medium", False, False]


def test_on_one_tier_program_b_uses_execute_think_unless_told_otherwise():
    kit = HarnessKit(cross_config(cross_check_tier=None, think="low"))
    kit.client.script(MID, draft(), draft())
    kit.sandbox_runner.script("run", program_output("42\n"), program_output("42\n"))

    ask(kit)

    assert [call.think for call in kit.client.calls] == ["low", "low"]


def test_program_b_never_hops_to_another_tier(cross_kit):
    cross_kit.client.script(MID, draft("print(42)"))
    cross_kit.client.script(CODE, error_result("http_5xx"))
    cross_kit.sandbox_runner.script("run", program_output("42\n"))

    result = ask(cross_kit)

    assert result.execution_status == "one_sided"
    assert cross_kit.client.models_called() == [MID, CODE], "code has on_error: mid, and it is not taken"


# ---- an option, off by default ----


def test_the_cross_check_is_off_by_default():
    config = make_execute_config()
    assert (config.execute.cross_check, config.execute.cross_check_tier) == (False, None)
    assert config.execute.cross_check_parallel is True and config.execute.cross_check_think is None
    kit = HarnessKit(config)
    kit.client.script(MID, draft())
    kit.sandbox_runner.script("run", program_output("42\n"))

    result = ask(kit)

    assert (result.execution_status, result.cross_check, result.cross_check_tier) == ("executed", False, None)
    assert kit.run_record.cross_check is False and kit.sandbox_records[0].program_label == "a"


def test_the_shipped_config_leaves_the_cross_check_off():
    from inference_arbiter.config import load_config
    from conftest import REAL_CONFIG_PATH

    shipped = load_config(REAL_CONFIG_PATH).execute
    assert shipped.cross_check is False
    assert shipped.cross_check_tier in (None, "mid", "code", "local"), "program B's tier is a free tier when set"


def test_a_request_turns_the_cross_check_on_and_names_program_bs_tier():
    kit = HarnessKit(make_execute_config({"cross_check_parallel": False}))
    kit.client.script(MID, draft())
    kit.client.script("local-model", draft())
    kit.sandbox_runner.script("run", program_output("42\n"), program_output("42\n"))

    result = ask(kit, cross_check_tier="local")

    assert (result.execution_status, result.cross_check_tier) == ("verified", "local")


def test_a_request_can_turn_a_configured_cross_check_off(cross_kit):
    cross_kit.client.script(MID, draft())
    cross_kit.sandbox_runner.script("run", program_output("42\n"))

    result = ask(cross_kit, cross_check=False)

    assert (result.execution_status, result.cross_check) == ("executed", False)
    assert cross_kit.client.models_called() == [MID]


def test_the_tier_flag_moves_program_a_and_leaves_program_b(cross_kit):
    cross_kit.client.script("local-model", draft())
    cross_kit.client.script(CODE, draft())
    cross_kit.sandbox_runner.script("run", program_output("42\n"), program_output("42\n"))

    result = ask(cross_kit, tier="local")

    assert (result.selected_tier, result.cross_check_tier, result.execution_status) == ("local", "code", "verified")


def test_an_unknown_cross_check_tier_is_refused_before_anything_runs(cross_kit):
    with pytest.raises(UnknownTierError, match="cross-check tier 'triage'"):
        ask(cross_kit, cross_check_tier="triage")
    assert cross_kit.client.calls == [] and cross_kit.sandbox_runner.commands == []


def test_config_refuses_a_cross_check_tier_that_is_not_an_answer_tier():
    with pytest.raises(ValidationError, match="execute.cross_check_tier 'triage' is not an answer tier"):
        make_execute_config({"cross_check_tier": "triage"})


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "route", "cross_check": True},
        {"mode": "auto", "cross_check_tier": "code"},
        {"mode": "execute", "cross_check": False, "cross_check_tier": "code"},
    ],
)
def test_cross_check_request_options_are_validated(overrides):
    with pytest.raises(ValidationError):
        AskRequest.from_question("q", **overrides)


def test_auto_mode_never_cross_checks(cross_kit):
    cross_kit.client.script(MID, ok_result("plain"))

    result = cross_kit.harness.answer(AskRequest.from_question("What is 6 times 7?"))

    assert (result.mode, result.execution_status, result.cross_check) == ("route", "skipped", False)
    assert cross_kit.sandbox_runner.commands == []


# ---- CLI ----


def parse_request_mode(argv: list[str]) -> str:
    return cli._mode_from_arguments(cli.build_parser().parse_args(argv))


def test_cli_cross_check_flags_select_execute_mode():
    assert parse_request_mode(["ask", "q", "--cross-check"]) == "execute"
    assert parse_request_mode(["ask", "q", "--cross-check-tier", "code"]) == "execute"
    assert parse_request_mode(["ask", "q", "--execute", "--cross-check"]) == "execute"


@pytest.mark.parametrize("extra", [["--chain"], ["--route"], ["--sources", "a.md"]])
def test_cli_refuses_cross_check_with_another_mode(extra, capsys):
    assert cli.main(["ask", "q", "--cross-check", *extra]) == cli.EXIT_USAGE
    assert "--execute and --cross-check cannot be combined" in capsys.readouterr().err


def test_cli_trace_shows_both_programs(cross_kit):
    cross_kit.client.script(MID, draft("print(25775)"))
    cross_kit.client.script(CODE, draft("print(7466)"))
    cross_kit.sandbox_runner.script("run", program_output("25775\n"), program_output("7466\n"))

    trace = cli.format_trace(ask(cross_kit))

    assert "execution     disputed, program A (mid) printed '25775', program B (code) printed '7466'" in trace
    assert "compared as integer" in trace
    assert "a first" in trace and "b first" in trace
