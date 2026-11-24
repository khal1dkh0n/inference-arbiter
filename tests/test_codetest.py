"""Codetest mode: the driver, the result reading, the decision table and the control flow.

The driver tests run the bundle with the host's own interpreter on small fixed sources
written here. Everything else uses fakes: no sandbox is started and no model is called.
"""

from __future__ import annotations

import ast
import inspect
import itertools
import json
import sqlite3

import pytest
from pydantic import ValidationError

from inference_arbiter import cli, codetest, codetest_driver, sandbox
from inference_arbiter.config import HarnessConfig
from inference_arbiter.router import CodeTestNotConfiguredError, Harness, UnknownTierError
from inference_arbiter.schemas import AskRequest, CodeInterface
from inference_arbiter.telemetry import Telemetry

from conftest import (
    BASE_CONFIG,
    FakeClient,
    FakeSandboxRunner,
    HarnessKit,
    error_result,
    make_execute_config,
    ok_result,
    program_output,
    run_on_host,
    sandbox_result,
)

LOCAL = "local-model"
MID = "mid-model"
CODE = "code-model"
QUESTION = "Write a Python function add(a, b) that returns the sum of two integers."

GOOD_SOLUTION = "def add(a, b):\n    return a + b\n"
WRONG_SOLUTION = "def add(a, b):\n    return a - b\n"
GOOD_TESTS = '''import unittest
from solution import add


class AddTests(unittest.TestCase):
    def test_small(self):
        self.assertEqual(add(2, 3), 5)

    def test_negative(self):
        self.assertEqual(add(-2, -3), -5)

    def test_zero(self):
        self.assertEqual(add(0, 0), 0)
'''


# --- the driver, run for real on the host interpreter ---


def test_a_right_solution_passes_and_the_stub_run_does_not(tmp_path):
    result = run_on_host(tmp_path, GOOD_SOLUTION, GOOD_TESTS)

    run_class, reason, summary = codetest.classify_test_run(result)
    assert (run_class, reason) == ("passed", None)
    assert result.exit_code == 0
    assert (summary.tests_run, summary.failed, summary.stub_passed) == (3, 0, False)
    assert result.stdout.count("\n") == 1, "the result line is the only line on stdout"


def test_a_wrong_solution_fails_and_the_failing_tests_are_named_with_their_lines(tmp_path):
    result = run_on_host(tmp_path, WRONG_SOLUTION, GOOD_TESTS)

    run_class, _, summary = codetest.classify_test_run(result)
    assert run_class == "tests_failed"
    assert result.exit_code == codetest_driver.EXIT_TESTS_FAILED
    assert (summary.ran, summary.failed) == (3, 2)
    assert any("test_small" in name for name in summary.failed_tests)
    assert "self.assertEqual(add(2, 3), 5)" in summary.detail, "tracebacks show the test source line"
    assert summary.stub_passed is None, "the stub run is only made after a clean run"


@pytest.mark.parametrize(
    "solution, stage",
    [
        ("def add(a, b)\n    return a + b\n", "solution_import"),
        ("import no_such_module_anywhere\n\ndef add(a, b):\n    return a + b\n", "solution_import"),
        ("raise SystemExit(0)\n", "solution_import"),
        ("def plus(a, b):\n    return a + b\n", "solution_missing_function"),
        ("add = 3\n", "solution_missing_function"),
    ],
)
def test_a_solution_that_does_not_load_is_solution_invalid(tmp_path, solution, stage):
    result = run_on_host(tmp_path, solution, GOOD_TESTS)

    run_class, reason, _ = codetest.classify_test_run(result)
    assert (run_class, reason) == ("solution_invalid", stage)
    assert result.exit_code == codetest_driver.EXIT_SOLUTION_INVALID


@pytest.mark.parametrize(
    "tests, stage",
    [
        ("import unittest\nfrom solution import add\nclass T(unittest.TestCase)\n    pass\n", "tests_import"),
        ("import unittest\nfrom solution import subtract\n", "tests_import"),
        ("import unittest\nfrom solution import add\n", "no_tests"),
        (
            "import unittest\nfrom solution import add\n\nclass T(unittest.TestCase):\n"
            "    @unittest.skip('later')\n    def test_a(self):\n        self.assertEqual(add(1, 1), 3)\n",
            "no_tests",
        ),
    ],
)
def test_tests_that_cannot_run_are_tests_invalid(tmp_path, tests, stage):
    result = run_on_host(tmp_path, GOOD_SOLUTION, tests)

    run_class, reason, _ = codetest.classify_test_run(result)
    assert (run_class, reason) == ("tests_invalid", stage)
    assert result.exit_code == codetest_driver.EXIT_TESTS_INVALID


@pytest.mark.parametrize(
    "test_body",
    [
        "self.assertTrue(True)",
        "self.assertEqual(1 + 1, 2)",
        "with self.assertRaises(Exception):\n            add(None, 1)",
        "try:\n            add(1, 2)\n        except Exception:\n            pass",
    ],
)
def test_tests_that_also_pass_against_the_raising_stub_are_vacuous(tmp_path, test_body):
    tests = (
        "import unittest\nfrom solution import add\n\nclass T(unittest.TestCase):\n"
        f"    def test_a(self):\n        {test_body}\n"
    )
    result = run_on_host(tmp_path, GOOD_SOLUTION, tests)

    run_class, _, summary = codetest.classify_test_run(result)
    assert run_class == "vacuous"
    assert result.exit_code == codetest_driver.EXIT_VACUOUS
    assert summary.stub_passed is True


def test_what_the_solution_and_the_tests_print_never_reaches_stdout(tmp_path):
    forged = codetest.RESULT_MARKER + '{"stage": "ran"}'
    solution = f"print({forged!r})\n" + WRONG_SOLUTION
    tests = GOOD_TESTS + f"\nprint({forged!r})\n"
    result = run_on_host(tmp_path, solution, tests)

    assert codetest.classify_test_run(result)[0] == "tests_failed"
    assert result.stdout.count(codetest.RESULT_MARKER) == 1
    assert forged in result.stderr, "captured output is kept on stderr for the audit row"


@pytest.mark.parametrize("ending", ["import os\nos._exit(0)\n", "import os\nos._exit(7)\n"])
def test_a_solution_that_ends_the_process_is_not_a_pass(tmp_path, ending):
    result = run_on_host(tmp_path, ending + GOOD_SOLUTION, GOOD_TESTS)

    assert codetest.classify_test_run(result)[:2] == ("not_finished", codetest.REASON_NO_RESULT)


def test_sources_with_quotes_backslashes_and_non_ascii_text_survive_the_bundle(tmp_path):
    awkward = "caf\u00e9 \U0001f600 ''' \"\"\" \\ \\n {} %s"
    solution = f"TEXT = {awkward!r}\n\ndef add(a, b):\n    return TEXT\n"
    tests = (
        "import unittest\nfrom solution import add\n\nclass T(unittest.TestCase):\n"
        f"    def test_a(self):\n        self.assertEqual(add(1, 2), {awkward!r})\n"
    )
    bundle = codetest.build_bundle(solution, tests, "add")

    assert bundle.isascii()
    header = ast.parse(bundle).body[:3]
    assert [ast.literal_eval(statement.value) for statement in header] == [solution, tests, "add"]
    assert codetest.classify_test_run(run_on_host(tmp_path, solution, tests))[0] == "passed"


def test_the_bundle_is_the_three_literals_followed_by_the_driver_file():
    bundle = codetest.build_bundle(GOOD_SOLUTION, GOOD_TESTS, "add")

    assert bundle.endswith(codetest.DRIVER_SOURCE)
    assert bundle[: -len(codetest.DRIVER_SOURCE)].count("\n") == 3
    assert "import subprocess" not in codetest.DRIVER_SOURCE and "open(" not in codetest.DRIVER_SOURCE


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({"stage": "solution_import"}, codetest_driver.EXIT_SOLUTION_INVALID),
        ({"stage": "solution_missing_function"}, codetest_driver.EXIT_SOLUTION_INVALID),
        ({"stage": "tests_import"}, codetest_driver.EXIT_TESTS_INVALID),
        ({"stage": "tests_aborted"}, codetest_driver.EXIT_TESTS_INVALID),
        ({"stage": "no_tests"}, codetest_driver.EXIT_TESTS_INVALID),
        ({"stage": "ran", "failures": 1}, codetest_driver.EXIT_TESTS_FAILED),
        ({"stage": "ran", "errors": 1}, codetest_driver.EXIT_TESTS_FAILED),
        ({"stage": "ran", "unexpected_successes": 1}, codetest_driver.EXIT_TESTS_FAILED),
        ({"stage": "ran", "failures": 1, "stub_passed": True}, codetest_driver.EXIT_TESTS_FAILED),
        ({"stage": "ran", "stub_passed": True}, codetest_driver.EXIT_VACUOUS),
        ({"stage": "ran", "stub_passed": False}, codetest_driver.EXIT_PASSED),
    ],
)
def test_exit_code_for(fields, expected):
    summary = codetest_driver.blank_summary(fields["stage"])
    summary.update(fields)
    assert codetest_driver.exit_code_for(summary) == expected


def test_no_driver_exit_code_collides_with_a_code_the_gate_gives_a_meaning():
    codes = {
        codetest_driver.EXIT_TESTS_FAILED, codetest_driver.EXIT_SOLUTION_INVALID,
        codetest_driver.EXIT_TESTS_INVALID, codetest_driver.EXIT_VACUOUS,
    }
    assert len(codes) == 4
    assert not codes & {0, 1, 2, sandbox.EXIT_TIMEOUT_TERM, sandbox.EXIT_SBX_COULD_NOT_RUN, sandbox.EXIT_KILLED}


# --- reading the result line ---


def result_line(stage: str = "ran", tests_run: int = 3, **fields) -> str:
    """A result line as the driver prints it: the stub run is reported only after a clean run."""
    summary = codetest_driver.blank_summary(stage)
    clean_run = stage == "ran" and not any(
        fields.get(name) for name in ("failures", "errors", "unexpected_successes")
    )
    summary.update(tests_run=tests_run if stage == "ran" else 0, stub_passed=False if clean_run else None)
    summary.update(fields)
    return codetest.RESULT_MARKER + json.dumps(summary)


def exit_code_of(line: str) -> int:
    return codetest_driver.exit_code_for(json.loads(line[len(codetest.RESULT_MARKER):]))


def run_output(stage: str = "ran", **fields):
    """What the gate's run command returns for a driver run with this result line."""
    line = result_line(stage, **fields)
    return program_output(stdout=line + "\n", exit_code=exit_code_of(line))


FAILED_FIELDS = {"failures": 2, "failed_tests": ["test_small (test_solution.AddTests.test_small)", "test_zero"]}


def test_parse_summary_reads_the_last_non_empty_line():
    summary = codetest.parse_summary("\n" + result_line(failures=1, skipped=1, failed_tests=["t"]) + "\n\n")

    assert (summary.stage, summary.tests_run, summary.ran, summary.failed) == ("ran", 3, 2, 1)
    assert summary.failed_tests == ("t",)


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "3 tests passed\n",
        codetest.RESULT_MARKER + "not json\n",
        codetest.RESULT_MARKER + "[1, 2]\n",
        result_line() + "\nsomething printed after the result line\n",
        "prefix " + result_line() + "\n",
        result_line(stage="made_up") + "\n",
        result_line(tests_run=-1) + "\n",
        result_line(failures="2") + "\n",
        result_line(errors=True) + "\n",
        result_line(failed_tests="test_a") + "\n",
        result_line(failed_tests=[1]) + "\n",
        result_line(detail=None) + "\n",
        result_line(stub_passed="no") + "\n",
        codetest.RESULT_MARKER + '{"stage": "ran"}\n',
    ],
)
def test_parse_summary_rejects_anything_that_is_not_a_well_formed_result_line(stdout):
    assert codetest.parse_summary(stdout) is None


@pytest.mark.parametrize(
    "result, expected",
    [
        # 1: the gate or the sandbox failed
        (sandbox_result(sandbox.ERROR_SANDBOX_UNAVAILABLE, exit_code=None), ("infra_failed", "sandbox_unavailable")),
        (sandbox_result(sandbox.ERROR_NETWORK_POLICY_NOT_APPLIED, exit_code=None), ("infra_failed", "network_policy_not_applied")),
        (sandbox_result(sandbox.ERROR_PROGRAM_TOO_LARGE, exit_code=None), ("infra_failed", "program_too_large")),
        (sandbox_result(sandbox.ERROR_RUNNER_NOT_ALLOWED, exit_code=None), ("infra_failed", "runner_not_allowed")),
        # 2: the run was cut
        (sandbox_result(sandbox.ERROR_TIMEOUT, result_line(), 124, timed_out=True), ("not_finished", "timeout")),
        (sandbox_result(sandbox.ERROR_OUTPUT_TOO_LARGE, result_line(), None), ("not_finished", "output_too_large")),
        # 3: no result line
        (sandbox_result(None, "", 0), ("not_finished", "no_result")),
        (sandbox_result(sandbox.ERROR_NONZERO_EXIT, "Traceback ...", 1), ("not_finished", "no_result")),
        # 4: the exit code and the result line disagree
        (sandbox_result(None, result_line(**FAILED_FIELDS), 0), ("not_finished", "result_mismatch")),
        (sandbox_result(sandbox.ERROR_NONZERO_EXIT, result_line(), 10), ("not_finished", "result_mismatch")),
        (sandbox_result(sandbox.ERROR_NONZERO_EXIT, result_line(), 1), ("not_finished", "result_mismatch")),
        (sandbox_result(None, result_line("solution_import"), 0), ("not_finished", "result_mismatch")),
        (sandbox_result(None, result_line(stub_passed=True), 0), ("not_finished", "result_mismatch")),
        (sandbox_result(None, result_line(stub_passed=None), 0), ("not_finished", "result_mismatch")),
        (sandbox_result(None, result_line(tests_run=2, skipped=2), 0), ("not_finished", "result_mismatch")),
        # 5 to 9: a result line the exit code agrees with
        (sandbox_result(sandbox.ERROR_NONZERO_EXIT, result_line("solution_import"), 11), ("solution_invalid", "solution_import")),
        (sandbox_result(sandbox.ERROR_NONZERO_EXIT, result_line("solution_missing_function"), 11), ("solution_invalid", "solution_missing_function")),
        (sandbox_result(sandbox.ERROR_NONZERO_EXIT, result_line("tests_import"), 12), ("tests_invalid", "tests_import")),
        (sandbox_result(sandbox.ERROR_NONZERO_EXIT, result_line("tests_aborted"), 12), ("tests_invalid", "tests_aborted")),
        (sandbox_result(sandbox.ERROR_NONZERO_EXIT, result_line("no_tests"), 12), ("tests_invalid", "no_tests")),
        (sandbox_result(sandbox.ERROR_NONZERO_EXIT, result_line(**FAILED_FIELDS), 10), ("tests_failed", None)),
        (sandbox_result(sandbox.ERROR_NONZERO_EXIT, result_line(errors=1), 10), ("tests_failed", None)),
        (sandbox_result(sandbox.ERROR_NONZERO_EXIT, result_line(unexpected_successes=1), 10), ("tests_failed", None)),
        (sandbox_result(sandbox.ERROR_NONZERO_EXIT, result_line(stub_passed=True), 13), ("vacuous", None)),
        (sandbox_result(None, result_line(), 0), ("passed", None)),
        (sandbox_result(None, result_line(tests_run=4, skipped=1), 0), ("passed", None)),
    ],
)
def test_classify_test_run(result, expected):
    run_class, reason, summary = codetest.classify_test_run(result)

    assert (run_class, reason) == expected
    assert (summary is None) == (run_class in ("infra_failed", "not_finished"))


def test_every_gate_error_type_has_a_class():
    for error_type in sandbox.INFRASTRUCTURE_ERRORS:
        assert codetest.classify_test_run(sandbox_result(error_type, exit_code=None))[0] == "infra_failed"
    assert set(sandbox.PROGRAM_ERRORS) == {
        sandbox.ERROR_TIMEOUT, sandbox.ERROR_OUTPUT_TOO_LARGE, sandbox.ERROR_NONZERO_EXIT
    }, "a new program error needs a row in classify_test_run"


# --- the decision table, every combination ---

RUN_CLASSES = ("passed", "vacuous", "tests_failed", "solution_invalid", "tests_invalid", "not_finished", "infra_failed")
REPAIR_STATES = ("not_attempted", *RUN_CLASSES, "draft_failed")
UNTESTABLE = ("vacuous", "tests_invalid", "not_finished", "infra_failed")


def expected_decision(first: str, repair: str, repairs_allowed: bool) -> tuple[str, str | None, str | None]:
    """The table written out a second time, row by row, as (action, status, whose solution)."""
    if first == "passed":
        return ("return", "tests_passed", "first")
    if first in UNTESTABLE:
        return ("return", "untested", "first")
    if repair == "not_attempted":
        if repairs_allowed:
            return ("repair", None, None)
        return ("return", "tests_failed", "first") if first == "tests_failed" else ("return", "failed", None)
    by_repair = {
        "passed": ("return", "tests_passed", "repair"),
        "tests_failed": ("return", "tests_failed", "repair"),
        "vacuous": ("return", "untested", "repair"),
    }
    if repair in by_repair:
        return by_repair[repair]
    if first == "tests_failed":
        return ("return", "tests_failed", "first")
    if repair in ("solution_invalid", "draft_failed"):
        return ("return", "failed", None)
    return ("return", "untested", "repair")


@pytest.mark.parametrize(
    "first, repair, repairs_allowed", list(itertools.product(RUN_CLASSES, REPAIR_STATES, (True, False)))
)
def test_decide_full_table(first, repair, repairs_allowed):
    decision = codetest.decide(first, repair, repairs_allowed)

    assert (decision.action, decision.status, decision.use) == expected_decision(first, repair, repairs_allowed)


def test_the_table_covers_every_class_and_never_says_verified():
    assert set(RUN_CLASSES) == set(codetest.RunClass.__args__)
    assert set(REPAIR_STATES) == set(codetest.RepairState.__args__)
    statuses = {
        codetest.decide(first, repair, allowed).status
        for first, repair, allowed in itertools.product(RUN_CLASSES, REPAIR_STATES, (True, False))
    }
    assert statuses == {None, "tests_passed", "tests_failed", "untested", "failed"}


def test_only_a_solution_fault_is_repaired():
    for first in RUN_CLASSES:
        action = codetest.decide(first, "not_attempted", True).action
        assert (action == "repair") == (first in ("tests_failed", "solution_invalid"))


# --- the interface check ---


def interface_of(question: str, function_name: str, signature: str):
    return codetest.check_interface(question, CodeInterface(function_name=function_name, signature=signature))


@pytest.mark.parametrize(
    "signature, canonical",
    [
        ("def add(a, b):", "def add(a, b)"),
        ("def add(a: int, b: int = 0) -> int", "def add(a: int, b: int=0) -> int"),
        ("  def add(a, *rest, key=None) -> list[int]:  ", "def add(a, *rest, key=None) -> list[int]"),
    ],
)
def test_a_valid_signature_is_returned_in_one_canonical_form(signature, canonical):
    interface, error_type, _ = interface_of("Write add.", "add", signature)

    assert error_type is None
    assert (interface.function_name, interface.signature) == ("add", canonical)


@pytest.mark.parametrize(
    "function_name, signature",
    [
        ("add", "add(a, b)"),
        ("add", "def add(a, b): return a + b\nimport os"),
        ("add", "def plus(a, b):"),
        ("add", "async def add(a, b):"),
        ("add", "class add:"),
        ("add", "@cache\ndef add(a, b):"),
        ("add", "def add(a, b"),
        ("class", "def class(a):"),
        ("2fast", "def 2fast(a):"),
        ("add two", "def add two(a):"),
    ],
)
def test_an_unusable_interface_is_rejected(function_name, signature):
    interface, error_type, message = interface_of("Write add.", function_name, signature)

    assert interface is None and error_type == codetest.ERROR_INTERFACE_INVALID and message


def test_a_function_the_question_defines_cannot_be_renamed():
    question = "Complete this:\n\ndef luhn_check_digit(payload: str) -> int:\n    ..."

    kept, _, _ = interface_of(question, "luhn_check_digit", "def luhn_check_digit(payload: str) -> int:")
    renamed, error_type, message = interface_of(question, "check_digit", "def check_digit(payload: str) -> int:")

    assert kept.function_name == "luhn_check_digit"
    assert renamed is None and error_type == codetest.ERROR_INTERFACE_MISMATCH
    assert "luhn_check_digit" in message


# --- messages ---

INTERFACE = codetest.Interface("add", "def add(a: int, b: int) -> int")
# Built once, here, so a test that swaps the builder cannot change what the leak check expects.
HONEST_TESTS_PROMPT = codetest.build_tests_messages(QUESTION, INTERFACE)


def test_the_test_writer_is_given_the_question_and_the_interface_and_has_no_other_input():
    assert list(inspect.signature(codetest.build_tests_messages).parameters) == ["question", "interface"]
    assert codetest.build_tests_messages(QUESTION, INTERFACE) == [
        {"role": "system", "content": codetest.TESTS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question:\n{QUESTION}\n\nRequired interface (module `solution`):\ndef add(a: int, b: int) -> int:",
        },
    ]


def test_solution_and_tests_get_the_same_task_text():
    solution_messages = codetest.build_solution_messages(QUESTION, INTERFACE)
    tests_messages = codetest.build_tests_messages(QUESTION, INTERFACE)

    assert solution_messages[1] == tests_messages[1]
    assert solution_messages[0]["content"] == codetest.SOLUTION_SYSTEM_PROMPT


def test_no_prompt_promises_more_than_the_mode_delivers():
    for prompt in (
        codetest.INTERFACE_SYSTEM_PROMPT, codetest.SOLUTION_SYSTEM_PROMPT,
        codetest.TESTS_SYSTEM_PROMPT, codetest.REPAIR_SYSTEM_PROMPT,
    ):
        assert prompt.isascii()
    assert "not shown their code" in codetest.TESTS_SYSTEM_PROMPT


# --- control flow through the harness, with fakes ---


def codetest_config(execute_overrides: dict | None = None, **code_test_overrides) -> HarnessConfig:
    """Interface on local, solution on mid, tests on code: three models, so scripted results
    line up. One after the other unless a test asks for parallel."""
    settings = {"interface_tier": "local", "solution_tier": "mid", "tests_tier": "code", "parallel": False}
    settings.update(code_test_overrides)
    return make_execute_config(execute_overrides, code_test=settings)


@pytest.fixture
def code_kit() -> HarnessKit:
    return HarnessKit(codetest_config())


def ask(kit: HarnessKit, question: str = QUESTION, **overrides):
    return kit.harness.answer(AskRequest.from_question(question, mode="codetest", **overrides))


def interface_json(function_name: str = "add", signature: str = "def add(a: int, b: int) -> int:"):
    return ok_result(json.dumps({"function_name": function_name, "signature": signature}))


def solution_json(solution: str = GOOD_SOLUTION, working: str = "add the two"):
    return ok_result(json.dumps({"working": working, "solution": solution}))


def unit_tests_json(tests: str = GOOD_TESTS, working: str = "three sums"):
    return ok_result(json.dumps({"working": working, "tests": tests}))


def script_drafts(kit: HarnessKit, *solutions, tests=None) -> None:
    kit.client.script(LOCAL, interface_json())
    kit.client.script(MID, *(solutions or (solution_json(),)))
    kit.client.script(CODE, tests if tests is not None else unit_tests_json())


def test_passed_run_returns_the_solution_with_the_honest_wording(code_kit):
    script_drafts(code_kit)
    code_kit.sandbox_runner.script("run", run_output())

    result = ask(code_kit)

    assert (result.status, result.mode, result.execution_status) == ("ok", "codetest", "tests_passed")
    assert result.answer.startswith(
        "Passed its generated tests: 3 of 3 unit tests, written by tier code, model code-model from the "
        "question alone, passed in an isolated sandbox. The tests were written by a model and were not "
        "checked against a reference, so this does not show the solution is correct."
    )
    assert result.answer.endswith(f"```python\n{GOOD_SOLUTION.rstrip()}\n```")
    assert "verified" not in result.answer.lower()
    assert (result.tests_run, result.tests_failed, result.repaired) == (3, 0, False)
    assert (result.code_function, result.code_signature) == ("add", "def add(a: int, b: int) -> int")
    assert (result.code_solution, result.code_first_solution, result.code_tests) == (
        GOOD_SOLUTION, GOOD_SOLUTION, GOOD_TESTS
    )
    assert (result.selected_tier, result.final_tier, result.tests_tier) == ("mid", "mid", "code")
    assert (result.route_reason, result.triage_status) == ("configured", "skipped")
    assert [(call.role, call.tier) for call in result.calls] == [
        ("interface", "local"), ("draft", "mid"), ("tests", "code"),
    ]
    assert result.code_first_outcome == "passed"


def test_the_one_program_the_gate_runs_is_the_driver_with_both_sources(code_kit):
    script_drafts(code_kit)
    code_kit.sandbox_runner.script("run", run_output())

    ask(code_kit)

    assert code_kit.sandbox_runner.verbs() == ["create", "check", "stage", "run", "rm"]
    staged = code_kit.sandbox_runner.stdin_of("stage")
    assert staged == codetest.build_bundle(GOOD_SOLUTION, GOOD_TESTS, "add")
    assert "python3" in code_kit.sandbox_runner.argv_of("run"), "the gate's own run command, unchanged"


def test_the_run_and_its_sandbox_row_are_recorded(code_kit):
    script_drafts(code_kit)
    code_kit.sandbox_runner.script("run", run_output())

    result = ask(code_kit)

    run = code_kit.run_record
    assert (run.mode, run.execution_status, run.status, run.final_tier) == ("codetest", "tests_passed", "ok", "mid")
    assert [(call.role, call.tier, call.status) for call in code_kit.call_records] == [
        ("interface", "local", "ok"), ("draft", "mid", "ok"), ("tests", "code", "ok"),
    ]
    (row,) = code_kit.sandbox_records
    assert (row.attempt, row.outcome, row.exit_code, row.sandbox_removed) == ("first", "passed", 0, True)
    assert row.program == code_kit.sandbox_runner.stdin_of("stage")
    assert json.loads(row.executed_answer)["tests_run"] == 3
    assert row.claimed_answer is None
    assert [(summary.attempt, summary.outcome) for summary in result.sandbox_runs] == [("first", "passed")]


@pytest.mark.parametrize(
    "run, status, first_outcome, note_start",
    [
        (run_output(stub_passed=True), "untested", "vacuous",
         "NOT TESTED: the generated tests also pass against a stub that raises on every call"),
        (run_output("tests_import"), "untested", "tests_invalid",
         "NOT TESTED: the generated tests could not be run (tests_import)"),
        (run_output("no_tests"), "untested", "tests_invalid",
         "NOT TESTED: the generated tests could not be run (no_tests)"),
        (program_output(stdout="start\n", exit_code=124), "untested", "not_finished",
         "NOT TESTED: the test run did not finish (timeout)"),
        (program_output(stdout="x", exit_code=None, stdout_truncated=True), "untested", "not_finished",
         "NOT TESTED: the test run did not finish (output_too_large)"),
        (program_output(stdout="", exit_code=0), "untested", "not_finished",
         "NOT TESTED: the test run did not finish (no_result)"),
        (program_output(stdout=result_line() + "\n", exit_code=10), "untested", "not_finished",
         "NOT TESTED: the test run did not finish (result_mismatch)"),
        (program_output(exit_code=125), "untested", "infra_failed",
         "NOT TESTED: the sandbox could not run the tests (sandbox_unavailable)"),
    ],
)
def test_a_run_that_tested_nothing_is_untested_and_is_not_repaired(code_kit, run, status, first_outcome, note_start):
    script_drafts(code_kit)
    code_kit.sandbox_runner.script("run", run)

    result = ask(code_kit)

    assert (result.status, result.execution_status, result.code_first_outcome) == ("ok", status, first_outcome)
    assert result.answer.startswith(note_start)
    assert "was not run against any test that could fail it" in result.answer
    assert result.answer.endswith(f"```python\n{GOOD_SOLUTION.rstrip()}\n```")
    assert [call.role for call in result.calls] == ["interface", "draft", "tests"], "no repair call"
    assert code_kit.sandbox_runner.verbs().count("run") == 1


def test_a_gate_refusal_is_untested(code_kit):
    code_kit = HarnessKit(codetest_config(max_program_bytes=100))
    script_drafts(code_kit)

    result = ask(code_kit)

    assert (result.execution_status, result.code_first_outcome) == ("untested", "infra_failed")
    assert "program_too_large" in result.answer
    assert code_kit.sandbox_runner.verbs() == [], "no sandbox is created for a refused program"


def test_failed_tests_trigger_one_repair_and_a_passing_repair_is_returned(code_kit):
    script_drafts(code_kit, solution_json(WRONG_SOLUTION), solution_json(GOOD_SOLUTION))
    code_kit.sandbox_runner.script("run", run_output(**FAILED_FIELDS), run_output())

    result = ask(code_kit)

    assert (result.execution_status, result.repaired, result.code_first_outcome) == ("tests_passed", True, "tests_failed")
    assert (result.code_solution, result.code_first_solution) == (GOOD_SOLUTION, WRONG_SOLUTION)
    assert "The first solution failed 2 of 3 of these tests; this is the repaired solution." in result.answer
    assert [(call.role, call.tier) for call in result.calls] == [
        ("interface", "local"), ("draft", "mid"), ("tests", "code"), ("repair", "mid"),
    ]
    assert [(run.attempt, run.outcome) for run in result.sandbox_runs] == [("first", "tests_failed"), ("repair", "passed")]
    assert [row.outcome for row in code_kit.sandbox_records] == ["tests_failed", "passed"]
    assert code_kit.sandbox_runner.stdin_of("stage", 1) == codetest.build_bundle(GOOD_SOLUTION, GOOD_TESTS, "add")


def test_the_repair_call_is_shown_its_own_module_and_the_failing_tests(code_kit):
    detail = "AssertionError: -1 != 5"
    script_drafts(code_kit, solution_json(WRONG_SOLUTION), solution_json(GOOD_SOLUTION))
    code_kit.sandbox_runner.script("run", run_output(detail=detail, **FAILED_FIELDS), run_output())

    ask(code_kit)

    repair_call = code_kit.client.calls[-1]
    assert repair_call.model == MID
    assert repair_call.messages[0]["content"] == codetest.REPAIR_SYSTEM_PROMPT
    user_text = repair_call.messages[1]["content"]
    assert QUESTION in user_text and WRONG_SOLUTION in user_text and detail in user_text
    assert "2 of 3 tests failed: test_small (test_solution.AddTests.test_small), test_zero" in user_text


def test_a_repair_that_still_fails_is_returned_as_failed_with_the_tests_named(code_kit):
    script_drafts(code_kit, solution_json(WRONG_SOLUTION), solution_json("def add(a, b):\n    return 0\n"))
    code_kit.sandbox_runner.script("run", run_output(**FAILED_FIELDS), run_output(failures=1, failed_tests=["test_small"]))

    result = ask(code_kit)

    assert (result.status, result.execution_status, result.repaired) == ("ok", "tests_failed", True)
    assert (result.tests_run, result.tests_failed) == (3, 1)
    assert result.answer.startswith(
        "FAILED its generated tests: 1 of 3 unit tests, written by tier code, model code-model from the "
        "question alone, failed in an isolated sandbox (test_small)."
    )
    assert "a failure can be a wrong test as well as a wrong solution" in result.answer
    assert result.code_solution == "def add(a, b):\n    return 0\n"


@pytest.mark.parametrize(
    "repair_run",
    [run_output("solution_import"), run_output("tests_import"), program_output(stdout="", exit_code=124),
     program_output(exit_code=125)],
)
def test_after_failed_tests_a_repair_that_cannot_be_tested_leaves_the_first_solution(code_kit, repair_run):
    script_drafts(code_kit, solution_json(WRONG_SOLUTION), solution_json("def add(a, b):\n    return 1\n"))
    code_kit.sandbox_runner.script("run", run_output(**FAILED_FIELDS), repair_run)

    result = ask(code_kit)

    assert (result.execution_status, result.repaired) == ("tests_failed", False)
    assert (result.code_solution, result.tests_run, result.tests_failed) == (WRONG_SOLUTION, 3, 2)
    assert "One repair was attempted and gave nothing that could be tested" in result.answer
    assert len(result.sandbox_runs) == 2, "both runs are kept"


@pytest.mark.parametrize("failed_repair", [error_result("timeout"), ok_result("not json")])
def test_a_failed_repair_call_leaves_the_first_solution_and_is_recorded(code_kit, failed_repair):
    script_drafts(code_kit, solution_json(WRONG_SOLUTION), failed_repair)
    code_kit.sandbox_runner.script("run", run_output(**FAILED_FIELDS))

    result = ask(code_kit)

    assert (result.status, result.execution_status, result.repaired) == ("ok", "tests_failed", False)
    assert result.repair_error_type in ("timeout", codetest.ERROR_SOLUTION_SCHEMA_INVALID)
    assert result.error_type is None
    assert result.calls[-1].role == "repair" and result.calls[-1].status != "ok"


def test_a_solution_that_does_not_load_is_repaired_once(code_kit):
    script_drafts(code_kit, solution_json("def add(a, b)\n"), solution_json(GOOD_SOLUTION))
    code_kit.sandbox_runner.script("run", run_output("solution_import", detail="SyntaxError: expected ':'"), run_output())

    result = ask(code_kit)

    assert (result.execution_status, result.repaired, result.code_first_outcome) == ("tests_passed", True, "solution_invalid")
    assert "The first solution could not be loaded; this is the repaired solution." in result.answer
    repair_text = code_kit.client.calls[-1].messages[1]["content"]
    assert "importing it raised an error" in repair_text and "SyntaxError: expected ':'" in repair_text


@pytest.mark.parametrize("second", ["solution_missing_function", None])
def test_a_solution_that_never_loads_fails_closed_with_no_answer(code_kit, second):
    repair = solution_json("def plus(a, b):\n    return a + b\n") if second else error_result("http_5xx")
    script_drafts(code_kit, solution_json("def add(a, b)\n"), repair)
    runs = [run_output("solution_import")] + ([run_output(second)] if second else [])
    code_kit.sandbox_runner.script("run", *runs)

    result = ask(code_kit)

    assert (result.status, result.answer, result.execution_status) == ("error", None, "failed")
    assert result.error_type == codetest.ERROR_SOLUTION_INVALID
    assert result.error_message.startswith("FAILED: no answer. The solution could not be loaded, also after one repair")
    assert result.code_solution is None
    assert code_kit.run_record.status == "error" and code_kit.run_record.error_type == "solution_invalid"


def test_with_no_repair_allowed_the_first_result_stands():
    failing = HarnessKit(codetest_config(max_repairs=0))
    script_drafts(failing, solution_json(WRONG_SOLUTION))
    failing.sandbox_runner.script("run", run_output(**FAILED_FIELDS))
    broken = HarnessKit(codetest_config(max_repairs=0))
    script_drafts(broken, solution_json("def add(a, b)\n"))
    broken.sandbox_runner.script("run", run_output("solution_import"))

    failed_tests, never_loaded = ask(failing), ask(broken)

    assert (failed_tests.execution_status, len(failed_tests.calls)) == ("tests_failed", 3)
    assert (never_loaded.status, never_loaded.execution_status) == ("error", "failed")
    assert "and no repair was allowed" in never_loaded.error_message


@pytest.mark.parametrize(
    "interface_result, error_type",
    [
        (error_result("timeout"), "timeout"),
        (ok_result("no json here"), codetest.ERROR_INTERFACE_INVALID),
        (interface_json("add", "def plus(a, b):"), codetest.ERROR_INTERFACE_INVALID),
    ],
)
def test_a_failed_interface_call_fails_the_run_before_anything_is_written(code_kit, interface_result, error_type):
    code_kit.client.script(LOCAL, interface_result)

    result = ask(code_kit)

    assert (result.status, result.error_type, result.execution_status) == ("error", error_type, "skipped")
    assert code_kit.client.models_called() == [LOCAL]
    assert code_kit.sandbox_runner.verbs() == []
    assert code_kit.call_records[0].status in ("timeout", "invalid")


def test_an_interface_that_renames_the_questions_function_fails_closed(code_kit):
    code_kit.client.script(LOCAL, interface_json("total", "def total(a, b):"))

    result = ask(code_kit, question="Complete this function:\n\ndef add(a, b):\n    ...")

    assert (result.status, result.error_type) == ("error", codetest.ERROR_INTERFACE_MISMATCH)
    assert code_kit.client.models_called() == [LOCAL]


@pytest.mark.parametrize(
    "solution_result, error_type",
    [(error_result("http_5xx"), "http_5xx"), (ok_result('{"working": "x"}'), codetest.ERROR_SOLUTION_SCHEMA_INVALID)],
)
def test_without_a_solution_there_is_no_answer_and_no_tests_are_asked_for(code_kit, solution_result, error_type):
    code_kit.client.script(LOCAL, interface_json())
    code_kit.client.script(MID, solution_result)

    result = ask(code_kit)

    assert (result.status, result.answer, result.error_type) == ("error", None, error_type)
    assert code_kit.client.models_called() == [LOCAL, MID]
    assert code_kit.sandbox_runner.verbs() == []


@pytest.mark.parametrize(
    "tests_result, error_type",
    [(error_result("timeout"), "timeout"), (ok_result('{"tests": ""}'), codetest.ERROR_TESTS_SCHEMA_INVALID)],
)
def test_without_tests_the_solution_is_returned_as_untested(code_kit, tests_result, error_type):
    script_drafts(code_kit, tests=tests_result)

    result = ask(code_kit)

    assert (result.status, result.execution_status) == ("ok", "untested")
    assert result.answer.startswith(f"NOT TESTED: no tests could be written ({error_type}).")
    assert (result.code_tests, result.tests_run, result.code_first_outcome) == (None, None, None)
    assert code_kit.sandbox_runner.verbs() == []


def test_simulated_answer_error_fails_the_solution_call_without_calling_the_model(code_kit):
    code_kit.client.script(LOCAL, interface_json())

    result = ask(code_kit, simulate="answer-error")

    assert (result.status, result.error_type) == ("error", "simulated_error")
    assert code_kit.client.models_called() == [LOCAL]


def test_tier_and_tests_tier_can_be_set_on_the_request(code_kit):
    code_kit.client.script(LOCAL, interface_json(), solution_json())
    code_kit.client.script(MID, unit_tests_json())
    code_kit.sandbox_runner.script("run", run_output())

    result = ask(code_kit, tier="local", tests_tier="mid")

    assert (result.selected_tier, result.route_reason, result.tests_tier) == ("local", "forced", "mid")
    assert [(call.role, call.tier) for call in result.calls] == [
        ("interface", "local"), ("draft", "local"), ("tests", "mid"),
    ]
    assert "written by tier mid, model mid-model" in result.answer


def test_think_is_sent_with_the_solution_calls_only():
    kit = HarnessKit(codetest_config(think="medium"))
    script_drafts(kit, solution_json(WRONG_SOLUTION), solution_json(GOOD_SOLUTION))
    kit.sandbox_runner.script("run", run_output(**FAILED_FIELDS), run_output())

    ask(kit)

    assert [(call.model, call.think) for call in kit.client.calls] == [
        (LOCAL, None), (MID, "medium"), (CODE, None), (MID, "medium"),
    ]


def test_every_call_is_typed(code_kit):
    script_drafts(code_kit)
    code_kit.sandbox_runner.script("run", run_output())

    ask(code_kit)

    schemas = [call.json_schema for call in code_kit.client.calls]
    assert [sorted(schema["required"]) for schema in schemas] == [
        ["function_name", "signature"], ["solution", "working"], ["tests", "working"],
    ]


def test_solution_and_tests_can_be_written_at_the_same_time():
    kit = HarnessKit(codetest_config(parallel=True))
    script_drafts(kit)
    kit.sandbox_runner.script("run", run_output())

    result = ask(kit)

    assert result.execution_status == "tests_passed"
    assert [(call.role, call.tier) for call in result.calls] == [
        ("interface", "local"), ("draft", "mid"), ("tests", "code"),
    ], "rows keep a fixed order"
    assert_solution_never_reaches_the_test_writer(kit)


def test_unknown_tests_tier_and_missing_block_are_usage_errors(code_kit):
    with pytest.raises(UnknownTierError, match="tests tier 'nowhere'"):
        ask(code_kit, tests_tier="nowhere")
    with pytest.raises(CodeTestNotConfiguredError):
        ask(HarnessKit(make_execute_config()))


def test_request_options_match_the_mode():
    with pytest.raises(ValidationError, match="tests_tier needs mode 'codetest'"):
        AskRequest.from_question(QUESTION, mode="execute", tests_tier="mid")
    with pytest.raises(ValidationError, match="supports simulate 'answer-error' only"):
        AskRequest.from_question(QUESTION, mode="codetest", simulate="triage-timeout")
    with pytest.raises(ValidationError, match="sources need mode"):
        AskRequest.from_question(QUESTION, mode="codetest", sources=["a.txt"])


def test_a_codetest_run_is_stored_by_the_real_telemetry_without_a_schema_change(tmp_path):
    db_path = tmp_path / "telemetry.db"
    client, runner = FakeClient(), FakeSandboxRunner()
    harness = Harness(
        codetest_config(), {"ollama": client, "openrouter": client}, Telemetry(db_path), sandbox_runner=runner
    )
    client.script(LOCAL, interface_json())
    client.script(MID, solution_json(WRONG_SOLUTION), solution_json(GOOD_SOLUTION))
    client.script(CODE, unit_tests_json())
    runner.script("run", run_output(**FAILED_FIELDS), run_output())

    result = harness.answer(AskRequest.from_question(QUESTION, mode="codetest"))

    with sqlite3.connect(db_path) as connection:
        run_row = connection.execute(
            "SELECT mode, execution_status, repaired FROM runs WHERE trace_id = ?", (result.trace_id,)
        ).fetchone()
        roles = [row[0] for row in connection.execute("SELECT role FROM calls ORDER BY seq")]
        sandbox_rows = connection.execute(
            "SELECT attempt, outcome, exit_code, executed_answer FROM sandbox_runs ORDER BY seq"
        ).fetchall()
    assert run_row == ("codetest", "tests_passed", 1)
    assert roles == ["interface", "draft", "tests", "repair"]
    assert [row[:3] for row in sandbox_rows] == [("first", "tests_failed", 10), ("repair", "passed", 0)]
    assert json.loads(sandbox_rows[0][3])["failures"] == 2


# --- guardrail: the solution never reaches the test writer ---

SOLUTION_MARK = "SOLUTION-ONLY-MARKER-7f3a"


def marked_solution(label: str, body: str = WRONG_SOLUTION):
    """A solution draft whose every field carries the marker."""
    return solution_json(f"# {SOLUTION_MARK} solution {label}\n{body}", working=f"{SOLUTION_MARK} working {label}")


def marked_run(stage: str = "ran", **fields):
    """A sandbox result whose output carries the marker, as a traceback through the solution would."""
    run = run_output(stage, detail=f"{SOLUTION_MARK} traceback line", **fields)
    return program_output(stdout=run.stdout, stderr=f"{SOLUTION_MARK} stderr", exit_code=run.exit_code)


def messages_sent_to(kit: HarnessKit, model: str) -> str:
    return "\n".join(
        message["content"] for call in kit.client.calls if call.model == model for message in call.messages
    )


def assert_solution_never_reaches_the_test_writer(kit: HarnessKit) -> None:
    tests_calls = [
        call for call in kit.client.calls
        if call.model == CODE or call.messages[0]["content"] == codetest.TESTS_SYSTEM_PROMPT
    ]
    assert tests_calls, "the scenario made a tests call"
    for call in tests_calls:
        text = "\n".join(message["content"] for message in call.messages)
        assert SOLUTION_MARK not in text, "the solution's text reached a prompt for the test writer"
        assert call.messages == HONEST_TESTS_PROMPT, "the tests prompt is the question and the interface, nothing else"
    assert SOLUTION_MARK not in messages_sent_to(kit, LOCAL), "the solution's text reached the interface call"


LEAK_SCENARIOS = {
    # name: (solution drafts, sandbox runs)
    "passes": ([marked_solution("first", GOOD_SOLUTION)], [marked_run()]),
    "fails_then_passes": (
        [marked_solution("first"), marked_solution("repair", GOOD_SOLUTION)],
        [marked_run(**FAILED_FIELDS), marked_run()],
    ),
    "fails_twice": (
        [marked_solution("first"), marked_solution("repair")],
        [marked_run(**FAILED_FIELDS), marked_run(**FAILED_FIELDS)],
    ),
    "does_not_load_then_passes": (
        [marked_solution("first", "def add(a, b)\n"), marked_solution("repair", GOOD_SOLUTION)],
        [marked_run("solution_import"), marked_run()],
    ),
    "never_loads": (
        [marked_solution("first", "def add(a, b)\n"), marked_solution("repair", "def add(a, b)\n")],
        [marked_run("solution_import"), marked_run("solution_import")],
    ),
    "tests_do_not_load": ([marked_solution("first", GOOD_SOLUTION)], [marked_run("tests_import")]),
    "vacuous": ([marked_solution("first", GOOD_SOLUTION)], [marked_run(stub_passed=True)]),
}


def run_leak_scenario(kit: HarnessKit, scenario: str) -> None:
    drafts, runs = LEAK_SCENARIOS[scenario]
    kit.client.script(LOCAL, interface_json())
    kit.client.script(MID, *drafts)
    kit.client.script(CODE, unit_tests_json())
    kit.sandbox_runner.script("run", *runs)
    ask(kit)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("scenario", sorted(LEAK_SCENARIOS))
def test_nothing_of_the_solution_reaches_the_test_writer(scenario, parallel):
    kit = HarnessKit(codetest_config(parallel=parallel))

    run_leak_scenario(kit, scenario)

    assert_solution_never_reaches_the_test_writer(kit)
    repairs_expected = len(LEAK_SCENARIOS[scenario][0]) - 1
    repair_calls = [call for call in kit.client.calls if call.messages[0]["content"] == codetest.REPAIR_SYSTEM_PROMPT]
    assert len(repair_calls) == repairs_expected, "the scenario really went through the repair prompt"


def test_the_markers_are_real_the_repair_prompt_does_carry_the_solution(code_kit):
    """The other half of the leak test: the marked text exists and goes where it is allowed to go."""
    run_leak_scenario(code_kit, "fails_then_passes")

    to_solution_tier = messages_sent_to(code_kit, MID)
    assert f"{SOLUTION_MARK} solution first" in to_solution_tier
    assert f"{SOLUTION_MARK} traceback line" in to_solution_tier
    assert f"{SOLUTION_MARK} solution first" in code_kit.sandbox_runner.stdin_of("stage")


def test_the_leak_check_catches_a_leak(code_kit, monkeypatch):
    """The check is not vacuous: a tests prompt that quotes the solution fails it."""
    honest_builder = codetest.build_tests_messages

    def leaky_builder(question, interface):
        messages = honest_builder(question, interface)
        messages[-1]["content"] += f"\n\nThe solution being tested:\n# {SOLUTION_MARK} solution first"
        return messages

    monkeypatch.setattr(codetest, "build_tests_messages", leaky_builder)
    run_leak_scenario(code_kit, "passes")

    with pytest.raises(AssertionError, match="the solution's text reached a prompt for the test writer"):
        assert_solution_never_reaches_the_test_writer(code_kit)


def test_the_leak_check_also_catches_a_leak_that_keeps_the_marker_out(code_kit, monkeypatch):
    """Any change to the tests prompt fails the check, not only one that carries the marker."""
    honest_builder = codetest.build_tests_messages

    def padded_builder(question, interface):
        return honest_builder(question, interface) + [{"role": "user", "content": "One more hint."}]

    monkeypatch.setattr(codetest, "build_tests_messages", padded_builder)
    run_leak_scenario(code_kit, "passes")

    with pytest.raises(AssertionError):
        assert_solution_never_reaches_the_test_writer(code_kit)


def test_one_model_writing_both_still_gets_two_separate_prompts():
    kit = HarnessKit(codetest_config(tests_tier="mid"))
    kit.client.script(LOCAL, interface_json())
    kit.client.script(MID, marked_solution("first", GOOD_SOLUTION), unit_tests_json())
    kit.sandbox_runner.script("run", marked_run())

    result = ask(kit)

    assert result.execution_status == "tests_passed"
    assert_solution_never_reaches_the_test_writer(kit)
    assert "written by tier mid, model mid-model" in result.answer


# --- config and command line ---


def test_the_code_test_block_needs_the_execute_block_and_answer_tiers():
    raw = json.loads(json.dumps(BASE_CONFIG))
    raw["code_test"] = {}
    with pytest.raises(ValidationError, match="code_test needs an `execute:` block"):
        HarnessConfig.model_validate(raw)
    raw["execute"] = {"tier": "mid"}
    assert HarnessConfig.model_validate(raw).code_test.max_repairs == 1
    for key in ("interface_tier", "solution_tier", "tests_tier"):
        with pytest.raises(ValidationError, match=f"code_test.{key} 'triage' is not an answer tier"):
            HarnessConfig.model_validate({**raw, "code_test": {key: "triage"}})
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate({**raw, "code_test": {"max_repairs": 2}})


def parse(*arguments: str):
    return cli.build_parser().parse_args(["ask", QUESTION, *arguments])


def test_code_test_flags_select_the_mode():
    assert cli._mode_from_arguments(parse("--code-test")) == "codetest"
    assert cli._mode_from_arguments(parse("--tests-tier", "mid")) == "codetest"
    assert cli._mode_from_arguments(parse()) == "auto"


@pytest.mark.parametrize(
    "other", [["--execute"], ["--cross-check"], ["--chain"], ["--route"], ["--sources", "a.txt"], ["--verifier-tier", "verify"]]
)
def test_code_test_cannot_be_combined_with_another_mode(other, capsys):
    assert cli.run_ask(parse("--code-test", *other)) == cli.EXIT_USAGE
    assert "--code-test cannot be combined" in capsys.readouterr().err


def test_the_trace_shows_the_test_result_and_each_sandbox_run(code_kit):
    script_drafts(code_kit, solution_json(WRONG_SOLUTION), solution_json(GOOD_SOLUTION))
    code_kit.sandbox_runner.script("run", run_output(**FAILED_FIELDS), run_output())

    trace = cli.format_trace(ask(code_kit))

    assert "mode          codetest" in trace
    assert (
        "tests         tests_passed, 3 of 3 tests passed, function add, tests written by code, "
        "repaired: True, first run: tests_failed"
    ) in trace
    assert "sandbox 1     first " in trace and "tests_failed, exit 10" in trace
    assert "sandbox 2     repair" in trace and "passed, exit 0" in trace
