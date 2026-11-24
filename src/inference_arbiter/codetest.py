"""Phase 3 codetest mode: a solution and its unit tests, written apart, run in one sandbox.

Plain control flow with a fixed ceiling of four model calls and two sandbox runs:

1. Interface. One small typed call names the function and its signature from the question
   alone. `check_interface` checks it in code; an interface that fails the check fails the
   run. The module name is fixed: `solution`.
2. Solution. The solution tier gets the question and the interface and returns the module.
3. Tests. The tests tier gets the question and the interface and returns a `unittest`
   module. It is never shown the solution: `build_tests_messages` has no parameter that
   could carry it, and no function here passes anything of the solution, its run or its
   repair into a message for the tests tier. The solution is not shown the tests either.
4. One sandbox run of both through the dispatch gate (sandbox.py, unchanged): the gate
   stages one program, and that program is the host-written driver in codetest_driver.py
   with the two sources embedded as string literals. `classify_test_run` reads the result
   and `decide`, a pure function, says what to return.
5. At most one repair of the solution, when the tests failed it or it could not be loaded:
   the solution tier gets its own module and what happened, and the new module runs once
   against the same tests. The tests are never rewritten.

What the statuses mean. "tests_passed" says every generated test passed and the same tests
did not pass against a stub that raises on every call. The tests are written by a model and
are not checked against a reference, so this is not called verified and does not say the
solution is correct. "tests_failed" says at least one generated test failed; a wrong test
fails a right solution too. "untested" says the solution was not run against any test that
could fail it.

This module does not record telemetry. It returns every model call as a step and every
sandbox run as an attempt; the router turns those into rows.
"""

from __future__ import annotations

import ast
import json
import keyword
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ValidationError

from inference_arbiter import codetest_driver, sandbox
from inference_arbiter.chain import ChainStep, TierCaller
from inference_arbiter.schemas import (
    STATUS_ERROR,
    STATUS_OK,
    CallResult,
    CodeInterface,
    ExecutionStatus,
    SolutionDraft,
    UnitTestsDraft,
)

ROLE_INTERFACE = "interface"
ROLE_DRAFT = "draft"
ROLE_TESTS = "tests"
ROLE_REPAIR = "repair"

ERROR_INTERFACE_INVALID = "interface_invalid"
ERROR_INTERFACE_MISMATCH = "interface_mismatch"
ERROR_SOLUTION_SCHEMA_INVALID = "solution_schema_invalid"
ERROR_TESTS_SCHEMA_INVALID = "tests_schema_invalid"
# The solution could not be loaded, also after the one repair: there is nothing to return.
ERROR_SOLUTION_INVALID = "solution_invalid"

REASON_NO_RESULT = "no_result"
REASON_RESULT_MISMATCH = "result_mismatch"

MAX_VALIDATION_MESSAGE_CHARS = 300
# How much of a failed run the repair call is shown.
REPAIR_DETAIL_CHARS = 3000
# How many failing tests the answer names.
MAX_FAILED_TESTS_IN_NOTE = 8

MODULE_NAME = codetest_driver.SOLUTION_MODULE
RESULT_MARKER = codetest_driver.RESULT_MARKER
DRIVER_SOURCE = Path(codetest_driver.__file__).read_text(encoding="utf-8")

RunClass = Literal[
    "passed", "vacuous", "tests_failed", "solution_invalid", "tests_invalid", "not_finished", "infra_failed"
]
RepairState = Literal[
    "not_attempted", "passed", "vacuous", "tests_failed", "solution_invalid", "tests_invalid",
    "not_finished", "infra_failed", "draft_failed",
]

INTERFACE_SYSTEM_PROMPT = """The user asks for Python code. Name the one function that answers the request, so that a solution and a test suite can be written separately against the same interface.

Rules:
- If the question names the function or gives its signature, copy the name and the signature exactly as given.
- Otherwise choose a short snake_case name and parameters that carry everything the question says the function receives, with type hints.
- One plain function. No class, no method, no decorator, no async.
- Do not write the function body and do not solve the problem.

Return JSON with these fields, in this order:
- function_name: the function's name
- signature: the function's def line, for example `def name(parameter: type) -> type:`"""

_SOLUTION_RULES = """Rules for the module:
- It must define the function given under "Required interface", with exactly that name and those parameters. It may define helpers.
- Standard library only. No third-party imports.
- It runs in an isolated sandbox with no network, no input and no files. Do not read input, open files or connect to anything.
- No code at module level besides imports, constants and definitions: nothing is printed and nothing runs when the module is imported.
- Do not write tests. A separate unittest suite, written without seeing your code, imports the function from the module `solution` and calls it.
- Follow what the question states for every case it mentions, including the errors it says to raise."""

SOLUTION_SYSTEM_PROMPT = f"""Write the Python 3 module `solution` that answers the user's question.

{_SOLUTION_RULES}

Return JSON with these fields, in this order:
- working: how the function works, in a sentence or two
- solution: the complete Python source of the module"""

TESTS_SYSTEM_PROMPT = """Write a Python 3 `unittest` module that tests a function someone else is writing for the user's question. You are not shown their code: write the tests from the question alone.

Rules for the test module:
- Import the function with `from solution import <name>`, using the name given under "Required interface", and call it with exactly those parameters.
- Standard library only: `unittest` and whatever else the checks need. No third-party imports, no network, no input, no files.
- Put the tests in `unittest.TestCase` classes with methods named `test_...`. No code at module level besides imports and definitions.
- Test only what the question states: the ordinary cases, the edge cases it mentions and the errors it says to raise. Every expected value must follow from the question; work each one out with care. Do not test behaviour the question leaves open.
- Every test must call the function and must fail when the function returns a wrong value.
- The whole suite must finish in a few seconds and give the same result on every run.
- Do not write the function itself.

Return JSON with these fields, in this order:
- working: what the tests cover, in a sentence or two
- tests: the complete Python source of the test module"""

REPAIR_SYSTEM_PROMPT = f"""A Python module written to answer the user's question failed when it was run against a unittest suite. You are given the question, the required interface, the module and what happened. Write a corrected module.

The tests were written from the question alone, without seeing the module, and can be wrong themselves. Correct the module so that it does what the question states. Do not bend it to a test that contradicts the question.

{_SOLUTION_RULES}

Return JSON with these fields, in this order:
- working: what was wrong and what the new module does
- solution: the complete corrected Python source of the module"""


# --- the bundle: one program for the gate ---


def build_bundle(solution_source: str, tests_source: str, function_name: str) -> str:
    """The one program the gate stages: three string literals, then the driver.

    `ascii()` writes each source as a Python literal with every non-ASCII character
    escaped, so the text of a source can never end the literal or add a statement.
    """
    header = (
        f"SOLUTION_SOURCE = {ascii(solution_source)}\n"
        f"TESTS_SOURCE = {ascii(tests_source)}\n"
        f"FUNCTION_NAME = {ascii(function_name)}\n"
    )
    return header + DRIVER_SOURCE


# --- reading the result ---


@dataclass(frozen=True)
class SuiteSummary:
    """The result line of one driver run."""

    stage: str
    tests_run: int
    failures: int
    errors: int
    skipped: int
    unexpected_successes: int
    failed_tests: tuple[str, ...]
    detail: str
    stub_passed: bool | None

    @property
    def ran(self) -> int:
        """Tests that ran: skipped ones are left out."""
        return self.tests_run - self.skipped

    @property
    def failed(self) -> int:
        return self.failures + self.errors + self.unexpected_successes

    def as_line_fields(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "tests_run": self.tests_run,
            "failures": self.failures,
            "errors": self.errors,
            "skipped": self.skipped,
            "unexpected_successes": self.unexpected_successes,
            "failed_tests": list(self.failed_tests),
            "stub_passed": self.stub_passed,
        }


_COUNT_FIELDS = ("tests_run", "failures", "errors", "skipped", "unexpected_successes")


def parse_summary(stdout: str) -> SuiteSummary | None:
    """The driver's result line, or None when stdout does not end in a well-formed one.

    The line must be the last non-empty line of stdout, start with the marker and hold a
    JSON object with every field of the right type. Anything else is no result.
    """
    last_line = None
    for line in reversed(stdout.splitlines()):
        if line.strip():
            last_line = line.strip()
            break
    if last_line is None or not last_line.startswith(RESULT_MARKER):
        return None
    try:
        fields = json.loads(last_line[len(RESULT_MARKER):])
    except ValueError:
        return None
    if not isinstance(fields, dict) or fields.get("stage") not in codetest_driver.STAGES:
        return None
    for name in _COUNT_FIELDS:
        value = fields.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
    failed_tests = fields.get("failed_tests")
    if not isinstance(failed_tests, list) or not all(isinstance(item, str) for item in failed_tests):
        return None
    if not isinstance(fields.get("detail"), str) or fields.get("stub_passed") not in (True, False, None):
        return None
    return SuiteSummary(
        stage=fields["stage"],
        tests_run=fields["tests_run"],
        failures=fields["failures"],
        errors=fields["errors"],
        skipped=fields["skipped"],
        unexpected_successes=fields["unexpected_successes"],
        failed_tests=tuple(failed_tests),
        detail=fields["detail"],
        stub_passed=fields["stub_passed"],
    )


def classify_test_run(result: sandbox.SandboxResult) -> tuple[RunClass, str | None, SuiteSummary | None]:
    """Turn one sandbox result into a run class. Returns (class, reason or None, summary or None).

    First match wins:

    | # | what came back                                                    | class            |
    |---|-------------------------------------------------------------------|------------------|
    | 1 | a gate or sandbox error (not the program's doing)                 | infra_failed     |
    | 2 | timeout, or output over the cap                                   | not_finished     |
    | 3 | no well-formed result line on stdout                              | not_finished     |
    | 4 | the exit code is not the one the result line implies              | not_finished     |
    | 5 | the solution did not load, or lacks the required function         | solution_invalid |
    | 6 | the tests did not load, the run was cut short, or no test ran     | tests_invalid    |
    | 7 | at least one test failed or raised                                | tests_failed     |
    | 8 | every test passed, and passed against the raising stub too        | vacuous          |
    | 9 | every test passed, and the stub run did not pass                  | passed           |

    Rows 3 and 4 are why a solution that ends the process early with exit code 0 is not a
    pass: a pass needs exit code 0 and a result line that says so.
    """
    if result.error_type in sandbox.INFRASTRUCTURE_ERRORS:
        return "infra_failed", result.error_type, None
    if result.error_type in (sandbox.ERROR_TIMEOUT, sandbox.ERROR_OUTPUT_TOO_LARGE):
        return "not_finished", result.error_type, None
    summary = parse_summary(result.stdout)
    if summary is None:
        return "not_finished", REASON_NO_RESULT, None
    line_fields = summary.as_line_fields()
    clean_run = summary.stage == codetest_driver.STAGE_RAN and summary.failed == 0
    stub_missing = clean_run and summary.stub_passed is None
    no_test_counted = summary.stage == codetest_driver.STAGE_RAN and summary.ran <= 0
    if stub_missing or no_test_counted or result.exit_code != codetest_driver.exit_code_for(line_fields):
        return "not_finished", REASON_RESULT_MISMATCH, None
    if summary.stage in codetest_driver.SOLUTION_STAGES:
        return "solution_invalid", summary.stage, summary
    if summary.stage in codetest_driver.TESTS_STAGES:
        return "tests_invalid", summary.stage, summary
    if summary.failed > 0:
        return "tests_failed", None, summary
    if summary.stub_passed:
        return "vacuous", None, summary
    return "passed", None, summary


# --- the decision table ---


@dataclass(frozen=True)
class CodeTestDecision:
    action: Literal["return", "repair"]
    # Set when action is "return".
    status: ExecutionStatus | None = None
    # Whose solution is returned; None when there is no answer.
    use: Literal["first", "repair"] | None = None


_REPAIRABLE = ("tests_failed", "solution_invalid")
_UNTESTED = ("vacuous", "tests_invalid", "not_finished", "infra_failed")


def decide(first: RunClass, repair: RepairState, repairs_allowed: bool) -> CodeTestDecision:
    """Decision table. First match wins.

    | #  | first run        | repair                                  | result                          |
    |----|------------------|-----------------------------------------|---------------------------------|
    | 1  | passed           | -                                       | tests_passed, the first solution|
    | 2  | vacuous, tests_invalid, not_finished, infra_failed | -     | untested, the first solution    |
    | 3  | tests_failed or solution_invalid | not attempted, allowed  | attempt the one repair          |
    | 4  | tests_failed     | not attempted, not allowed              | tests_failed, the first solution|
    | 5  | solution_invalid | not attempted, not allowed              | failed, no answer               |
    | 6  | either           | passed                                  | tests_passed, the repair        |
    | 7  | either           | tests_failed                            | tests_failed, the repair        |
    | 8  | either           | vacuous                                 | untested, the repair            |
    | 9  | tests_failed     | anything else                           | tests_failed, the first solution|
    | 10 | solution_invalid | tests_invalid, not_finished, infra_failed | untested, the repair          |
    | 11 | solution_invalid | solution_invalid, draft_failed          | failed, no answer               |

    Row 2: nothing here is the solution's proven fault, so no repair is attempted. A run
    that did not finish may be the solution's doing (an endless loop) or the tests'; the
    result says not tested and does not guess. Row 9: the repair could not be tested or
    could not be written, and what is known is that the first solution ran and failed
    tests, so that is what is returned. Rows 5 and 11: no solution that loads exists, so
    nothing is returned. No row returns `verified`.
    """
    if first == "passed":
        return CodeTestDecision("return", "tests_passed", "first")
    if first in _UNTESTED:
        return CodeTestDecision("return", "untested", "first")
    if repair == "not_attempted":
        if repairs_allowed:
            return CodeTestDecision("repair")
        if first == "tests_failed":
            return CodeTestDecision("return", "tests_failed", "first")
        return CodeTestDecision("return", "failed", None)
    if repair == "passed":
        return CodeTestDecision("return", "tests_passed", "repair")
    if repair == "tests_failed":
        return CodeTestDecision("return", "tests_failed", "repair")
    if repair == "vacuous":
        return CodeTestDecision("return", "untested", "repair")
    if first == "tests_failed":
        return CodeTestDecision("return", "tests_failed", "first")
    if repair in ("tests_invalid", "not_finished", "infra_failed"):
        return CodeTestDecision("return", "untested", "repair")
    return CodeTestDecision("return", "failed", None)


# --- the interface, checked in code ---

_QUESTION_DEF = re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


@dataclass(frozen=True)
class Interface:
    function_name: str
    # The canonical def line, without the colon: `def name(parameter: type) -> type`.
    signature: str


def check_interface(question: str, draft: CodeInterface) -> tuple[Interface | None, str | None, str | None]:
    """Returns (interface, None, None) or (None, error type, why it was rejected).

    The signature must parse as one plain function definition whose name is
    `function_name`, a Python identifier that is not a keyword. When the question itself
    holds one or more `def name(` lines, the name must be one of those: an interface that
    renames a function the question fixed would make solution and tests agree with each
    other and not with the question.
    """
    name = draft.function_name.strip()
    if not name.isidentifier() or keyword.iskeyword(name):
        return None, ERROR_INTERFACE_INVALID, f"'{name}' is not a usable function name"
    header = draft.signature.strip().rstrip(":").rstrip()
    try:
        parsed = ast.parse(f"{header}:\n    pass\n")
    except (SyntaxError, ValueError) as parse_error:
        return None, ERROR_INTERFACE_INVALID, f"the signature does not parse: {parse_error}"[:MAX_VALIDATION_MESSAGE_CHARS]
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.FunctionDef):
        return None, ERROR_INTERFACE_INVALID, "the signature is not one plain function definition"
    definition = parsed.body[0]
    if definition.name != name or definition.decorator_list:
        return None, ERROR_INTERFACE_INVALID, f"the signature defines '{definition.name}', not '{name}'"
    stated_names = set(_QUESTION_DEF.findall(question))
    if stated_names and name not in stated_names:
        return (
            None,
            ERROR_INTERFACE_MISMATCH,
            f"the question defines {sorted(stated_names)} and the interface names '{name}'",
        )
    signature = f"def {name}({ast.unparse(definition.args)})"
    if definition.returns is not None:
        signature += f" -> {ast.unparse(definition.returns)}"
    return Interface(function_name=name, signature=signature), None, None


# --- messages ---


def _task_text(question: str, interface: Interface) -> str:
    return f"Question:\n{question}\n\nRequired interface (module `{MODULE_NAME}`):\n{interface.signature}:"


def build_interface_messages(conversation: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"role": "system", "content": INTERFACE_SYSTEM_PROMPT}] + conversation


def build_solution_messages(question: str, interface: Interface) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SOLUTION_SYSTEM_PROMPT},
        {"role": "user", "content": _task_text(question, interface)},
    ]


def build_tests_messages(question: str, interface: Interface) -> list[dict[str, str]]:
    """The whole prompt of the test writer: the question and the interface. There is no
    parameter for a solution, a run or a repair, so none can be passed in."""
    return [
        {"role": "system", "content": TESTS_SYSTEM_PROMPT},
        {"role": "user", "content": _task_text(question, interface)},
    ]


def describe_failure(attempt: "CodeTestAttempt") -> str:
    """What the repair call is told about the failed run."""
    summary = attempt.summary
    if attempt.outcome == "solution_invalid":
        happened = "The module could not be used: " + (
            "it does not define the required function."
            if attempt.reason == codetest_driver.STAGE_SOLUTION_MISSING_FUNCTION
            else "importing it raised an error."
        )
    else:
        happened = f"{summary.failed} of {summary.ran} tests failed: {', '.join(summary.failed_tests)}"
    return f"{happened}\n\nDetails (last {REPAIR_DETAIL_CHARS} characters):\n{summary.detail[-REPAIR_DETAIL_CHARS:] or '(none)'}"


def build_repair_messages(question: str, interface: Interface, attempt: "CodeTestAttempt") -> list[dict[str, str]]:
    user_content = (
        f"{_task_text(question, interface)}\n\nModule that failed:\n{attempt.solution}\n\n"
        f"What happened when it ran:\n{describe_failure(attempt)}"
    )
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# --- control flow ---


@dataclass
class CodeTestAttempt:
    """One run of a solution against the generated tests."""

    attempt: Literal["first", "repair"]
    # The whole program the gate ran: the driver with both sources embedded.
    program: str
    solution: str
    result: sandbox.SandboxResult
    outcome: RunClass
    reason: str | None
    summary: SuiteSummary | None

    @property
    def result_line(self) -> str | None:
        """The counts of the run as JSON, for the audit row. None when there was no result."""
        if self.summary is None:
            return None
        return json.dumps(self.summary.as_line_fields())


@dataclass
class CodeTestOutcome:
    tier: str
    tests_tier: str
    status: str = STATUS_ERROR
    answer: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    final_effort: str | None = None
    execution_status: ExecutionStatus = "skipped"
    execution_note: str | None = None
    repaired: bool = False
    repair_error_type: str | None = None
    function_name: str | None = None
    signature: str | None = None
    # The solution in the answer, the first draft, and the generated tests.
    solution: str | None = None
    first_solution: str | None = None
    tests: str | None = None
    # Counts of the returned solution's run, skipped tests left out.
    tests_run: int | None = None
    tests_failed: int | None = None
    # How the first solution's run was classed, whatever the repair did afterwards.
    first_outcome: str | None = None
    steps: list[ChainStep] = field(default_factory=list)
    attempts: list[CodeTestAttempt] = field(default_factory=list)


def _schema_of(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema["required"] = list(model.model_fields)
    return schema


def _parse(model: type[BaseModel], call_result: CallResult) -> tuple[Any, str | None]:
    """The schema gate. Returns (parsed, None) or (None, why it was rejected)."""
    try:
        return model.model_validate_json(call_result.text), None
    except ValidationError as validation_error:
        return None, " ".join(str(validation_error).split())[:MAX_VALIDATION_MESSAGE_CHARS]


def _why_untested(attempt: CodeTestAttempt) -> str:
    if attempt.outcome == "infra_failed":
        return f"the sandbox could not run the tests ({attempt.reason})"
    if attempt.outcome == "not_finished":
        return f"the test run did not finish ({attempt.reason})"
    if attempt.outcome == "tests_invalid":
        return f"the generated tests could not be run ({attempt.reason})"
    return (
        "the generated tests also pass against a stub that raises on every call, so they check nothing"
    )


def _first_attempt_text(first: CodeTestAttempt) -> str:
    if first.outcome == "solution_invalid":
        return "The first solution could not be loaded; this is the repaired solution."
    return (
        f"The first solution failed {first.summary.failed} of {first.summary.ran} of these tests; "
        "this is the repaired solution."
    )


def format_answer(note: str, solution: str) -> str:
    """The note first, then the solution in one code fence, which is the last thing in the answer."""
    return f"{note}\n\n```python\n{solution.rstrip()}\n```"


def run_codetest(
    question: str,
    conversation: list[dict[str, str]],
    interface_tier: str,
    solution_tier: str,
    tests_tier: str,
    tests_writer: str,
    max_repairs: int,
    effort_override: str | None,
    call_interface_tier: TierCaller,
    call_solution_tier: TierCaller,
    call_tests_tier: TierCaller,
    run_in_sandbox: Callable[[str], sandbox.SandboxResult],
    simulated_answer_failure: Callable[[], CallResult] | None = None,
    parallel: bool = False,
) -> CodeTestOutcome:
    """`tests_writer` names who wrote the tests (tier and model) for the answer text. With
    `parallel` the solution and the tests are written at the same time: neither call has
    anything of the other as an input, so the order cannot matter."""
    outcome = CodeTestOutcome(tier=solution_tier, tests_tier=tests_tier)

    def typed_call(
        role: str,
        tier_name: str,
        call_tier: TierCaller,
        messages: list[dict[str, str]],
        model: type[BaseModel],
        invalid_error: str,
        simulated: Callable[[], CallResult] | None = None,
    ) -> tuple[Any, ChainStep, str | None, str | None]:
        """One model call through its schema gate. Returns (parsed or None, step, error type, message)."""
        if simulated is not None:
            call_result, effort = simulated(), effort_override
        else:
            call_result, effort = call_tier(tier_name, messages, _schema_of(model), effort_override)
        if not call_result.ok:
            step = ChainStep(role, tier_name, effort, call_result)
            return None, step, call_result.error_type, call_result.error_message
        parsed, rejection = _parse(model, call_result)
        step = ChainStep(role, tier_name, effort, call_result, invalid=parsed is None)
        if parsed is None:
            return None, step, invalid_error, rejection
        return parsed, step, None, None

    def fail(error_type: str | None, error_message: str | None) -> CodeTestOutcome:
        outcome.error_type, outcome.error_message = error_type, error_message
        return outcome

    interface_draft, step, error_type, error_message = typed_call(
        ROLE_INTERFACE, interface_tier, call_interface_tier, build_interface_messages(conversation),
        CodeInterface, ERROR_INTERFACE_INVALID,
    )
    outcome.steps.append(step)
    if interface_draft is None:
        return fail(error_type, error_message)
    interface, error_type, error_message = check_interface(question, interface_draft)
    if interface is None:
        step.invalid = True
        return fail(error_type, error_message)
    outcome.function_name, outcome.signature = interface.function_name, interface.signature

    def write_solution():
        return typed_call(
            ROLE_DRAFT, solution_tier, call_solution_tier, build_solution_messages(question, interface),
            SolutionDraft, ERROR_SOLUTION_SCHEMA_INVALID, simulated_answer_failure,
        )

    def write_tests():
        return typed_call(
            ROLE_TESTS, tests_tier, call_tests_tier, build_tests_messages(question, interface),
            UnitTestsDraft, ERROR_TESTS_SCHEMA_INVALID,
        )

    if parallel:
        with ThreadPoolExecutor(max_workers=2) as pool:
            solution_future, tests_future = pool.submit(write_solution), pool.submit(write_tests)
            solution_call, tests_call = solution_future.result(), tests_future.result()
    else:
        solution_call = write_solution()
        # Without a solution there is nothing to test, so the tests are not asked for.
        tests_call = write_tests() if solution_call[0] is not None else None

    solution_draft, solution_step, error_type, error_message = solution_call
    outcome.steps.append(solution_step)
    outcome.final_effort = solution_step.effort
    if tests_call is not None:
        outcome.steps.append(tests_call[1])
    if solution_draft is None:
        return fail(error_type, error_message)
    outcome.solution = outcome.first_solution = solution_draft.solution

    tests_draft, _, tests_error_type, _ = tests_call
    if tests_draft is None:
        outcome.execution_status = "untested"
        outcome.execution_note = (
            f"NOT TESTED: no tests could be written ({tests_error_type}). The solution below was not "
            "run against any test that could fail it."
        )
        outcome.answer = format_answer(outcome.execution_note, outcome.solution)
        outcome.status = STATUS_OK
        return outcome
    outcome.tests = tests_draft.tests

    def run_attempt(label: Literal["first", "repair"], solution: str) -> CodeTestAttempt:
        program = build_bundle(solution, tests_draft.tests, interface.function_name)
        result = run_in_sandbox(program)
        run_class, reason, summary = classify_test_run(result)
        attempt = CodeTestAttempt(label, program, solution, result, run_class, reason, summary)
        outcome.attempts.append(attempt)
        return attempt

    first_attempt = run_attempt("first", solution_draft.solution)
    outcome.first_outcome = first_attempt.outcome
    repair_state: RepairState = "not_attempted"
    repair_attempt: CodeTestAttempt | None = None
    decision = decide(first_attempt.outcome, repair_state, max_repairs > 0)
    if decision.action == "repair":
        repair_draft, repair_step, outcome.repair_error_type, _ = typed_call(
            ROLE_REPAIR, solution_tier, call_solution_tier,
            build_repair_messages(question, interface, first_attempt),
            SolutionDraft, ERROR_SOLUTION_SCHEMA_INVALID,
        )
        outcome.steps.append(repair_step)
        if repair_draft is None:
            repair_state = "draft_failed"
        else:
            repair_attempt = run_attempt("repair", repair_draft.solution)
            repair_state = repair_attempt.outcome
        decision = decide(first_attempt.outcome, repair_state, repairs_allowed=False)

    outcome.execution_status = decision.status
    if decision.status == "failed":
        loaded = "also after one repair" if repair_state != "not_attempted" else "and no repair was allowed"
        outcome.execution_note = (
            f"FAILED: no answer. The solution could not be loaded, {loaded} "
            f"({first_attempt.reason}): it does not import or does not define {interface.function_name}."
        )
        outcome.solution = None
        return fail(ERROR_SOLUTION_INVALID, outcome.execution_note)

    final_attempt = repair_attempt if decision.use == "repair" else first_attempt
    outcome.repaired = decision.use == "repair"
    outcome.solution = final_attempt.solution
    if final_attempt.summary is not None and final_attempt.outcome in ("passed", "vacuous", "tests_failed"):
        outcome.tests_run, outcome.tests_failed = final_attempt.summary.ran, final_attempt.summary.failed

    if decision.status == "tests_passed":
        note = (
            f"Passed its generated tests: {outcome.tests_run} of {outcome.tests_run} unit tests, written by "
            f"{tests_writer} from the question alone, passed in an isolated sandbox. The tests were "
            "written by a model and were not checked against a reference, so this does not show the "
            "solution is correct."
        )
    elif decision.status == "tests_failed":
        named = ", ".join(final_attempt.summary.failed_tests[:MAX_FAILED_TESTS_IN_NOTE])
        note = (
            f"FAILED its generated tests: {outcome.tests_failed} of {outcome.tests_run} unit tests, written "
            f"by {tests_writer} from the question alone, failed in an isolated sandbox ({named}). The "
            "tests were written by a model and were not checked against a reference, so a failure can "
            "be a wrong test as well as a wrong solution."
        )
        if repair_state != "not_attempted" and not outcome.repaired:
            note += " One repair was attempted and gave nothing that could be tested; this is the first solution."
    else:
        note = (
            f"NOT TESTED: {_why_untested(final_attempt)}. The solution below was not run against any "
            "test that could fail it."
        )
    if outcome.repaired:
        note += " " + _first_attempt_text(first_attempt)
    outcome.execution_note = note
    outcome.answer = format_answer(note, outcome.solution)
    outcome.status = STATUS_OK
    return outcome
