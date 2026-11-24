"""The program that runs inside the sandbox in codetest mode.

`codetest.build_bundle` puts three assignments in front of this text (SOLUTION_SOURCE,
TESTS_SOURCE and FUNCTION_NAME) and the result is the one program the dispatch gate stages
and runs. Nothing here is written by a model. Standard library only, no file is written.

What it does, in one process:

1. Load the solution source as the module `solution`. A solution that does not load, or
   that does not define the required function, ends the run.
2. Load the test source as the module `test_solution` and run every `unittest` test in it.
3. When every test passed, run the same tests once more against a stub: a copy of the
   solution module whose required function raises on every call. Tests that also pass
   against the stub check nothing.
4. Print one result line on stdout and exit with the code that line implies
   (`exit_code_for`). The host accepts a result only when the two agree.

Whatever the solution and the tests print is captured and never reaches stdout, so the
result line is the only line there. The solution and the tests share this process with
the driver: code that set out to forge the result line could. See the README.
"""

import io
import json
import linecache
import os
import sys
import traceback
import types
import unittest

RESULT_MARKER = "CODETEST_RESULT "

SOLUTION_MODULE = "solution"
TESTS_MODULE = "test_solution"
SOLUTION_FILE = "solution.py"
TESTS_FILE = "test_solution.py"

STAGE_SOLUTION_IMPORT = "solution_import"
STAGE_SOLUTION_MISSING_FUNCTION = "solution_missing_function"
STAGE_TESTS_IMPORT = "tests_import"
STAGE_TESTS_ABORTED = "tests_aborted"
STAGE_NO_TESTS = "no_tests"
STAGE_RAN = "ran"
SOLUTION_STAGES = (STAGE_SOLUTION_IMPORT, STAGE_SOLUTION_MISSING_FUNCTION)
TESTS_STAGES = (STAGE_TESTS_IMPORT, STAGE_TESTS_ABORTED, STAGE_NO_TESTS)
STAGES = SOLUTION_STAGES + TESTS_STAGES + (STAGE_RAN,)

# None of these is 1 (an uncaught Python error), 124, 125 or 137 (timeout and sbx codes).
EXIT_PASSED = 0
EXIT_TESTS_FAILED = 10
EXIT_SOLUTION_INVALID = 11
EXIT_TESTS_INVALID = 12
EXIT_VACUOUS = 13

MAX_FAILED_TESTS_LISTED = 20
MAX_TEST_ID_CHARS = 200
MAX_TRACEBACKS = 3
MAX_TRACEBACK_CHARS = 1200
MAX_DETAIL_CHARS = 4000
MAX_CAPTURED_OUTPUT_CHARS = 2000


class SolutionStubCalled(Exception):
    """Raised by the stub that stands in for the required function."""


def exit_code_for(summary):
    """The exit code a result line implies. The host uses the same function to check it.

    | stage                                   | tests            | stub run   | exit |
    |-----------------------------------------|------------------|------------|------|
    | solution_import, solution_missing_function | -             | -          | 11   |
    | tests_import, tests_aborted, no_tests   | -                | -          | 12   |
    | ran                                     | at least one failed | -       | 10   |
    | ran                                     | all passed       | passed too | 13   |
    | ran                                     | all passed       | did not    | 0    |
    """
    stage = summary["stage"]
    if stage in SOLUTION_STAGES:
        return EXIT_SOLUTION_INVALID
    if stage in TESTS_STAGES:
        return EXIT_TESTS_INVALID
    if summary["failures"] + summary["errors"] + summary["unexpected_successes"] > 0:
        return EXIT_TESTS_FAILED
    if summary["stub_passed"]:
        return EXIT_VACUOUS
    return EXIT_PASSED


def blank_summary(stage, detail=""):
    return {
        "stage": stage,
        "tests_run": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "unexpected_successes": 0,
        "failed_tests": [],
        "detail": detail[-MAX_DETAIL_CHARS:],
        "stub_passed": None,
    }


def load_module(name, filename, source):
    """Run `source` as the module `name`. The text is registered so tracebacks show its lines."""
    module = types.ModuleType(name)
    module.__file__ = filename
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    sys.modules[name] = module
    try:
        exec(compile(source, filename, "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def stub_of(solution_module, function_name):
    """A copy of the solution module whose required function raises on every call."""
    stub = types.ModuleType(SOLUTION_MODULE)
    for name, value in solution_module.__dict__.items():
        if not name.startswith("__"):
            stub.__dict__[name] = value

    def stub_function(*args, **kwargs):
        raise SolutionStubCalled(f"{function_name} was called on the stub")

    stub.__dict__[function_name] = stub_function
    return stub


def run_tests(tests_source):
    """Load the test module against whatever `solution` is in sys.modules and run it.

    Returns (TestResult, None), or (None, (stage, traceback text)) when the tests could
    not be loaded or the run was cut short.
    """
    try:
        tests_module = load_module(TESTS_MODULE, TESTS_FILE, tests_source)
    except BaseException:
        return None, (STAGE_TESTS_IMPORT, traceback.format_exc())
    result = unittest.TestResult()
    try:
        unittest.defaultTestLoader.loadTestsFromModule(tests_module).run(result)
    except BaseException:
        return None, (STAGE_TESTS_ABORTED, traceback.format_exc())
    return result, None


def tests_that_ran(result):
    return result.testsRun - len(result.skipped)


def all_passed(result):
    return (
        tests_that_ran(result) > 0
        and not result.failures
        and not result.errors
        and not result.unexpectedSuccesses
    )


def summarise(result):
    summary = blank_summary(STAGE_RAN if tests_that_ran(result) > 0 else STAGE_NO_TESTS)
    failed = list(result.failures) + list(result.errors)
    failed_ids = [str(test)[:MAX_TEST_ID_CHARS] for test, _ in failed]
    failed_ids += [str(test)[:MAX_TEST_ID_CHARS] for test in result.unexpectedSuccesses]
    tracebacks = [text[-MAX_TRACEBACK_CHARS:] for _, text in failed[:MAX_TRACEBACKS]]
    summary.update(
        tests_run=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
        skipped=len(result.skipped),
        unexpected_successes=len(result.unexpectedSuccesses),
        failed_tests=failed_ids[:MAX_FAILED_TESTS_LISTED],
        detail="\n".join(tracebacks)[-MAX_DETAIL_CHARS:],
    )
    return summary


def run_bundle(solution_source, tests_source, function_name):
    """Steps 1 to 3 of the module docstring. Returns the summary."""
    try:
        solution_module = load_module(SOLUTION_MODULE, SOLUTION_FILE, solution_source)
    except BaseException:
        return blank_summary(STAGE_SOLUTION_IMPORT, traceback.format_exc())
    if not callable(getattr(solution_module, function_name, None)):
        return blank_summary(
            STAGE_SOLUTION_MISSING_FUNCTION, f"the solution does not define a callable named {function_name}"
        )

    result, problem = run_tests(tests_source)
    if result is None:
        return blank_summary(problem[0], problem[1])
    summary = summarise(result)
    if summary["stage"] == STAGE_RAN and all_passed(result):
        sys.modules[SOLUTION_MODULE] = stub_of(solution_module, function_name)
        stub_result, _ = run_tests(tests_source)
        summary["stub_passed"] = stub_result is not None and all_passed(stub_result)
    return summary


def main():
    real_stdout, real_stderr = sys.__stdout__, sys.__stderr__
    captured = io.StringIO()
    sys.stdout = sys.stderr = captured
    try:
        summary = run_bundle(SOLUTION_SOURCE, TESTS_SOURCE, FUNCTION_NAME)  # noqa: F821
    finally:
        sys.stdout, sys.stderr = real_stdout, real_stderr
    printed = captured.getvalue()
    if printed:
        real_stderr.write("output of the solution and the tests (last part):\n")
        real_stderr.write(printed[-MAX_CAPTURED_OUTPUT_CHARS:] + "\n")
    real_stderr.flush()
    real_stdout.write(RESULT_MARKER + json.dumps(summary) + "\n")
    real_stdout.flush()
    # Leave at once: a thread or an exit handler left by the code under test must not
    # change the exit code or hold the process open.
    os._exit(exit_code_for(summary))


if __name__ == "__main__":
    main()
