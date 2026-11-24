"""The code question set: schema, and every reference file proven by running it.

Each case's reference tests must pass its reference solution and fail every seeded wrong
solution. The runs use the codetest driver on the host interpreter; the sources are the
repository's own files, so no sandbox is needed and no model is called.
"""

from __future__ import annotations

import ast
import difflib
import re
import sys
from pathlib import Path

import pytest

from inference_arbiter import codetest
from inference_arbiter.grading import grade_answer
from inference_arbiter.schemas import CodeInterface

from conftest import REPO_ROOT, run_on_host

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_eval  # noqa: E402

CODE_PATH = REPO_ROOT / "eval" / "code_questions.yaml"
REFERENCE_DIR = REPO_ROOT / "eval" / "code_reference"
GOLDEN_PATH = REPO_ROOT / "eval" / "golden_questions.yaml"
HARD_PATH = REPO_ROOT / "eval" / "hard_questions.yaml"

REQUIRED_FIELDS = (
    "id", "question", "expected_answer", "match", "category", "expected_route", "difficulty", "notes",
)
CATEGORIES = {"encoding", "parsing", "numeric", "sequence", "rules"}

CASES, GOLDEN_INFO = run_eval.load_golden(CODE_PATH)
BROKEN = [(case, path) for case in CASES for path in case["broken_solutions"]]


def read(path_text: str) -> str:
    return (REPO_ROOT / path_text).read_text(encoding="utf-8")


def test_file_loads_through_the_eval_loader():
    assert GOLDEN_INFO["version"] == 1
    assert 20 <= len(CASES) <= 30


def test_ids_are_unique_and_do_not_collide_with_the_other_sets():
    ids = [case["id"] for case in CASES]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"CQ-\d\d", case_id) for case_id in ids)
    other_ids = {
        case["id"] for path in (GOLDEN_PATH, HARD_PATH) for case in run_eval.load_golden(path)[0]
    }
    assert not other_ids & set(ids)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_case_schema(case):
    for field in REQUIRED_FIELDS:
        assert case.get(field), f"{case['id']} lacks {field}"
    assert case["category"] in CATEGORIES
    assert case["difficulty"] in ("easy", "medium", "hard")
    assert case["expected_route"] in ("local", "code", "mid", "frontier")
    assert set(case["match"]) == {"type", "function", "tests"}
    assert case["match"]["type"] == run_eval.MATCH_REFERENCE_TESTS
    assert case["match"]["tests"] == f"eval/code_reference/{case['id']}_tests.py"
    assert case["reference_solution"] == f"eval/code_reference/{case['id']}.py"
    assert len(case["broken_solutions"]) >= 2
    assert case["broken_solutions"] == [
        f"eval/code_reference/{case['id']}_broken_{number}.py"
        for number in range(1, len(case["broken_solutions"]) + 1)
    ]
    assert case["question"].isascii()


def test_every_reference_file_belongs_to_a_case():
    named = {
        (REPO_ROOT / path).resolve()
        for case in CASES
        for path in (case["match"]["tests"], case["reference_solution"], *case["broken_solutions"])
    }
    on_disk = {path.resolve() for path in REFERENCE_DIR.glob("*.py")}
    assert named == on_disk


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_the_question_states_the_signature_the_reference_files_use(case):
    function = case["match"]["function"]
    stated = re.findall(r"^def .+$", case["question"], flags=re.MULTILINE)
    assert len(stated) == 1, "the question gives exactly one def line"
    interface, error_type, message = codetest.check_interface(
        case["question"], CodeInterface(function_name=function, signature=stated[0])
    )
    assert error_type is None, message
    reference = ast.parse(read(case["reference_solution"]))
    definitions = [node for node in reference.body if isinstance(node, ast.FunctionDef) and node.name == function]
    assert len(definitions) == 1
    assert f"def {function}({ast.unparse(definitions[0].args)}) -> {ast.unparse(definitions[0].returns)}" == interface.signature
    assert f"from solution import {function}\n" in read(case["match"]["tests"])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_reference_tests_pass_the_reference_solution(case, tmp_path):
    result = run_on_host(
        tmp_path, read(case["reference_solution"]), read(case["match"]["tests"]), case["match"]["function"]
    )

    run_class, reason, summary = codetest.classify_test_run(result)
    assert run_class == "passed", (reason, summary.detail if summary else result.stderr[-800:])
    assert summary.ran >= 5 and summary.skipped == 0
    assert summary.stub_passed is False, "the reference tests do not pass against a stub"


@pytest.mark.parametrize("case, broken_path", BROKEN, ids=lambda value: value["id"] if isinstance(value, dict) else Path(value).stem)
def test_reference_tests_fail_every_seeded_wrong_solution(case, broken_path, tmp_path):
    result = run_on_host(tmp_path, read(broken_path), read(case["match"]["tests"]), case["match"]["function"])

    run_class, reason, summary = codetest.classify_test_run(result)
    assert run_class == "tests_failed", (run_class, reason)
    assert summary.failed >= 1
    assert run_eval.reference_grade_of(
        {"run_class": run_class, "reason": reason, "tests_run": summary.ran, "tests_failed": summary.failed}
    )[0] == "fail"


@pytest.mark.parametrize("case, broken_path", BROKEN, ids=lambda value: value["id"] if isinstance(value, dict) else Path(value).stem)
def test_a_seeded_wrong_solution_differs_from_the_reference_by_its_bug_only(case, broken_path):
    def code_of(source: str) -> str:
        module = ast.parse(source)
        module.body = module.body[1:]  # the file's own docstring says which file it is
        return ast.unparse(module)

    reference, broken = code_of(read(case["reference_solution"])), code_of(read(broken_path))
    changed = [
        line for line in difflib.ndiff(reference.splitlines(), broken.splitlines()) if line[:1] in ("+", "-")
    ]
    assert 1 <= len(changed) <= 8, "a seeded bug is a small change to the reference solution"


def test_reference_files_mark_themselves_as_never_part_of_a_prompt():
    for path in REFERENCE_DIR.glob("*.py"):
        assert "never part of a prompt" in ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))


def test_no_question_quotes_its_reference_tests():
    for case in CASES:
        test_lines = {
            line.strip() for line in read(case["match"]["tests"]).splitlines() if "assert" in line
        }
        assert not [line for line in test_lines if line in case["question"]]


def test_the_text_graders_do_not_grade_a_code_case():
    """Without the sandbox path a reference_tests case is ungraded, never a silent pass or fail."""
    grade = grade_answer(CASES[0], read(CASES[0]["reference_solution"]), None)
    assert grade.status == "ungraded"
