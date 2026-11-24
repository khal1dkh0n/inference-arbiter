"""The hard closed-book set: schema, and every computed answer re-derived by its reference program."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from inference_arbiter.grading import grade_answer

REPO_ROOT = Path(__file__).resolve().parent.parent
HARD_PATH = REPO_ROOT / "eval" / "hard_questions.yaml"
GOLDEN_PATH = REPO_ROOT / "eval" / "golden_questions.yaml"
REFERENCE_DIR = REPO_ROOT / "eval" / "hard_reference"

REQUIRED_FIELDS = {
    "id", "question", "expected_answer", "match", "category", "expected_route", "difficulty", "notes",
}
EXECUTABLE_CATEGORY = "executable"
CATEGORIES = {EXECUTABLE_CATEGORY, "reasoning", "domain", "trap"}
ROUTES = {"local", "code", "mid", "frontier"}
DIFFICULTIES = {"easy", "medium", "hard"}
MATCH_FIELDS = {
    "exact": [],
    "numeric": ["value", "tolerance"],
    "rubric": ["criteria", "pass_rule"],
}
# The reference program's stdout is the expected answer, except where the answer is prose.
REFERENCE_OUTPUT_OVERRIDES = {
    "HT-04": "count=14 net=4848.39 trailer=5770.49 MISMATCH",
    "HT-05": "MALFORMED",
    "HT-06": "-3333 -3333 -3333 sum=-9999",
}
REFERENCE_TIMEOUT_S = 30


def load_cases() -> list[dict]:
    return yaml.safe_load(HARD_PATH.read_text(encoding="utf-8"))["cases"]


CASES = load_cases()
CASES_WITH_REFERENCE = [case for case in CASES if "reference" in case]


def without_whitespace(text: str) -> str:
    return re.sub(r"\s+", "", text)


def load_reference_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"hard_reference_{path.stem.replace('-', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_file_loads_through_the_eval_loader():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import run_eval
    finally:
        sys.path.pop(0)
    cases, golden_info = run_eval.load_golden(HARD_PATH)
    assert len(cases) == len(CASES)
    assert golden_info["version"] == 1
    executable = run_eval.filter_cases(cases, None, [EXECUTABLE_CATEGORY])
    assert [case["id"] for case in executable] == [case["id"] for case in CASES if case["id"].startswith("HX-")]


def test_ids_are_unique_and_do_not_collide_with_the_golden_set():
    ids = [case["id"] for case in CASES]
    assert len(ids) == len(set(ids))
    golden_ids = {case["id"] for case in yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]}
    assert not golden_ids & set(ids)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_case_schema(case):
    assert REQUIRED_FIELDS <= set(case), f"missing fields: {REQUIRED_FIELDS - set(case)}"
    for field in ("id", "question", "expected_answer", "category", "expected_route", "difficulty", "notes"):
        assert isinstance(case[field], str) and case[field].strip(), f"{field} must be a non-empty string"
    assert case["category"] in CATEGORIES
    assert case["expected_route"] in ROUTES
    assert case["difficulty"] in DIFFICULTIES
    match = case["match"]
    match_type = match["type"]
    if match_type == "contains":
        assert match.get("all_of") or match.get("any_of")
    else:
        assert match_type in MATCH_FIELDS, f"unknown match type {match_type}"
        for field in MATCH_FIELDS[match_type]:
            assert field in match, f"{match_type} match needs {field}"
    if match_type == "rubric":
        assert match["criteria"] and all(isinstance(criterion, str) for criterion in match["criteria"])


def test_about_half_the_set_is_executable_and_each_has_a_reference():
    executable = [case for case in CASES if case["category"] == EXECUTABLE_CATEGORY]
    assert len(executable) >= 10
    assert all(case["id"].startswith("HX-") for case in executable)
    assert all("reference" in case for case in executable)
    assert all(case["match"]["type"] != "rubric" for case in executable)


def test_every_reference_file_belongs_to_a_case():
    referenced = {(REPO_ROOT / case["reference"]).resolve() for case in CASES_WITH_REFERENCE}
    on_disk = {path.resolve() for path in REFERENCE_DIR.glob("*.py")}
    assert referenced == on_disk
    for case in CASES_WITH_REFERENCE:
        assert Path(case["reference"]).stem == case["id"]


@pytest.mark.parametrize("case", CASES_WITH_REFERENCE, ids=lambda case: case["id"])
def test_reference_program_prints_the_expected_answer(case):
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / case["reference"])],
        capture_output=True,
        text=True,
        timeout=REFERENCE_TIMEOUT_S,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    expected_output = REFERENCE_OUTPUT_OVERRIDES.get(case["id"], case["expected_answer"])
    assert completed.stdout.strip() == expected_output


@pytest.mark.parametrize("case", CASES_WITH_REFERENCE, ids=lambda case: case["id"])
def test_reference_program_uses_the_data_in_the_question(case):
    """Each reference lists the literals it computes from; all must appear in the question."""
    module = load_reference_module(REPO_ROOT / case["reference"])
    question = without_whitespace(case["question"])
    assert module.QUESTION_DATA
    for literal in module.QUESTION_DATA:
        assert without_whitespace(literal) in question, f"{literal!r} is not in the question"


@pytest.mark.parametrize(
    "case", [case for case in CASES if case["match"]["type"] != "rubric"], ids=lambda case: case["id"]
)
def test_grader_passes_the_expected_answer_and_fails_a_wrong_one(case):
    expected = case["expected_answer"]
    for phrasing in (expected, f"Working shown above.\n\nAnswer: {expected}", f"**Answer: {expected}**"):
        assert grade_answer(case, phrasing).status == "pass", phrasing
    assert grade_answer(case, expected.lower()).status == "pass"
    assert grade_answer(case, "Answer: 7").status == "fail"


def test_numeric_expected_answers_agree_with_their_match_value():
    for case in CASES:
        if case["match"]["type"] == "numeric":
            assert float(case["expected_answer"]) == float(case["match"]["value"]), case["id"]
            assert case["match"]["tolerance"] == 0, case["id"]
