"""Run answer strategies over an eval set, grade them and print a scorecard.

    uv run python scripts/run_eval.py --strategies tier:local tier:mid router
    uv run python scripts/run_eval.py --strategies tier:frontier chain --merge data/eval/eval_X.json
    uv run python scripts/run_eval.py --regrade data/eval/eval_X.json
    uv run python scripts/run_eval.py --suite openbook --strategies closed:mid stuffed:mid openbook

Closed-book suite (default, eval/golden_questions.yaml):
    tier:<name>   force one answer tier (route mode). A run that fell back to another
                  tier is flagged and excluded from that tier's accuracy.
    router        triage plus route (route mode); --triage-tier selects the triage tier.
    chain         chain mode; --verifier-tier is passed through when the harness accepts it.
    auto          auto mode with the computation check off: one call to auto.closed_book_tier.
    auto:check    auto mode with the computation check on: a question the check says a program
                  can compute goes to execute mode, the rest take the closed-book call.

Open-book suite (--suite openbook, eval/openbook/questions.yaml over eval/openbook/corpus.yaml):
    openbook        the grounded pipeline (mode openbook) over the case's source documents.
    stuffed:<tier>  one call to the forced tier with every source document pasted before the
                    question (mode stuffed; built here when the harness has no such mode yet).
    closed:<tier>   one call to the forced tier with no sources: what memory alone gets.
The open-book suite does not run unless every corpus file matches the manifest hash
(scripts/fetch_corpus.py fetches and checks them).

Grading rules are in eval/README.md and eval/openbook/README.md and implemented in
inference_arbiter.grading. Results go to a timestamped JSON under data/eval/ and telemetry to
data/eval/telemetry.db. The closed-book suite also writes the cheapest tier measured correct
for each case to data/eval/measured_routes.json, but only when the results cover every tier
the routes are derived from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, get_args

import yaml

import fetch_corpus
from inference_arbiter.clients import build_clients
from inference_arbiter.config import DEFAULT_CONFIG_PATH, load_config, load_env_file
from inference_arbiter.grading import (
    DEFAULT_JUDGE_TIER,
    GRADE_FAIL,
    GRADE_PASS,
    GRADE_UNGRADED,
    Grade,
    LlmRubricJudge,
    OpenBookGrade,
    RubricJudge,
    grade_answer,
    grade_openbook,
    normalise_whitespace,
)
from inference_arbiter.router import build_harness
from inference_arbiter.schemas import AskRequest

SUITE_CLOSEDBOOK = "closedbook"
SUITE_OPENBOOK = "openbook"

DEFAULT_GOLDEN_PATH = Path("eval/golden_questions.yaml")
DEFAULT_OPENBOOK_QUESTIONS_PATH = Path("eval/openbook/questions.yaml")
DEFAULT_CORPUS_MANIFEST_PATH = Path("eval/openbook/corpus.yaml")
DEFAULT_EVAL_DIR = Path("data/eval")
DEFAULT_TELEMETRY_DB = DEFAULT_EVAL_DIR / "telemetry.db"
DEFAULT_ROUTES_PATH = DEFAULT_EVAL_DIR / "measured_routes.json"
BASELINE_STRATEGY = "tier:mid"
# Call roles that send a draft to a stronger tier: `repair` in chain mode, `escalate` in openbook mode.
ESCALATION_CALL_ROLES = ("repair", "escalate")
ESCALATION_SUCCEEDED = "succeeded"
ESCALATION_FAILED = "failed"
OPENBOOK_BASELINE_STRATEGY = "stuffed:mid"

KIND_TIER = "tier"
KIND_ROUTER = "router"
KIND_CHAIN = "chain"
KIND_OPENBOOK = "openbook"
KIND_STUFFED = "stuffed"
KIND_CLOSED = "closed"
# execute:<tier>: the tier writes a program, a sandbox runs it, code compares (execute mode).
KIND_EXECUTE = "execute"
# crosscheck:<tier A>+<tier B>: execute mode with two independently written programs, A by
# the first tier and B by the second; only matching outputs are `verified`.
# crosscheck:<tier> runs both on one tier (B is then asked for a different method).
KIND_CROSSCHECK = "crosscheck"
# auto: auto mode with the computation check forced off.
# auto:check: the same with the check forced on, whatever auto.computation_check says.
KIND_AUTO = "auto"
KIND_AUTO_CHECK = "auto:check"
MODE_EXECUTE = "execute"
VERDICT_NOT_RUN = "not_run"
VERDICT_YES = "yes"
COMPUTATION_VERDICTS = ("yes", "no", "low_confidence", "failed")
ROLE_CHECK = "check"
EXECUTION_STATUSES = (
    "verified", "disputed", "one_sided", "claim_matched", "corrected", "executed", "unverified", "failed"
)
STATUS_DISPUTED = "disputed"
# Match types a single program's output can be graded on without a judge call.
PER_PROGRAM_MATCH_TYPES = ("numeric", "exact", "contains")
SUITE_KINDS = {
    SUITE_CLOSEDBOOK: (
        KIND_TIER, KIND_ROUTER, KIND_CHAIN, KIND_EXECUTE, KIND_CROSSCHECK, KIND_AUTO, KIND_AUTO_CHECK,
    ),
    SUITE_OPENBOOK: (KIND_OPENBOOK, KIND_STUFFED, KIND_CLOSED),
}
SUITE_DEFAULT_STRATEGIES = {
    SUITE_CLOSEDBOOK: ["tier:local", "tier:mid", "router"],
    SUITE_OPENBOOK: ["closed:mid", "stuffed:mid", "openbook"],
}
# Strategies that force one answer tier; a run that left that tier says nothing about it.
FORCED_TIER_KINDS = (KIND_TIER, KIND_STUFFED, KIND_CLOSED, KIND_EXECUTE, KIND_CROSSCHECK)

MODE_OPENBOOK = "openbook"
MODE_STUFFED = "stuffed"
STUFFED_BY_HARNESS = "harness"
STUFFED_BY_RUNNER = "runner"
# Used only while the harness has no stuffed mode of its own: the runner pastes the sources.
STUFFED_PROMPT_HEADER = (
    "Answer the question at the end using only the source documents below. If the documents "
    "do not contain the answer, say that the sources do not contain it; do not answer from memory."
)

OUTCOME_PASS = "pass"
OUTCOME_FAIL = "fail"
OUTCOME_UNGRADED = "ungraded"
OUTCOME_ERROR = "error"
OUTCOME_FALLBACK_EXCLUDED = "fallback_excluded"
COUNTED_OUTCOMES = (OUTCOME_PASS, OUTCOME_FAIL)
ANSWERED_OUTCOMES = (OUTCOME_PASS, OUTCOME_FAIL, OUTCOME_UNGRADED)

# Measured-route ordering. local and code are the same strength (both run on this machine
# at zero cost); code is only considered for cases in the code category.
TIER_STRENGTH = {"local": 0, "code": 0, "mid": 1, "frontier": 2}
CODE_CATEGORY = "code"
ROUTE_ORDER_GENERAL = ["local", "mid", "frontier"]
ROUTE_ORDER_CODE = ["code", "local", "mid", "frontier"]
ROUTE_ORDERING_NOTE = (
    "Cheapest first: local and code (strength 0, on this machine, zero cost), then mid "
    "(strength 1), then frontier (strength 2). The code tier is only considered for cases "
    "in the code category, where it is tried before local; elsewhere the order is local, "
    "mid, frontier. 'acceptable' lists every tier of the cheapest correct strength."
)

ROUTE_MEASURED = "measured"
ROUTE_PROVISIONAL = "provisional"
ROUTE_NONE_CORRECT = "none_correct"
ROUTE_UNRESOLVED = "unresolved"

TIER_RESULT_MISSING = "missing"

WILSON_Z_95 = 1.96
SMALL_SAMPLE_CAVEAT = (
    "Caveat: small sample. With {cases} cases one case moves accuracy by {step:.0f} points, and "
    "each case was answered once (answer calls are not temperature 0), so differences of one or "
    "two cases between strategies are within noise. The 95% interval shown is a Wilson interval."
)


@dataclass(frozen=True)
class Strategy:
    name: str
    kind: str
    tier: str | None = None
    # crosscheck only: the tier that writes program B.
    cross_check_tier: str | None = None


HarnessFactory = Callable[[Strategy], Any]


def parse_strategy(text: str) -> Strategy:
    if text == KIND_ROUTER:
        return Strategy(name=text, kind=KIND_ROUTER)
    if text == KIND_CHAIN:
        return Strategy(name=text, kind=KIND_CHAIN)
    if text == KIND_OPENBOOK:
        return Strategy(name=text, kind=KIND_OPENBOOK)
    if text in (KIND_AUTO, KIND_AUTO_CHECK):
        return Strategy(name=text, kind=text)
    crosscheck_prefix = f"{KIND_CROSSCHECK}:"
    if text.startswith(crosscheck_prefix):
        tier_a, _, tier_b = text[len(crosscheck_prefix):].partition("+")
        if tier_a and (tier_b or "+" not in text):
            return Strategy(name=text, kind=KIND_CROSSCHECK, tier=tier_a, cross_check_tier=tier_b or tier_a)
        raise ValueError(f"unknown strategy '{text}'; use crosscheck:<tier A>+<tier B> or crosscheck:<tier>")
    for kind in FORCED_TIER_KINDS:
        prefix = f"{kind}:"
        if text.startswith(prefix) and len(text) > len(prefix):
            return Strategy(name=text, kind=kind, tier=text.split(":", 1)[1])
    raise ValueError(
        f"unknown strategy '{text}'; use tier:<name>, router, chain, execute:<tier>, "
        "crosscheck:<tier A>+<tier B>, auto or auto:check (closed-book suite) or "
        "openbook, stuffed:<tier> or closed:<tier> (open-book suite)"
    )


def load_golden(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_text = Path(path).read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw_text)
    golden_info = {
        "path": str(path),
        "version": parsed.get("version"),
        "sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
    }
    return parsed["cases"], golden_info


def filter_cases(
    cases: list[dict[str, Any]], case_ids: list[str] | None, categories: list[str] | None
) -> list[dict[str, Any]]:
    selected = cases
    if case_ids:
        wanted_ids = set(case_ids)
        selected = [case for case in selected if case["id"] in wanted_ids]
    if categories:
        wanted_categories = set(categories)
        selected = [case for case in selected if case["category"] in wanted_categories]
    return selected


class QuestionSetError(Exception):
    """The open-book question file does not fit the corpus it is graded against."""


ALL_SOURCES = "all"


def validate_openbook_cases(cases: list[dict[str, Any]], corpus: dict[str, dict[str, Any]]) -> list[str]:
    """Problems that would make a grade meaningless: unknown documents, evidence not in the source."""
    problems: list[str] = []
    for case in cases:
        case_id = case.get("id", "?")
        sources = case.get("sources") or []
        if not sources:
            problems.append(f"{case_id}: no sources")
        for document_id in sources:
            if document_id not in corpus:
                problems.append(f"{case_id}: source '{document_id}' is not in the corpus manifest")
        evidence = case.get("evidence") or []
        if case.get("answerable", True) and not evidence:
            problems.append(f"{case_id}: answerable case without an evidence quote")
        if not case.get("answerable", True) and evidence:
            problems.append(f"{case_id}: unanswerable case must not have evidence")
        for item in evidence:
            document_id = item.get("document")
            if document_id not in sources:
                problems.append(f"{case_id}: evidence document '{document_id}' is not among its sources")
                continue
            if document_id not in corpus:
                continue
            quote = normalise_whitespace(str(item.get("quote", "")))
            if not quote or quote not in normalise_whitespace(corpus[document_id]["text"]):
                problems.append(f"{case_id}: evidence quote not found verbatim in '{document_id}'")
    return problems


def load_openbook(
    questions_path: str | Path, manifest_path: str | Path, corpus_root: str | Path = fetch_corpus.REPO_ROOT
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    """Load the open-book cases and the verified corpus they are graded against.

    Raises fetch_corpus.CorpusError when a corpus file is missing or differs from the
    manifest, and QuestionSetError when a case does not fit the corpus.
    """
    corpus = fetch_corpus.verify_corpus(manifest_path, corpus_root)
    cases, golden_info = load_golden(questions_path)
    for case in cases:
        # `sources: all` supplies every document of the manifest, for cases that are asked
        # of a whole corpus too large to list by hand.
        if case.get("sources") == ALL_SOURCES:
            case["sources"] = list(corpus)
    problems = validate_openbook_cases(cases, corpus)
    if problems:
        raise QuestionSetError(f"{questions_path} does not fit the corpus:\n  " + "\n  ".join(problems))
    golden_info["corpus_manifest"] = str(manifest_path)
    golden_info["corpus_fingerprint"] = fetch_corpus.corpus_fingerprint(corpus)
    return cases, golden_info, corpus


def same_question_set(prior_golden: dict[str, Any], golden_info: dict[str, Any]) -> bool:
    """True when an earlier result file was graded against these questions and this corpus."""
    return prior_golden.get("sha256") == golden_info["sha256"] and prior_golden.get(
        "corpus_fingerprint"
    ) == golden_info.get("corpus_fingerprint")


def chain_mode_supported() -> bool:
    return "mode" in AskRequest.model_fields


def harness_mode_supported(mode_name: str) -> bool:
    """True when AskRequest accepts this mode and a sources list."""
    mode_field = AskRequest.model_fields.get("mode")
    if mode_field is None or "sources" not in AskRequest.model_fields:
        return False
    return mode_name in get_args(mode_field.annotation)


def unsupported_reason(strategy: Strategy) -> str | None:
    """Why the harness cannot run this strategy yet, or None when it can."""
    if strategy.kind == KIND_CHAIN and not chain_mode_supported():
        return "AskRequest has no 'mode' field yet; chain mode is not available"
    if strategy.kind in (KIND_EXECUTE, KIND_CROSSCHECK) and not harness_mode_supported(MODE_EXECUTE):
        return "AskRequest does not accept mode 'execute' yet; execute mode is not available in this checkout"
    if strategy.kind == KIND_CROSSCHECK and "cross_check_tier" not in AskRequest.model_fields:
        return "AskRequest has no 'cross_check_tier' field yet; the cross-check is not available in this checkout"
    if strategy.kind in (KIND_AUTO, KIND_AUTO_CHECK) and "computation_check" not in AskRequest.model_fields:
        return "AskRequest has no 'computation_check' field yet; the computation check is not available in this checkout"
    if strategy.kind == KIND_OPENBOOK and not harness_mode_supported(MODE_OPENBOOK):
        return (
            "AskRequest does not accept mode 'openbook' with sources yet; the grounded pipeline "
            "is not available in this checkout"
        )
    return None


def build_stuffed_prompt(question: str, documents: dict[str, dict[str, Any]]) -> str:
    blocks = [
        f'<source id="{document_id}">\n{document["text"]}\n</source>'
        for document_id, document in documents.items()
    ]
    return STUFFED_PROMPT_HEADER + "\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def build_openbook_request(
    strategy: Strategy,
    case: dict[str, Any],
    documents: dict[str, dict[str, Any]],
    verifier_tier: str | None = None,
) -> tuple[Any, str | None]:
    """The request for one open-book suite strategy. Returns (request, who pasted the sources).

    verifier_tier applies to the openbook strategy only: it replaces the configured
    grounding verifier.
    """
    source_paths = [document["path"] for document in documents.values()]
    if strategy.kind == KIND_OPENBOOK:
        overrides: dict[str, Any] = {"mode": MODE_OPENBOOK, "sources": source_paths}
        if verifier_tier is not None:
            overrides["verifier_tier"] = verifier_tier
        return AskRequest.from_question(case["question"], **overrides), None
    if strategy.kind == KIND_STUFFED:
        if harness_mode_supported(MODE_STUFFED):
            request = AskRequest.from_question(
                case["question"], tier=strategy.tier, mode=MODE_STUFFED, sources=source_paths
            )
            return request, STUFFED_BY_HARNESS
        prompt = build_stuffed_prompt(case["question"], documents)
        return AskRequest.from_question(prompt, tier=strategy.tier, mode="route"), STUFFED_BY_RUNNER
    return AskRequest.from_question(case["question"], tier=strategy.tier, mode="route"), None


def build_request(strategy: Strategy, question: str, verifier_tier: str | None) -> AskRequest:
    overrides: dict[str, Any] = {}
    if strategy.kind in (KIND_TIER, KIND_EXECUTE, KIND_CROSSCHECK):
        overrides["tier"] = strategy.tier
    if strategy.kind == KIND_TIER and "no_fallback" in AskRequest.model_fields:
        # A fallback answer is excluded from this tier's score anyway; failing fast saves
        # the fallback call (up to its full timeout, and its cost on a paid tier).
        overrides["no_fallback"] = True
    if strategy.kind == KIND_EXECUTE:
        overrides["mode"] = MODE_EXECUTE
        if "cross_check" in AskRequest.model_fields:
            # execute:<tier> measures one program, whatever execute.cross_check says.
            overrides["cross_check"] = False
    elif strategy.kind == KIND_CROSSCHECK:
        overrides.update(mode=MODE_EXECUTE, cross_check=True, cross_check_tier=strategy.cross_check_tier)
    elif strategy.kind in (KIND_AUTO, KIND_AUTO_CHECK):
        overrides.update(mode="auto", computation_check=strategy.kind == KIND_AUTO_CHECK)
    elif chain_mode_supported():
        overrides["mode"] = "chain" if strategy.kind == KIND_CHAIN else "route"
    if (
        strategy.kind == KIND_CHAIN
        and verifier_tier is not None
        and "verifier_tier" in AskRequest.model_fields
    ):
        overrides["verifier_tier"] = verifier_tier
    return AskRequest.from_question(question, **overrides)


def _measured_tier_error(result: Any, measured_tier: str | None) -> str | None:
    """The error of the measured tier's own call, when the run fell back off it."""
    for call in getattr(result, "calls", None) or []:
        if call.tier == measured_tier and call.status != "ok":
            return call.error_type or call.status
    return None


def run_case(
    harness: Any,
    strategy: Strategy,
    case: dict[str, Any],
    judge: RubricJudge | None,
    verifier_tier: str | None = None,
) -> dict[str, Any]:
    """Run one strategy on one case and grade it. Never raises for a harness failure."""
    record = blank_record(strategy, case)

    started_at = time.monotonic()
    try:
        result = harness.answer(build_request(strategy, case["question"], verifier_tier))
    except Exception as harness_error:  # the eval must survive one broken run
        record["error_type"] = "exception"
        record["error_message"] = repr(harness_error)[:300]
        record["latency_ms"] = round((time.monotonic() - started_at) * 1000, 1)
        return record

    copy_result_fields(record, result)
    if excluded_as_fallback(record, strategy, result):
        return record

    if result.status != "ok" or result.answer is None:
        record["outcome"] = OUTCOME_ERROR
        return record

    if record.get("cross_check"):
        grade_each_program(record, case)
    if record.get("execution_status") == STATUS_DISPUTED:
        # A disputed cross-check returns both outputs and no answer. It is scored as a
        # fail without asking the grader: a text that lists two values can hold the
        # expected one and still not be an answer.
        record.update(
            outcome=OUTCOME_FAIL, grade=OUTCOME_FAIL,
            grade_reason="disputed: the two programs disagreed, so no answer was returned",
        )
        return record
    apply_grade(record, grade_answer(case, result.answer, judge))
    return record


def grade_each_program(record: dict[str, Any], case: dict[str, Any]) -> None:
    """Grade each program's own output of a cross-check, for the cases a judge is not needed
    for. Says whether A or B was right when they disagreed. Not part of the run's outcome."""
    if case["match"]["type"] not in PER_PROGRAM_MATCH_TYPES:
        return
    for field, grade_field in (("executed_answer", "program_a_grade"), ("executed_answer_b", "program_b_grade")):
        value = record.get(field)
        if value is not None:
            record[grade_field] = grade_answer(case, f"Answer: {value}", None).status


def blank_record(strategy: Strategy, case: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy": strategy.name,
        "case_id": case["id"],
        "category": case["category"],
        "match_type": case["match"]["type"],
        "outcome": OUTCOME_ERROR,
        "answer": None,
        "grade": None,
        "grade_reason": None,
        "grade_detail": None,
        "selected_tier": None,
        "final_tier": None,
        "final_model": None,
        "fallback_used": False,
        "fell_back_off_measured_tier": False,
        "measured_tier_error": None,
        "triage_label": None,
        "triage_confidence": None,
        "route_reason": None,
        "mode": None,
        "verification_status": None,
        "verification_critique": None,
        "repaired": None,
        "schema_escalated": None,
        "verifier_tier": None,
        "repair_error_type": None,
        # execute strategies only: verified | corrected | executed | unverified | failed, and the
        # two answers compared (claimed_answer is None when the model declined to claim).
        "execution_status": None,
        "claimed_answer": None,
        "executed_answer": None,
        # crosscheck strategies only: B's tier, B's output, how the two were compared, and
        # each program's own grade (numeric, exact and contains cases; None otherwise).
        "cross_check": None,
        "cross_check_tier": None,
        "executed_answer_b": None,
        "cross_check_rule": None,
        "program_a_grade": None,
        "program_b_grade": None,
        # auto strategies only: why auto picked the mode, the computation check's verdict
        # (yes | no | low_confidence | failed | not_run), its confidence and the latency of
        # its call, and the case's own `computable` label when the question file has one.
        "auto_reason": None,
        "computation_verdict": None,
        "computation_check_tier": None,
        "computation_check_confidence": None,
        "computation_check_latency_ms": None,
        "expected_computable": case.get("computable"),
        "failed_call_messages": [],
        "escalation_attempted": None,
        "escalation_cost_usd": None,
        "latency_ms": None,
        "cost_usd": 0.0,
        "judge_tier": None,
        "judge_model": None,
        "judge_cost_usd": 0.0,
        "judge_latency_ms": None,
        "self_graded": False,
        "error_type": None,
        "error_message": None,
        "trace_id": None,
    }


def escalation_calls(calls: list[Any]) -> list[Any]:
    return [call for call in calls if getattr(call, "role", None) in ESCALATION_CALL_ROLES]


def check_call_latency_ms(calls: list[Any]) -> float | None:
    for call in calls:
        if getattr(call, "role", None) == ROLE_CHECK:
            return call.latency_ms
    return None


def copy_result_fields(record: dict[str, Any], result: Any) -> None:
    stronger_tier_calls = escalation_calls(getattr(result, "calls", []))
    record.update(
        auto_reason=getattr(result, "auto_reason", None),
        computation_verdict=getattr(result, "computation_verdict", None),
        computation_check_tier=getattr(result, "computation_check_tier", None),
        computation_check_confidence=getattr(result, "computation_check_confidence", None),
        computation_check_latency_ms=check_call_latency_ms(getattr(result, "calls", None) or []),
        escalation_attempted=bool(stronger_tier_calls),
        escalation_cost_usd=sum(getattr(call, "cost_usd", None) or 0.0 for call in stronger_tier_calls),
        answer=result.answer,
        selected_tier=result.selected_tier,
        final_tier=result.final_tier,
        final_model=result.final_model,
        fallback_used=bool(result.fallback_used),
        triage_label=result.triage_label,
        triage_confidence=result.triage_confidence,
        route_reason=result.route_reason,
        mode=getattr(result, "mode", None),
        verification_status=getattr(result, "verification_status", None),
        verification_critique=getattr(result, "verification_critique", None),
        repaired=getattr(result, "repaired", None),
        schema_escalated=getattr(result, "schema_escalated", None),
        verifier_tier=getattr(result, "verifier_tier", None),
        repair_error_type=getattr(result, "repair_error_type", None),
        execution_status=getattr(result, "execution_status", None),
        claimed_answer=getattr(result, "claimed_answer", None),
        executed_answer=getattr(result, "executed_answer", None),
        cross_check=getattr(result, "cross_check", None),
        cross_check_tier=getattr(result, "cross_check_tier", None),
        executed_answer_b=getattr(result, "executed_answer_b", None),
        cross_check_rule=getattr(result, "cross_check_rule", None),
        # Why each failed model call failed (provider text), so a refused escalation is diagnosable.
        failed_call_messages=[
            f"{call.role} {call.tier}: {call.error_message}"
            for call in getattr(result, "calls", [])
            if getattr(call, "error_message", None)
        ],
        latency_ms=result.total_latency_ms,
        cost_usd=result.total_cost_usd or 0.0,
        error_type=result.error_type,
        error_message=result.error_message,
        trace_id=result.trace_id,
    )


def excluded_as_fallback(record: dict[str, Any], strategy: Strategy, result: Any) -> bool:
    """Mark a forced-tier run that was answered by another tier. Returns True when excluded."""
    if strategy.kind not in FORCED_TIER_KINDS:
        return False
    if not result.fallback_used and result.final_tier == strategy.tier:
        return False
    # The answer, if any, came from another tier: it says nothing about this one.
    record["outcome"] = OUTCOME_FALLBACK_EXCLUDED
    record["fell_back_off_measured_tier"] = True
    record["measured_tier_error"] = _measured_tier_error(result, strategy.tier)
    return True


def _plain_citations(citations: Any) -> list[dict[str, Any]]:
    plain: list[dict[str, Any]] = []
    for citation in citations or []:
        if isinstance(citation, dict):
            plain.append(dict(citation))
        elif hasattr(citation, "model_dump"):
            plain.append(citation.model_dump())
        else:
            plain.append(
                {field: getattr(citation, field, None) for field in ("source", "passage_id", "quote")}
            )
    return plain


def case_documents(case: dict[str, Any], corpus: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {document_id: corpus[document_id] for document_id in case["sources"]}


def run_openbook_case(
    harness: Any,
    strategy: Strategy,
    case: dict[str, Any],
    judge: RubricJudge | None,
    corpus: dict[str, dict[str, Any]],
    verifier_tier: str | None = None,
) -> dict[str, Any]:
    """Run one open-book suite strategy on one case and grade it on four separate axes."""
    record = blank_record(strategy, case)
    record.update(
        suite=SUITE_OPENBOOK,
        answerable=bool(case.get("answerable", True)),
        sources=list(case["sources"]),
        stuffed_by=None,
        abstained_flag=None,
        abstained=None,
        citations=[],
        grounding_status=None,
        passages_retrieved=None,
        answer_grade=None,
        answer_grade_reason=None,
        abstention_grade=None,
        abstention_reason=None,
        abstention_decided_by=None,
        citation_validity=None,
        citation_validity_reason=None,
        evidence_hit=None,
        evidence_hit_reason=None,
    )
    documents = case_documents(case, corpus)

    started_at = time.monotonic()
    try:
        request, stuffed_by = build_openbook_request(strategy, case, documents, verifier_tier)
        record["stuffed_by"] = stuffed_by
        result = harness.answer(request)
    except Exception as harness_error:  # the eval must survive one broken run
        record["error_type"] = "exception"
        record["error_message"] = repr(harness_error)[:300]
        record["latency_ms"] = round((time.monotonic() - started_at) * 1000, 1)
        return record

    copy_result_fields(record, result)
    record.update(
        abstained_flag=getattr(result, "abstained", None),
        citations=_plain_citations(getattr(result, "citations", None)),
        grounding_status=getattr(result, "grounding_status", None),
        passages_retrieved=getattr(result, "passages_retrieved", None),
    )
    if excluded_as_fallback(record, strategy, result):
        return record

    # An abstention may come back with no answer text; that is a result, not an error.
    if result.status != "ok" or (result.answer is None and not record["abstained_flag"]):
        record["outcome"] = OUTCOME_ERROR
        return record

    apply_openbook_grade(record, case, documents, judge)
    return record


def apply_openbook_grade(
    record: dict[str, Any],
    case: dict[str, Any],
    documents: dict[str, dict[str, Any]],
    judge: RubricJudge | None,
) -> None:
    """Grade a record's stored answer, abstained flag and citations. Used by runs and regrades.

    outcome is the per-case result used for the head-to-head: the answer grade on an
    answerable case, the abstention grade on an unanswerable one. The four grades stay in
    their own fields.
    """
    grade: OpenBookGrade = grade_openbook(
        case,
        record["answer"],
        abstained=record.get("abstained_flag"),
        citations=record.get("citations"),
        documents=documents,
        judge=judge,
        citations_expected=record["strategy"] == KIND_OPENBOOK,
    )
    deciding = grade.answer if record["answerable"] else grade.abstention
    judge_grades = [item for item in (grade.answer, grade.abstention) if item.judge_model is not None]
    record.update(
        outcome=deciding.status,
        grade=deciding.status,
        grade_reason=deciding.reason,
        grade_detail=deciding.detail,
        abstained=grade.abstained,
        answer_grade=grade.answer.status,
        answer_grade_reason=grade.answer.reason,
        abstention_grade=grade.abstention.status,
        abstention_reason=grade.abstention.reason,
        abstention_decided_by=grade.abstention.detail.get("decided_by"),
        citation_validity=grade.citation_validity.status,
        citation_validity_reason=grade.citation_validity.reason,
        citation_validity_detail=grade.citation_validity.detail,
        evidence_hit=grade.evidence_hit.status,
        evidence_hit_reason=grade.evidence_hit.reason,
        evidence_hit_detail=grade.evidence_hit.detail,
        judge_tier=judge_grades[0].judge_tier if judge_grades else None,
        judge_model=judge_grades[0].judge_model if judge_grades else None,
        judge_cost_usd=grade.judge_cost_usd,
        judge_latency_ms=grade.judge_latency_ms,
        self_graded=bool(judge_grades) and judge_grades[0].judge_model == record["final_model"],
    )


def regrade_openbook_records(
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    judge: RubricJudge | None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Grade stored open-book answers and citations again. No model answer is re-run."""
    cases_by_id = {case["id"]: case for case in cases}
    regraded: list[dict[str, Any]] = []
    for record in records:
        updated = dict(record)
        case = cases_by_id.get(record["case_id"])
        has_result = record["answer"] or record.get("abstained_flag")
        if case is not None and record["outcome"] in ANSWERED_OUTCOMES and has_result:
            apply_openbook_grade(updated, case, case_documents(case, corpus), judge)
        regraded.append(updated)
        if progress is not None:
            progress(updated)
    return regraded


def apply_grade(record: dict[str, Any], grade: Grade) -> None:
    record.update(
        outcome=grade.status,
        grade=grade.status,
        grade_reason=grade.reason,
        grade_detail=grade.detail,
        judge_tier=grade.judge_tier,
        judge_model=grade.judge_model,
        judge_cost_usd=grade.judge_cost_usd,
        judge_latency_ms=grade.judge_latency_ms,
        self_graded=grade.judge_model is not None and grade.judge_model == record["final_model"],
    )


def regrade_records(
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    judge: RubricJudge | None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Grade stored answers again with the current grader. No model answer is re-run.

    Error and fallback-excluded records have no answer to grade and are kept as they are.
    """
    cases_by_id = {case["id"]: case for case in cases}
    regraded: list[dict[str, Any]] = []
    for record in records:
        updated = dict(record)
        case = cases_by_id.get(record["case_id"])
        if record.get("execution_status") == STATUS_DISPUTED:
            pass  # scored as a fail when it ran (run_case): there is no answer to grade again
        elif case is not None and record["outcome"] in ANSWERED_OUTCOMES and record["answer"]:
            apply_grade(updated, grade_answer(case, record["answer"], judge))
        regraded.append(updated)
        if progress is not None:
            progress(updated)
    return regraded


def run_strategies(
    cases: list[dict[str, Any]],
    strategies: list[Strategy],
    harness_factory: HarnessFactory,
    judge: RubricJudge | None,
    max_cost_usd: float | None = None,
    verifier_tier: str | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    case_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run every strategy over every case. Stops cleanly once the cost cap is passed.

    case_runner(harness, strategy, case, judge) replaces the closed-book run_case; the
    open-book suite passes one bound to its corpus.
    """
    records: list[dict[str, Any]] = []
    not_runnable: dict[str, str] = {}
    cumulative_cost_usd = 0.0
    stopped_early = False
    stop_reason: str | None = None

    for strategy in strategies:
        if stopped_early:
            break
        reason = unsupported_reason(strategy)
        if reason is not None:
            not_runnable[strategy.name] = reason
            continue
        try:
            harness = harness_factory(strategy)
        except Exception as build_error:
            not_runnable[strategy.name] = f"harness could not be built: {build_error!r}"[:300]
            continue

        for case in cases:
            if max_cost_usd is not None and cumulative_cost_usd > max_cost_usd:
                stopped_early = True
                stop_reason = (
                    f"cost cap passed: {cumulative_cost_usd:.4f} USD spent, cap {max_cost_usd:.4f} USD; "
                    f"stopped before {strategy.name} / {case['id']}"
                )
                break
            try:
                if case_runner is not None:
                    record = case_runner(harness, strategy, case, judge)
                else:
                    record = run_case(harness, strategy, case, judge, verifier_tier)
            except KeyboardInterrupt:
                stopped_early = True
                stop_reason = f"interrupted during {strategy.name} / {case['id']}"
                break
            records.append(record)
            cumulative_cost_usd += record["cost_usd"] + record["judge_cost_usd"]
            if progress is not None:
                progress(record)

    return {
        "records": records,
        "not_runnable": not_runnable,
        "cumulative_cost_usd": round(cumulative_cost_usd, 8),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
    }


def percentile(values: list[float], percent: float) -> float | None:
    """Nearest-rank percentile: the smallest value with at least percent% of values at or below it."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percent / 100 * len(ordered)))
    return ordered[rank - 1]


def wilson_interval(passed: int, graded: int) -> tuple[float, float] | None:
    if graded == 0:
        return None
    proportion = passed / graded
    z_squared = WILSON_Z_95**2
    denominator = 1 + z_squared / graded
    centre = (proportion + z_squared / (2 * graded)) / denominator
    half_width = (
        WILSON_Z_95
        * math.sqrt(proportion * (1 - proportion) / graded + z_squared / (4 * graded**2))
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def _accuracy(passed: int, graded: int) -> float | None:
    if graded == 0:
        return None
    return passed / graded


def escalation_outcome(record: dict[str, Any]) -> str | None:
    """`succeeded` when the stronger tier's answer was used, `failed` when it was called and its answer was not.

    Older records carry no `escalation_attempted`; for those a recorded `repair_error_type` marks the failed call.
    """
    if record.get("repaired"):
        return ESCALATION_SUCCEEDED
    if record.get("escalation_attempted") or record.get("repair_error_type"):
        return ESCALATION_FAILED
    return None


def summarise_escalations(own_records: list[dict[str, Any]]) -> dict[str, Any]:
    succeeded = [record for record in own_records if escalation_outcome(record) == ESCALATION_SUCCEEDED]
    failed = [record for record in own_records if escalation_outcome(record) == ESCALATION_FAILED]

    def total_cost(escalated: list[dict[str, Any]]) -> float | None:
        # None when a record has no per-call cost (written before the field existed, telemetry not found).
        if any(record.get("escalation_cost_usd") is None for record in escalated):
            return None
        return round(sum(record["escalation_cost_usd"] for record in escalated), 8)

    return {
        "escalations_attempted": len(succeeded) + len(failed),
        "escalations_succeeded": len(succeeded),
        "escalations_failed": len(failed),
        "escalation_cost_usd": total_cost(succeeded + failed),
        "failed_escalation_cost_usd": total_cost(failed),
        "failed_escalation_cases": [record["case_id"] for record in failed],
    }


def backfill_escalation_costs(records: list[dict[str, Any]], db_path: str | Path) -> int:
    """Fill `escalation_attempted` and `escalation_cost_usd` on older records from the telemetry calls table."""
    missing = [
        record for record in records if record.get("escalation_cost_usd") is None and record.get("trace_id")
    ]
    if not missing or not Path(db_path).exists():
        return 0
    filled = 0
    placeholders = ", ".join("?" for _ in ESCALATION_CALL_ROLES)
    connection = sqlite3.connect(str(db_path))
    try:
        for record in missing:
            run_known = connection.execute(
                "SELECT 1 FROM runs WHERE trace_id = ?", (record["trace_id"],)
            ).fetchone()
            if run_known is None:
                continue
            call_costs = connection.execute(
                f"SELECT cost_usd FROM calls WHERE trace_id = ? AND role IN ({placeholders})",
                (record["trace_id"], *ESCALATION_CALL_ROLES),
            ).fetchall()
            record["escalation_attempted"] = bool(call_costs)
            record["escalation_cost_usd"] = sum(cost or 0.0 for (cost,) in call_costs)
            filled += 1
    except sqlite3.Error:
        return filled
    finally:
        connection.close()
    return filled


def summarise_execution(own_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Verified / corrected / executed / unverified / failed counts of an execute strategy, each with how many
    of them the grader passed. None for a strategy that did not run in execute mode."""
    if not any(record.get("mode") == MODE_EXECUTE for record in own_records):
        return None
    execution: dict[str, Any] = {}
    for status in EXECUTION_STATUSES:
        with_status = [record for record in own_records if record.get("execution_status") == status]
        execution[status] = {
            "count": len(with_status),
            "pass": sum(1 for record in with_status if record["outcome"] == OUTCOME_PASS),
            "graded": sum(1 for record in with_status if record["outcome"] in COUNTED_OUTCOMES),
        }
    execution["repaired"] = sum(1 for record in own_records if record.get("repaired"))
    disputed = [record for record in own_records if record.get("execution_status") == STATUS_DISPUTED]
    if any(record.get("cross_check") for record in own_records):
        graded_both = [
            record for record in disputed
            if record.get("program_a_grade") is not None and record.get("program_b_grade") is not None
        ]
        execution["disputed_programs"] = {
            "a_right": sum(
                1 for record in graded_both
                if record["program_a_grade"] == OUTCOME_PASS and record["program_b_grade"] != OUTCOME_PASS
            ),
            "b_right": sum(
                1 for record in graded_both
                if record["program_b_grade"] == OUTCOME_PASS and record["program_a_grade"] != OUTCOME_PASS
            ),
            "both_right": sum(
                1 for record in graded_both
                if record["program_a_grade"] == OUTCOME_PASS and record["program_b_grade"] == OUTCOME_PASS
            ),
            "neither_right": sum(
                1 for record in graded_both
                if record["program_a_grade"] != OUTCOME_PASS and record["program_b_grade"] != OUTCOME_PASS
            ),
            "not_graded_per_program": len(disputed) - len(graded_both),
        }
    return execution


def summarise_computation_check(own_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """What the computation check did over a strategy's runs. None when it never ran.

    `sent_to_execute` counts the runs the check answered yes for, by category. The confusion
    counts compare that with each case's `computable` label and cover labelled cases only.
    `execute_fallbacks` are the yes runs where nothing could be executed and the answer is
    the closed-book call. `added_latency` is the check call's own latency on the runs that
    were not sent to execute mode: what the check cost a question it did not help.
    """
    checked = [record for record in own_records if record.get("computation_verdict") in COMPUTATION_VERDICTS]
    if not checked:
        return None
    sent = [record for record in checked if record["computation_verdict"] == VERDICT_YES]
    sent_by_category: dict[str, int] = {}
    for record in sent:
        sent_by_category[record["category"]] = sent_by_category.get(record["category"], 0) + 1
    labelled = [record for record in checked if record.get("expected_computable") is not None]

    def cases_where(expected: bool, said_yes: bool) -> list[str]:
        return [
            record["case_id"]
            for record in labelled
            if bool(record["expected_computable"]) is expected
            and (record["computation_verdict"] == VERDICT_YES) is said_yes
        ]

    not_sent_latencies = [
        record["computation_check_latency_ms"]
        for record in checked
        if record["computation_verdict"] != VERDICT_YES and record.get("computation_check_latency_ms") is not None
    ]
    return {
        "checked": len(checked),
        "verdicts": {
            verdict: sum(1 for record in checked if record["computation_verdict"] == verdict)
            for verdict in COMPUTATION_VERDICTS
        },
        "sent_to_execute": len(sent),
        "sent_to_execute_by_category": sent_by_category,
        "execute_fallbacks": [
            record["case_id"]
            for record in sent
            if record.get("mode") != MODE_EXECUTE
        ],
        "labelled": len(labelled),
        "true_positive": len(cases_where(True, True)),
        "false_positive_cases": cases_where(False, True),
        "false_negative_cases": cases_where(True, False),
        "true_negative": len(cases_where(False, False)),
        "added_latency_p50_ms": percentile(not_sent_latencies, 50),
        "added_latency_p95_ms": percentile(not_sent_latencies, 95),
    }


def summarise_strategy(strategy_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    own_records = [record for record in records if record["strategy"] == strategy_name]
    counts = {
        outcome: sum(1 for record in own_records if record["outcome"] == outcome)
        for outcome in (
            OUTCOME_PASS,
            OUTCOME_FAIL,
            OUTCOME_UNGRADED,
            OUTCOME_ERROR,
            OUTCOME_FALLBACK_EXCLUDED,
        )
    }
    graded = counts[OUTCOME_PASS] + counts[OUTCOME_FAIL]
    interval = wilson_interval(counts[OUTCOME_PASS], graded)
    total_cost_usd = sum(record["cost_usd"] for record in own_records)
    latencies = [
        record["latency_ms"]
        for record in own_records
        if record["outcome"] in ANSWERED_OUTCOMES and record["latency_ms"] is not None
    ]

    by_category: dict[str, dict[str, Any]] = {}
    for record in own_records:
        category = by_category.setdefault(record["category"], {"cases": 0, "pass": 0, "graded": 0})
        category["cases"] += 1
        if record["outcome"] in COUNTED_OUTCOMES:
            category["graded"] += 1
        if record["outcome"] == OUTCOME_PASS:
            category["pass"] += 1
    for category in by_category.values():
        category["accuracy"] = _accuracy(category["pass"], category["graded"])

    return {
        "strategy": strategy_name,
        "cases": len(own_records),
        "pass": counts[OUTCOME_PASS],
        "fail": counts[OUTCOME_FAIL],
        "ungraded": counts[OUTCOME_UNGRADED],
        "error": counts[OUTCOME_ERROR],
        "fallback_excluded": counts[OUTCOME_FALLBACK_EXCLUDED],
        "graded": graded,
        "accuracy": _accuracy(counts[OUTCOME_PASS], graded),
        "accuracy_ci95": list(interval) if interval else None,
        "total_cost_usd": round(total_cost_usd, 8),
        "cost_per_question_usd": round(total_cost_usd / len(own_records), 8) if own_records else None,
        "judge_cost_usd": round(sum(record["judge_cost_usd"] for record in own_records), 8),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "fallback_used": sum(1 for record in own_records if record["fallback_used"]),
        "self_graded": sum(1 for record in own_records if record["self_graded"]),
        "adjudicated": sum(
            1 for record in own_records if (record.get("grade_detail") or {}).get("adjudicated")
        ),
        **summarise_escalations(own_records),
        "execution": summarise_execution(own_records),
        "computation_check": summarise_computation_check(own_records),
        "by_category": by_category,
    }


def head_to_head(
    strategy_name: str, baseline_name: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare one strategy with the baseline on the cases both have a pass or fail for."""
    by_key = {(record["strategy"], record["case_id"]): record for record in records}
    case_ids = sorted({record["case_id"] for record in records})
    result = {
        "strategy": strategy_name,
        "baseline": baseline_name,
        "won": 0,
        "lost": 0,
        "both_right": 0,
        "both_wrong": 0,
        "not_comparable": 0,
        # Not-comparable cases where the strategy answered correctly and the baseline run
        # ended in an error (for example a stuffed prompt above the context window). They
        # are reported on their own and never added to "won".
        "right_where_baseline_errored": 0,
        "won_cases": [],
        "lost_cases": [],
    }
    strategy_cost = 0.0
    baseline_cost = 0.0
    for case_id in case_ids:
        own = by_key.get((strategy_name, case_id))
        base = by_key.get((baseline_name, case_id))
        if own is not None and base is not None:
            strategy_cost += own["cost_usd"]
            baseline_cost += base["cost_usd"]
        if (
            own is None
            or base is None
            or own["outcome"] not in COUNTED_OUTCOMES
            or base["outcome"] not in COUNTED_OUTCOMES
        ):
            result["not_comparable"] += 1
            if (
                own is not None
                and base is not None
                and own["outcome"] == OUTCOME_PASS
                and base["outcome"] == OUTCOME_ERROR
            ):
                result["right_where_baseline_errored"] += 1
            continue
        own_passed = own["outcome"] == OUTCOME_PASS
        base_passed = base["outcome"] == OUTCOME_PASS
        if own_passed and base_passed:
            result["both_right"] += 1
        elif own_passed:
            result["won"] += 1
            result["won_cases"].append(case_id)
        elif base_passed:
            result["lost"] += 1
            result["lost_cases"].append(case_id)
        else:
            result["both_wrong"] += 1
    result["strategy_cost_usd"] = round(strategy_cost, 8)
    result["baseline_cost_usd"] = round(baseline_cost, 8)
    # Costs are summed over the cases both strategies ran. A zero baseline has no ratio.
    result["cost_ratio"] = round(strategy_cost / baseline_cost, 4) if baseline_cost > 0 else None
    return result


def build_scorecard(records: list[dict[str, Any]], baseline_name: str = BASELINE_STRATEGY) -> dict[str, Any]:
    strategy_names: list[str] = []
    for record in records:
        if record["strategy"] not in strategy_names:
            strategy_names.append(record["strategy"])
    comparisons = []
    if baseline_name in strategy_names:
        comparisons = [
            head_to_head(strategy_name, baseline_name, records)
            for strategy_name in strategy_names
            if strategy_name != baseline_name
        ]
    return {
        "baseline": baseline_name,
        "baseline_present": baseline_name in strategy_names,
        "strategies": [summarise_strategy(strategy_name, records) for strategy_name in strategy_names],
        "head_to_head": comparisons,
    }


def _count(records: list[dict[str, Any]], field: str, status: str) -> int:
    return sum(1 for record in records if record.get(field) == status)


def _rate_cell(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    """pass, fail and pass / (pass + fail) for one grade field; other statuses are left out."""
    passed = _count(records, field, GRADE_PASS)
    failed = _count(records, field, GRADE_FAIL)
    return {"pass": passed, "fail": failed, "graded": passed + failed, "rate": _accuracy(passed, passed + failed)}


def summarise_openbook_strategy(strategy_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """One strategy's open-book results. Every measure keeps its own numerator and denominator."""
    own_records = [record for record in records if record["strategy"] == strategy_name]
    graded_records = [record for record in own_records if record["outcome"] in ANSWERED_OUTCOMES]
    answerable = [record for record in graded_records if record["answerable"]]
    unanswerable = [record for record in graded_records if not record["answerable"]]

    answer = _rate_cell(answerable, "answer_grade")
    answer_interval = wilson_interval(answer["pass"], answer["graded"])
    abstention = _rate_cell(unanswerable, "abstention_grade")
    total_cost_usd = sum(record["cost_usd"] for record in own_records)
    latencies = [record["latency_ms"] for record in graded_records if record["latency_ms"] is not None]

    by_category: dict[str, dict[str, Any]] = {}
    for record in own_records:
        category = by_category.setdefault(record["category"], {"cases": 0, "pass": 0, "graded": 0})
        category["cases"] += 1
        if record["outcome"] in COUNTED_OUTCOMES:
            category["graded"] += 1
        if record["outcome"] == OUTCOME_PASS:
            category["pass"] += 1
    for category in by_category.values():
        category["accuracy"] = _accuracy(category["pass"], category["graded"])

    return {
        "strategy": strategy_name,
        "cases": len(own_records),
        "answerable_cases": sum(1 for record in own_records if record["answerable"]),
        "unanswerable_cases": sum(1 for record in own_records if not record["answerable"]),
        "answer": answer,
        "answer_accuracy": answer["rate"],
        "answer_accuracy_ci95": list(answer_interval) if answer_interval else None,
        "abstention": abstention,
        "abstention_accuracy": abstention["rate"],
        # Answerable cases the run declined, and unanswerable cases it answered anyway.
        "false_abstentions": sum(1 for record in answerable if record.get("abstained") is True),
        "hallucinated_answers": _count(unanswerable, "abstention_grade", GRADE_FAIL),
        "citation_validity": _rate_cell(graded_records, "citation_validity"),
        "evidence_hit": _rate_cell(answerable, "evidence_hit"),
        **summarise_escalations(own_records),
        "ungraded": _count(answerable, "answer_grade", GRADE_UNGRADED)
        + _count(unanswerable, "abstention_grade", GRADE_UNGRADED),
        "error": _count(own_records, "outcome", OUTCOME_ERROR),
        "fallback_excluded": _count(own_records, "outcome", OUTCOME_FALLBACK_EXCLUDED),
        "total_cost_usd": round(total_cost_usd, 8),
        "cost_per_question_usd": round(total_cost_usd / len(own_records), 8) if own_records else None,
        "judge_cost_usd": round(sum(record["judge_cost_usd"] for record in own_records), 8),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "self_graded": sum(1 for record in own_records if record["self_graded"]),
        "stuffed_by": sorted({record["stuffed_by"] for record in own_records if record.get("stuffed_by")}),
        "by_category": by_category,
    }


def build_openbook_scorecard(
    records: list[dict[str, Any]], baseline_name: str = OPENBOOK_BASELINE_STRATEGY
) -> dict[str, Any]:
    strategy_names: list[str] = []
    for record in records:
        if record["strategy"] not in strategy_names:
            strategy_names.append(record["strategy"])
    comparisons = []
    if baseline_name in strategy_names:
        comparisons = [
            head_to_head(strategy_name, baseline_name, records)
            for strategy_name in strategy_names
            if strategy_name != baseline_name
        ]
    return {
        "suite": SUITE_OPENBOOK,
        "baseline": baseline_name,
        "baseline_present": baseline_name in strategy_names,
        "strategies": [summarise_openbook_strategy(strategy_name, records) for strategy_name in strategy_names],
        "head_to_head": comparisons,
    }


def route_coverage_gaps(cases: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[str]:
    """Why these results cannot stand as the measured routes; empty when they can.

    The routes are derived from forced-tier runs of local, mid and frontier (and code on
    code cases). Results that lack one of those tiers, or cover only some cases, would
    turn measured routes into unresolved or provisional ones, so they must not replace a
    fuller file.
    """
    ran: dict[str, set[str]] = {}
    counted: set[str] = set()
    for record in records:
        if not record["strategy"].startswith("tier:"):
            continue
        tier_name = record["strategy"].split(":", 1)[1]
        ran.setdefault(tier_name, set()).add(record["case_id"])
        if record["outcome"] in COUNTED_OUTCOMES:
            counted.add(tier_name)

    gaps: list[str] = []
    needed_tiers = list(dict.fromkeys([*ROUTE_ORDER_GENERAL, *ROUTE_ORDER_CODE]))
    for tier_name in needed_tiers:
        if tier_name == CODE_CATEGORY:
            needed_cases = [case["id"] for case in cases if case["category"] == CODE_CATEGORY]
        else:
            needed_cases = [case["id"] for case in cases]
        if not needed_cases:
            continue
        missing_cases = [case_id for case_id in needed_cases if case_id not in ran.get(tier_name, set())]
        if len(missing_cases) == len(needed_cases):
            gaps.append(f"tier:{tier_name} has no results")
        elif missing_cases:
            gaps.append(f"tier:{tier_name} has no result for {len(missing_cases)} of {len(needed_cases)} cases")
        elif tier_name not in counted:
            gaps.append(f"tier:{tier_name} has no pass or fail result (every run errored or fell back)")
    return gaps


def derive_measured_routes(
    cases: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """For each case, the cheapest tier whose forced-tier run answered correctly."""
    tier_outcomes: dict[tuple[str, str], str] = {}
    for record in records:
        if record["strategy"].startswith("tier:"):
            tier_name = record["strategy"].split(":", 1)[1]
            tier_outcomes[(record["case_id"], tier_name)] = record["outcome"]

    routes: dict[str, dict[str, Any]] = {}
    for case in cases:
        order = ROUTE_ORDER_CODE if case["category"] == CODE_CATEGORY else ROUTE_ORDER_GENERAL
        outcomes = {
            tier_name: tier_outcomes.get((case["id"], tier_name), TIER_RESULT_MISSING) for tier_name in order
        }
        # A tier with no pass or fail (not run, error, fell back, ungraded) is missing.
        missing_tiers = [tier_name for tier_name in order if outcomes[tier_name] not in COUNTED_OUTCOMES]
        passing_tiers = [tier_name for tier_name in order if outcomes[tier_name] == OUTCOME_PASS]

        route: str | None = None
        acceptable: list[str] = []
        if passing_tiers:
            route = passing_tiers[0]
            route_strength = TIER_STRENGTH[route]
            acceptable = [
                tier_name for tier_name in passing_tiers if TIER_STRENGTH[tier_name] == route_strength
            ]
            cheaper_missing = [
                tier_name for tier_name in missing_tiers if TIER_STRENGTH[tier_name] < route_strength
            ]
            status = ROUTE_PROVISIONAL if cheaper_missing else ROUTE_MEASURED
        elif missing_tiers:
            status = ROUTE_UNRESOLVED
        else:
            status = ROUTE_NONE_CORRECT

        routes[case["id"]] = {
            "category": case["category"],
            "status": status,
            "route": route,
            "acceptable": acceptable,
            "tier_outcomes": outcomes,
            "missing_tiers": missing_tiers,
            "expected_route": case.get("expected_route"),
        }
    return routes


def summarise_routes(routes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, list[str]] = {}
    by_route: dict[str, int] = {}
    agrees_with_expected = 0
    resolved = 0
    for case_id, entry in routes.items():
        by_status.setdefault(entry["status"], []).append(case_id)
        if entry["route"] is not None:
            resolved += 1
            by_route[entry["route"]] = by_route.get(entry["route"], 0) + 1
            if entry["expected_route"] in entry["acceptable"]:
                agrees_with_expected += 1
    return {
        "cases": len(routes),
        "by_status": by_status,
        "by_route": by_route,
        "resolved": resolved,
        "hand_written_route_agrees": agrees_with_expected,
    }


def write_measured_routes(
    path: Path, routes: dict[str, dict[str, Any]], golden_info: dict[str, Any], sources: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "golden": golden_info,
        "sources": sources,
        "ordering": ROUTE_ORDERING_NOTE,
        "statuses": {
            ROUTE_MEASURED: "cheapest correct tier found and every cheaper tier measured wrong",
            ROUTE_PROVISIONAL: "a correct tier found but a cheaper tier has no pass/fail result",
            ROUTE_NONE_CORRECT: "every tier was measured and none answered correctly",
            ROUTE_UNRESOLVED: "no tier answered correctly so far and at least one tier has no result",
        },
        "summary": summarise_routes(routes),
        "routes": routes,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def merge_prior_records(
    new_records: list[dict[str, Any]], merge_paths: list[str], golden_info: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Add records from earlier result files for (strategy, case) pairs this run did not produce.

    Returns (all records, files used, warnings). A file graded against a different golden
    set is skipped. Later files win over earlier ones; this run wins over every file.
    """
    present = {(record["strategy"], record["case_id"]) for record in new_records}
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    used: list[str] = []
    warnings: list[str] = []
    for merge_path in merge_paths:
        try:
            prior = json.loads(Path(merge_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as read_error:
            warnings.append(f"skipped {merge_path}: {read_error!r}")
            continue
        if not same_question_set(prior.get("golden", {}), golden_info):
            warnings.append(f"skipped {merge_path}: it was graded against a different golden set")
            continue
        used.append(merge_path)
        for record in prior.get("records", []):
            key = (record["strategy"], record["case_id"])
            if key in present:
                continue
            merged[key] = {**record, "merged_from": merge_path}
    return [*merged.values(), *new_records], used, warnings


def print_table(headers: list[str], table_rows: list[list]) -> None:
    widths = [max(len(str(value)) for value in column) for column in zip(headers, *table_rows)]
    for line in [headers, ["-" * width for width in widths], *table_rows]:
        print("  ".join(str(value).ljust(width) for value, width in zip(line, widths)))


def _percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.0f}%"


def _milliseconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f}"


def _usd(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.4f}"


def print_escalation_costs(summaries: list[dict[str, Any]]) -> None:
    escalating = [summary for summary in summaries if summary["escalations_attempted"]]
    if not escalating:
        return
    print("\nEscalations to a stronger tier (a failed one is billed and its answer is not used)\n")
    print_table(
        ["strategy", "tried", "ok", "failed", "escalation usd", "of which failed usd", "failed cases"],
        [
            [
                summary["strategy"],
                summary["escalations_attempted"],
                summary["escalations_succeeded"],
                summary["escalations_failed"],
                _usd(summary["escalation_cost_usd"]),
                _usd(summary["failed_escalation_cost_usd"]),
                ", ".join(summary["failed_escalation_cases"]) or "-",
            ]
            for summary in escalating
        ],
    )


def _execution_count(summary: dict[str, Any], status: str) -> str:
    execution = summary.get("execution")
    return str(execution[status]["count"]) if execution else "-"


def print_execution(summaries: list[dict[str, Any]]) -> None:
    executing = [summary for summary in summaries if summary.get("execution")]
    if not executing:
        return
    print(
        "\nExecution (execute and crosscheck strategies; each cell is graded-pass / graded of the "
        "runs with that status. Cross-check: verified: two independently written programs printed "
        "matching values. disputed: both ran and the values differ; no answer is returned and the "
        "run is scored as a fail. one_sided: only one program ran, its output is returned as not "
        "cross-checked. One program: claim_matched: the output agreed with the claim of the model "
        "that wrote it. corrected: it disagreed and the executed output was returned. executed: it "
        "ran cleanly and the model gave no claim to compare. unverified: nothing ran cleanly, the "
        "model's claim was returned. failed: nothing ran cleanly and there was nothing to return, "
        "the run is an error; under auto:check it is then answered by the closed-book call and "
        "graded as that answer)\n"
    )
    rows = []
    for summary in executing:
        execution = summary["execution"]
        rows.append(
            [summary["strategy"]]
            + [f"{execution[status]['pass']}/{execution[status]['graded']}" for status in EXECUTION_STATUSES]
            + [execution["repaired"]]
        )
    print_table(["strategy", *EXECUTION_STATUSES, "answered by a repaired program"], rows)
    cross_checking = [summary for summary in executing if summary["execution"].get("disputed_programs")]
    if cross_checking:
        print(
            "\nDisputed cross-checks: which program held the expected value (each program's output "
            "graded on its own; numeric, exact and contains cases only)\n"
        )
        print_table(
            ["strategy", "disputed", "A right", "B right", "both right", "neither right", "not graded per program"],
            [
                [
                    summary["strategy"],
                    summary["execution"][STATUS_DISPUTED]["count"],
                    summary["execution"]["disputed_programs"]["a_right"],
                    summary["execution"]["disputed_programs"]["b_right"],
                    summary["execution"]["disputed_programs"]["both_right"],
                    summary["execution"]["disputed_programs"]["neither_right"],
                    summary["execution"]["disputed_programs"]["not_graded_per_program"],
                ]
                for summary in cross_checking
            ],
        )


def print_computation_check(summaries: list[dict[str, Any]]) -> None:
    checking = [summary for summary in summaries if summary.get("computation_check")]
    if not checking:
        return
    print(
        "\nComputation check (auto:check; sent = the check said yes and the question went to execute "
        "mode. TP, FP, FN, TN compare that with the case's `computable` label. fell back = sent, "
        "nothing could be executed, answered by the closed-book call as not verified. added "
        "latency = the check call on the runs it did not send to execute mode)\n"
    )
    rows = []
    for summary in checking:
        check = summary["computation_check"]
        by_category = ", ".join(
            f"{category} {count}" for category, count in sorted(check["sent_to_execute_by_category"].items())
        )
        rows.append(
            [
                summary["strategy"],
                check["checked"],
                check["sent_to_execute"],
                by_category or "-",
                check["true_positive"],
                ",".join(check["false_positive_cases"]) or "-",
                ",".join(check["false_negative_cases"]) or "-",
                check["true_negative"],
                check["verdicts"]["low_confidence"],
                check["verdicts"]["failed"],
                ",".join(check["execute_fallbacks"]) or "-",
                _milliseconds(check["added_latency_p50_ms"]),
                _milliseconds(check["added_latency_p95_ms"]),
            ]
        )
    print_table(
        ["strategy", "checked", "sent", "sent by category", "TP", "FP cases", "FN cases", "TN",
         "low confidence", "check failed", "fell back", "added p50 ms", "added p95 ms"],
        rows,
    )


def print_scorecard(scorecard: dict[str, Any], case_count: int) -> None:
    summaries = scorecard["strategies"]
    if not summaries:
        print("\nNo results to score.")
        return

    print("\nScorecard (accuracy = pass / (pass + fail); ungraded, error and fallback runs are left out)\n")
    table_rows = []
    for summary in summaries:
        interval = summary["accuracy_ci95"]
        interval_text = f"{interval[0] * 100:.0f}-{interval[1] * 100:.0f}%" if interval else "n/a"
        table_rows.append(
            [
                summary["strategy"],
                summary["cases"],
                summary["pass"],
                summary["fail"],
                summary["ungraded"],
                summary["error"],
                summary["fallback_excluded"],
                _percent(summary["accuracy"]),
                interval_text,
                f"{summary['total_cost_usd']:.4f}",
                f"{summary['cost_per_question_usd']:.5f}",
                _milliseconds(summary["latency_p50_ms"]),
                _milliseconds(summary["latency_p95_ms"]),
                *[_execution_count(summary, status) for status in EXECUTION_STATUSES],
            ]
        )
    print_table(
        ["strategy", "cases", "pass", "fail", "ungraded", "error", "fell back", "accuracy", "95% CI",
         "cost usd", "usd/q", "p50 ms", "p95 ms", *EXECUTION_STATUSES],
        table_rows,
    )
    print_escalation_costs(summaries)
    print_execution(summaries)
    print_computation_check(summaries)

    categories: list[str] = []
    for summary in summaries:
        for category in summary["by_category"]:
            if category not in categories:
                categories.append(category)
    print("\nAccuracy by category (pass/graded)\n")
    category_rows = []
    for summary in summaries:
        line = [summary["strategy"]]
        for category in categories:
            cell = summary["by_category"].get(category)
            line.append(f"{cell['pass']}/{cell['graded']}" if cell else "-")
        category_rows.append(line)
    print_table(["strategy", *categories], category_rows)

    baseline = scorecard["baseline"]
    if not scorecard["baseline_present"]:
        print(f"\nHead-to-head skipped: baseline {baseline} has no results (run it or --merge a file that has it).")
    elif scorecard["head_to_head"]:
        print(f"\nHead-to-head against {baseline} (cases where both have a pass or fail)\n")
        comparison_rows = []
        for comparison in scorecard["head_to_head"]:
            if comparison["cost_ratio"] is None:
                cost_ratio = "n/a (baseline cost 0)"
            else:
                cost_ratio = f"{comparison['cost_ratio']:.2f}x"
            comparison_rows.append(
                [
                    comparison["strategy"],
                    comparison["won"],
                    comparison["lost"],
                    comparison["both_right"],
                    comparison["both_wrong"],
                    comparison["not_comparable"],
                    comparison.get("right_where_baseline_errored", 0),
                    cost_ratio,
                    ",".join(comparison["won_cases"]) or "-",
                    ",".join(comparison["lost_cases"]) or "-",
                ]
            )
        print_table(
            ["strategy", "won", "lost", "both right", "both wrong", "not comparable",
             "right, baseline errored", "cost ratio",
             "won cases", "lost cases"],
            comparison_rows,
        )

    self_graded_total = sum(summary["self_graded"] for summary in summaries)
    if self_graded_total:
        details = ", ".join(
            f"{summary['strategy']} {summary['self_graded']}" for summary in summaries if summary["self_graded"]
        )
        print(f"\nSelf-grading: {self_graded_total} answers were judged by the model that wrote them ({details}).")
    adjudicated_total = sum(summary["adjudicated"] for summary in summaries)
    if adjudicated_total:
        details = ", ".join(
            f"{summary['strategy']} {summary['adjudicated']}" for summary in summaries if summary["adjudicated"]
        )
        print(
            f"Adjudicated: {adjudicated_total} exact or numeric answers held the expected value but not in "
            f"final position and were decided by the judge ({details})."
        )
    graded_counts = [summary["graded"] for summary in summaries if summary["graded"]]
    sample_size = max(graded_counts) if graded_counts else case_count
    if sample_size:
        print("\n" + SMALL_SAMPLE_CAVEAT.format(cases=sample_size, step=100 / sample_size))


def _fraction(cell: dict[str, Any]) -> str:
    if cell["graded"] == 0:
        return "n/a"
    return f"{cell['pass']}/{cell['graded']} {_percent(cell['rate'])}"


def print_openbook_scorecard(scorecard: dict[str, Any]) -> None:
    summaries = scorecard["strategies"]
    if not summaries:
        print("\nNo results to score.")
        return

    print(
        "\nOpen-book scorecard. Each column is its own measure with its own denominator "
        "(pass/graded); ungraded, error and fallback runs are left out of every rate.\n"
        "  answer acc    correct answers on answerable cases\n"
        "  abstain acc   unanswerable cases where the run said the sources do not contain the answer\n"
        "  false abst    answerable cases the run declined to answer\n"
        "  halluc        unanswerable cases the run answered anyway\n"
        "  cite valid    runs whose every citation quote is in the cited source (n/a: strategy does not cite)\n"
        "  evid hit      answerable runs with a citation on the case's evidence quote\n"
        "  esc tried     runs where the pipeline called a stronger tier\n"
        "  esc ok        of those, runs whose stronger-tier answer was used\n"
        "  esc failed    of those, runs whose stronger-tier call gave nothing usable (the draft was kept)\n"
    )
    table_rows = []
    for summary in summaries:
        interval = summary["answer_accuracy_ci95"]
        interval_text = f"{interval[0] * 100:.0f}-{interval[1] * 100:.0f}%" if interval else "n/a"
        table_rows.append(
            [
                summary["strategy"],
                summary["cases"],
                _fraction(summary["answer"]),
                interval_text,
                _fraction(summary["abstention"]),
                summary["false_abstentions"],
                summary["hallucinated_answers"],
                _fraction(summary["citation_validity"]),
                _fraction(summary["evidence_hit"]),
                summary["escalations_attempted"],
                summary["escalations_succeeded"],
                summary["escalations_failed"],
                summary["ungraded"],
                summary["error"] + summary["fallback_excluded"],
                f"{summary['total_cost_usd']:.4f}",
                _milliseconds(summary["latency_p50_ms"]),
                _milliseconds(summary["latency_p95_ms"]),
            ]
        )
    print_table(
        ["strategy", "cases", "answer acc", "95% CI", "abstain acc", "false abst", "halluc", "cite valid",
         "evid hit", "esc tried", "esc ok", "esc failed", "ungraded", "err/fell back", "cost usd", "p50 ms",
         "p95 ms"],
        table_rows,
    )
    print_escalation_costs(summaries)

    categories: list[str] = []
    for summary in summaries:
        for category in summary["by_category"]:
            if category not in categories:
                categories.append(category)
    print("\nBy category (pass/graded; unanswerable counts correct abstentions)\n")
    category_rows = []
    for summary in summaries:
        line = [summary["strategy"]]
        for category in categories:
            cell = summary["by_category"].get(category)
            line.append(f"{cell['pass']}/{cell['graded']}" if cell else "-")
        category_rows.append(line)
    print_table(["strategy", *categories], category_rows)

    baseline = scorecard["baseline"]
    if not scorecard["baseline_present"]:
        print(f"\nHead-to-head skipped: baseline {baseline} has no results (run it or --merge a file that has it).")
    elif scorecard["head_to_head"]:
        print(
            f"\nHead-to-head against {baseline} (a case is right when the answer is correct, or, for an "
            "unanswerable case, when the run abstained; cases where both have a pass or fail)\n"
        )
        comparison_rows = []
        for comparison in scorecard["head_to_head"]:
            if comparison["cost_ratio"] is None:
                cost_ratio = "n/a (baseline cost 0)"
            else:
                cost_ratio = f"{comparison['cost_ratio']:.2f}x"
            comparison_rows.append(
                [
                    comparison["strategy"],
                    comparison["won"],
                    comparison["lost"],
                    comparison["both_right"],
                    comparison["both_wrong"],
                    comparison["not_comparable"],
                    comparison.get("right_where_baseline_errored", 0),
                    cost_ratio,
                    ",".join(comparison["won_cases"]) or "-",
                    ",".join(comparison["lost_cases"]) or "-",
                ]
            )
        print_table(
            ["strategy", "won", "lost", "both right", "both wrong", "not comparable",
             "right, baseline errored", "cost ratio",
             "won cases", "lost cases"],
            comparison_rows,
        )

    runner_stuffed = [summary["strategy"] for summary in summaries if STUFFED_BY_RUNNER in summary["stuffed_by"]]
    if runner_stuffed:
        print(
            f"\nStuffing: {', '.join(runner_stuffed)} pasted the sources in this script (route mode, forced "
            "tier) because the harness has no stuffed mode yet."
        )
    self_graded_total = sum(summary["self_graded"] for summary in summaries)
    if self_graded_total:
        details = ", ".join(
            f"{summary['strategy']} {summary['self_graded']}" for summary in summaries if summary["self_graded"]
        )
        print(f"\nSelf-grading: {self_graded_total} answers were judged by the model that wrote them ({details}).")
    answerable_counts = [summary["answer"]["graded"] for summary in summaries if summary["answer"]["graded"]]
    if answerable_counts:
        sample_size = max(answerable_counts)
        print("\n" + SMALL_SAMPLE_CAVEAT.format(cases=sample_size, step=100 / sample_size))
    unanswerable_counts = [summary["abstention"]["graded"] for summary in summaries if summary["abstention"]["graded"]]
    if unanswerable_counts:
        print(
            f"Abstention accuracy rests on {max(unanswerable_counts)} unanswerable cases: one case moves it by "
            f"{100 / max(unanswerable_counts):.0f} points."
        )


def print_routes(routes: dict[str, dict[str, Any]], routes_path: Path, written: bool = True) -> None:
    summary = summarise_routes(routes)
    if written:
        print(f"\nMeasured routes ({routes_path})\n")
    else:
        print(f"\nRoutes from these results alone (NOT written; {routes_path} is unchanged)\n")
    print_table(
        ["case", "hand-written", "measured", "status", "tier outcomes"],
        [
            [
                case_id,
                entry["expected_route"],
                "/".join(entry["acceptable"]) or "-",
                entry["status"],
                " ".join(f"{tier_name}={outcome}" for tier_name, outcome in entry["tier_outcomes"].items()),
            ]
            for case_id, entry in routes.items()
        ],
    )
    status_text = ", ".join(f"{status} {len(case_ids)}" for status, case_ids in summary["by_status"].items())
    route_text = ", ".join(f"{route} {count}" for route, count in summary["by_route"].items())
    print(f"\nstatus: {status_text}")
    print(f"cheapest correct tier: {route_text or 'none'}")
    print(
        f"hand-written expected_route agrees with the measured route on "
        f"{summary['hand_written_route_agrees']} of {summary['resolved']} resolved cases"
    )


def _progress_line(record: dict[str, Any]) -> None:
    latency = f"{record['latency_ms']:.0f}ms" if record["latency_ms"] is not None else "-"
    note = record["grade_reason"] or record["error_type"] or record["measured_tier_error"] or ""
    print(
        f"  {record['strategy']:<14} {record['case_id']:<6} {record['outcome'] or '-':<17} "
        f"{record['final_tier'] or '-':<9} {latency:>8}  {note[:90]}",
        file=sys.stderr,
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run answer strategies over an eval set and score them")
    parser.add_argument("--suite", choices=[SUITE_CLOSEDBOOK, SUITE_OPENBOOK], default=SUITE_CLOSEDBOOK,
                        help="closedbook: the golden set (default). openbook: questions over supplied sources")
    parser.add_argument("--strategies", nargs="+",
                        help="closedbook: tier:<name>, router, chain, execute:<tier>, crosscheck:<tier A>+<tier B>, auto, auto:check (default: tier:local tier:mid router). "
                        "openbook: openbook, stuffed:<tier>, closed:<tier> (default: closed:mid stuffed:mid openbook)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--golden", help="question file (default: eval/golden_questions.yaml, or "
                        "eval/openbook/questions.yaml for the openbook suite)")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS_MANIFEST_PATH),
                        help="openbook suite: corpus manifest the source files are checked against")
    parser.add_argument("--cases", nargs="+", help="only these case ids")
    parser.add_argument("--category", nargs="+", help="only these categories")
    parser.add_argument("--triage-tier", help="triage tier for the router and chain strategies")
    parser.add_argument("--verifier-tier", help="verifier tier for the chain strategy, or the grounding "
                        "verifier tier for the openbook strategy")
    parser.add_argument("--judge-tier", default=DEFAULT_JUDGE_TIER, help="tier that judges rubric cases")
    parser.add_argument("--max-cost-usd", type=float, help="stop once cumulative reported cost passes this")
    parser.add_argument("--out", help="result JSON path (default: data/eval/eval_<utc timestamp>.json)")
    parser.add_argument("--db", default=str(DEFAULT_TELEMETRY_DB), help="telemetry database for eval runs")
    parser.add_argument("--merge", nargs="+", default=[],
                        help="earlier result JSON files to include in the scorecard and measured routes")
    parser.add_argument("--regrade", help="grade the stored answers of this result file again with the "
                        "current grader instead of running any strategy")
    parser.add_argument("--routes-out", default=str(DEFAULT_ROUTES_PATH))
    parser.add_argument("--allow-partial-routes", action="store_true",
                        help="write the measured routes even when the results lack some tiers or cases")
    arguments = parser.parse_args(argv)

    openbook_suite = arguments.suite == SUITE_OPENBOOK
    strategy_texts = arguments.strategies or SUITE_DEFAULT_STRATEGIES[arguments.suite]
    try:
        strategies = [parse_strategy(text) for text in strategy_texts]
    except ValueError as strategy_error:
        print(f"error: {strategy_error}", file=sys.stderr)
        return 2
    wrong_suite = [strategy.name for strategy in strategies if strategy.kind not in SUITE_KINDS[arguments.suite]]
    if wrong_suite:
        print(
            f"error: {wrong_suite} do not belong to the {arguments.suite} suite "
            f"(its strategies: {', '.join(SUITE_KINDS[arguments.suite])})",
            file=sys.stderr,
        )
        return 2

    load_env_file(".env")
    config = load_config(arguments.config)
    unknown_tiers = [
        tier
        for strategy in strategies
        if strategy.kind in FORCED_TIER_KINDS
        for tier in (strategy.tier, strategy.cross_check_tier)
        if tier is not None and tier not in config.answer_tier_names()
    ]
    if unknown_tiers:
        print(f"error: not answer tiers: {unknown_tiers}; choose from {config.answer_tier_names()}", file=sys.stderr)
        return 2
    if arguments.judge_tier not in config.answer_tier_names():
        print(f"error: judge tier '{arguments.judge_tier}' is not an answer tier", file=sys.stderr)
        return 2

    corpus: dict[str, dict[str, Any]] = {}
    if openbook_suite:
        try:
            all_cases, golden_info, corpus = load_openbook(
                arguments.golden or DEFAULT_OPENBOOK_QUESTIONS_PATH, arguments.corpus
            )
        except (fetch_corpus.CorpusError, QuestionSetError) as corpus_error:
            print(f"error: {corpus_error}", file=sys.stderr)
            return 2
    else:
        all_cases, golden_info = load_golden(arguments.golden or DEFAULT_GOLDEN_PATH)
    cases = filter_cases(all_cases, arguments.cases, arguments.category)
    if not cases:
        print("error: no cases match the filters", file=sys.stderr)
        return 2

    judge_tier_config = config.tiers[arguments.judge_tier]
    judge = LlmRubricJudge(
        build_clients(config)[judge_tier_config.provider], judge_tier_config, arguments.judge_tier
    )

    db_path = Path(arguments.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    harness_cache: dict[str | None, Any] = {}

    def harness_factory(strategy: Strategy) -> Any:
        triage_tier = arguments.triage_tier if strategy.kind not in FORCED_TIER_KINDS else None
        if triage_tier not in harness_cache:
            harness_cache[triage_tier] = build_harness(
                config_path=arguments.config, db_path=db_path, triage_tier=triage_tier
            )
        return harness_cache[triage_tier]

    started_utc = datetime.now(timezone.utc)
    out_path = Path(arguments.out) if arguments.out else (
        DEFAULT_EVAL_DIR
        / f"{'openbook' if openbook_suite else 'eval'}_{started_utc.strftime('%Y%m%dT%H%M%SZ')}.json"
    )

    def openbook_case_runner(harness: Any, strategy: Strategy, case: dict[str, Any], case_judge: Any) -> dict:
        return run_openbook_case(harness, strategy, case, case_judge, corpus, arguments.verifier_tier)

    run_started_at = time.monotonic()
    if arguments.regrade:
        prior = json.loads(Path(arguments.regrade).read_text(encoding="utf-8"))
        if not same_question_set(prior.get("golden", {}), golden_info):
            print(
                f"error: {arguments.regrade} was graded against a different golden set or corpus",
                file=sys.stderr,
            )
            return 2
        print(f"regrading {arguments.regrade}; judge {arguments.judge_tier} ({judge.model})", file=sys.stderr)
        if openbook_suite:
            regraded = regrade_openbook_records(
                prior["records"], all_cases, corpus, judge, progress=_progress_line
            )
        else:
            regraded = regrade_records(prior["records"], all_cases, judge, progress=_progress_line)
        strategies = [parse_strategy(name) for name in dict.fromkeys(row["strategy"] for row in regraded)]
        run = {
            "records": regraded,
            "not_runnable": {},
            "cumulative_cost_usd": round(sum(row["judge_cost_usd"] for row in regraded), 8),
            "stopped_early": False,
            "stop_reason": None,
        }
    else:
        print(
            f"running {[strategy.name for strategy in strategies]} over {len(cases)} cases; "
            f"judge {arguments.judge_tier} ({judge.model}); telemetry {db_path}",
            file=sys.stderr,
        )
        run = run_strategies(
            cases,
            strategies,
            harness_factory,
            judge,
            max_cost_usd=arguments.max_cost_usd,
            verifier_tier=arguments.verifier_tier,
            progress=_progress_line,
            case_runner=openbook_case_runner if openbook_suite else None,
        )

    all_records, merged_files, merge_warnings = merge_prior_records(run["records"], arguments.merge, golden_info)
    backfill_escalation_costs(all_records, db_path)
    routes: dict[str, dict[str, Any]] = {}
    route_gaps: list[str] = []
    routes_path = Path(arguments.routes_out)
    if openbook_suite:
        scorecard = build_openbook_scorecard(all_records)
        has_tier_results = False
    else:
        scorecard = build_scorecard(all_records)
        routes = derive_measured_routes(all_cases, all_records)
        route_gaps = route_coverage_gaps(all_cases, all_records)
        has_tier_results = any(record["strategy"].startswith("tier:") for record in all_records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "started_utc": started_utc.isoformat(timespec="seconds"),
                "elapsed_s": round(time.monotonic() - run_started_at, 1),
                "suite": arguments.suite,
                "golden": golden_info,
                "config_sha256": config.sha256,
                "strategies": [strategy.name for strategy in strategies],
                "case_ids": [case["id"] for case in cases],
                "judge_tier": arguments.judge_tier,
                "judge_model": judge.model,
                "triage_tier": arguments.triage_tier or config.triage_tier,
                "verifier_tier": arguments.verifier_tier,
                "max_cost_usd": arguments.max_cost_usd,
                "cumulative_cost_usd": run["cumulative_cost_usd"],
                "regraded_from": arguments.regrade,
                "stopped_early": run["stopped_early"],
                "stop_reason": run["stop_reason"],
                "not_runnable": run["not_runnable"],
                "merged_files": merged_files,
                "scorecard": scorecard,
                "records": all_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for warning in merge_warnings:
        print(f"warning: {warning}")
    for strategy_name, reason in run["not_runnable"].items():
        print(f"\nNOT RUN: {strategy_name}: {reason}")
    if run["stopped_early"]:
        print(f"\nSTOPPED EARLY: {run['stop_reason']}. The scorecard below covers only the runs completed.")
    if merged_files:
        print(f"\nIncludes earlier results from: {', '.join(merged_files)}")

    if openbook_suite:
        print_openbook_scorecard(scorecard)
    else:
        print_scorecard(scorecard, len(cases))
    if has_tier_results and (not route_gaps or arguments.allow_partial_routes):
        write_measured_routes(routes_path, routes, golden_info, [str(out_path), *merged_files])
        print_routes(routes, routes_path)
    elif has_tier_results:
        print_routes(routes, routes_path, written=False)
        print(
            f"\nMeasured routes not written: {'; '.join(route_gaps)}. Add the missing results with --merge, "
            "or pass --allow-partial-routes to write them anyway."
        )
    print(
        f"\nspent {run['cumulative_cost_usd']:.4f} USD (answers plus judge), "
        f"elapsed {time.monotonic() - run_started_at:.0f}s"
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
