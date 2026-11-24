"""Phase 3 execute mode: the model writes a program, a sandbox runs it, code compares.

Plain control flow with a fixed ceiling of two model calls and two sandbox runs (four and
four with the cross-check, which runs this course once per program):

1. Draft. The answering tier returns an `ExecuteDraft`: a short Python program (standard
   library only) that prints the answer as its last line, and optionally the answer it
   claims. The prompt tells the model to leave the claim null rather than work a
   calculation out by hand: measured on the hard executable cases, a required claim made
   the model spend about 95% of the call's output re-deriving the answer in its reasoning,
   and three of fifteen calls hit the 120 s timeout before any program existed.
   A failed call or invalid output fails the run: there is nothing to return.
2. Run the program through the dispatch gate (sandbox.py).
3. Decide, in `decide`, a pure function. The model is never asked whether the output
   and the claim agree: `answers_agree` compares them with the normalisation stated there.
4. At most one repair, only when the program itself failed or timed out: the same tier
   gets the program and its stderr, and its new program is run once. The repair call
   never takes the `on_error` hop.

What the statuses mean. "claim_matched" says the program ran to completion in the sandbox
and printed what the model claimed. It does not say the program is a correct model of the
question: a program that prints a constant matches a constant. Both sides come from one
model call, so this is not called verified. "Executed" says less: the program ran to
completion and there was no claim to compare.

"verified" is reserved for the cross-check (`run_cross_check`, off by default): two
programs written independently from the question alone, each run in its own sandbox,
printed values that `values_match` finds equal. Program B's prompt is built from the
question only; nothing of program A (its code, output, explanation or error) is an
input to any function that builds a message for B, and the reverse holds too.

The explanation. The program prints its reasoning before the answer line, built from the
values it computed, and that text goes into the answer as printed. No model call writes
or rewrites it, so the executed output is the source of every value the user sees.

This module does not record telemetry. It returns every model call as a step and every
sandbox run as an attempt; the router turns those into rows.
"""

from __future__ import annotations

import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Literal

from pydantic import ValidationError

from inference_arbiter import sandbox
from inference_arbiter.chain import ChainStep, TierCaller
from inference_arbiter.schemas import (
    STATUS_ERROR,
    STATUS_OK,
    CallResult,
    ExecuteDraft,
    ExecutionStatus,
)

ROLE_DRAFT = "draft"
ROLE_REPAIR = "repair"

ERROR_DRAFT_SCHEMA_INVALID = "draft_schema_invalid"
# No program ran cleanly and the model gave no claim: there is nothing to return.
ERROR_NO_ANSWER = "execute_no_answer"

MAX_VALIDATION_MESSAGE_CHARS = 300
# How much of a failed program's output the repair call is shown.
REPAIR_STDERR_CHARS = 2000
REPAIR_STDOUT_CHARS = 500

AttemptClass = Literal["agree", "disagree", "no_claim", "program_failed", "infra_failed"]
RepairState = Literal[
    "not_attempted", "agree", "disagree", "no_claim", "program_failed", "infra_failed", "draft_failed"
]
AnswerFrom = Literal["executed_first", "executed_repair", "claim_first", "claim_repair"]

REASON_NO_OUTPUT = "no_output"

# How much of the program's printed explanation goes into the answer text.
EXPLANATION_MAX_CHARS = 4000
EXPLANATION_TRUNCATED_MARK = "[explanation truncated]"
EXPLANATION_HEADING = "Explanation printed by the program:"
# The last line of a disputed answer. It is deliberately not an "Answer:" line.
DISPUTED_LAST_LINE = "No answer is confirmed: the two programs disagree."

EXECUTE_SYSTEM_PROMPT = """Answer the user's question by writing a short Python 3 program that computes the answer.

Rules for the program:
- Standard library only. No third-party imports.
- It runs in an isolated sandbox with no network, no input and no files to read. Do not read input, open files or connect to anything.
- It must finish in a few seconds.
- It must compute the answer, not print a value you worked out yourself.
- Do not work the answer out yourself at all, in your reasoning or anywhere else. Decide the method, write the program, stop.
- Before the final answer, it must print the explanation the question needs, as a few short plain sentences: what was checked, the intermediate values that matter and why the result follows from them. Build every number and every conclusion in those sentences from values the program computed (f-strings), and where the outcome depends on a check, print the reason for the branch that was taken. Never write a value in a sentence that you worked out yourself. For a question that only asks for a value, one or two lines of key intermediate values are enough.
- Its last line of output must be the final answer and nothing else: no units unless the question asks for them, no explanation. The lines before it are shown to the user as the explanation, so they must make sense without the program.

Return JSON with these fields, in this order:
- working: how you will compute it, in a sentence or two
- program: the complete Python source
- claimed_answer: null, unless you already know the exact answer without working anything out (a fact you recall, or arithmetic you can do in one step), in which case give it as a string. Never compute, trace or simulate the program by hand to fill this field, and do not check the program by working through it: the program is run and its output is the answer. null is the expected value for any real calculation"""

REPAIR_SYSTEM_PROMPT = """A Python program written to answer the user's question failed when it was run. You are given the question, the program and what happened. Write a corrected program.

Rules for the program:
- Standard library only. No third-party imports.
- It runs in an isolated sandbox with no network, no input and no files to read. Do not read input, open files or connect to anything.
- It must finish in a few seconds.
- It must compute the answer, not print a value you worked out yourself.
- Do not work the answer out yourself at all, in your reasoning or anywhere else. Decide the method, write the program, stop.
- Before the final answer, it must print the explanation the question needs, as a few short plain sentences: what was checked, the intermediate values that matter and why the result follows from them. Build every number and every conclusion in those sentences from values the program computed (f-strings), and where the outcome depends on a check, print the reason for the branch that was taken. Never write a value in a sentence that you worked out yourself. For a question that only asks for a value, one or two lines of key intermediate values are enough.
- Its last line of output must be the final answer and nothing else: no units unless the question asks for them, no explanation. The lines before it are shown to the user as the explanation, so they must make sense without the program.

Return JSON with these fields, in this order:
- working: what was wrong and how the new program computes the answer
- program: the complete corrected Python source
- claimed_answer: null, unless you already know the exact answer without working anything out (a fact you recall, or arithmetic you can do in one step), in which case give it as a string. Never compute, trace or simulate the program by hand to fill this field, and do not check the program by working through it: the program is run and its output is the answer. null is the expected value for any real calculation"""


# --- agreement between the claim and the executed output, decided in code ---

_ANSWER_LABEL = re.compile(r"^\W*(?:final answer|answer|output|result)\W*[:=]\s*", re.IGNORECASE)
_OWN_ANSWER_LABEL = re.compile(r"^(?:final answer|answer)\s*[:=]\s*", re.IGNORECASE)
_WRAPPING_CHARS = "\"'`* \t"
_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+(?:[eE][-+]?\d+)?")
# A string that is one number and nothing else, apart from a currency or percent sign.
_BARE_NUMBER_EXTRAS = re.compile(r"[\s$\u20ac\u00a3%]")
NUMERIC_RELATIVE_TOLERANCE = Decimal("1e-9")
NUMERIC_ABSOLUTE_TOLERANCE = Decimal("1e-12")


def executed_answer(stdout: str) -> str | None:
    """The program's answer: the last non-empty line of stdout, stripped. None when there is none."""
    for line in reversed(stdout.splitlines()):
        if line.strip():
            return line.strip()
    return None


def normalise_answer(text: str) -> str:
    """NFKC, drop an 'Answer:' style label, strip quotes, backticks and asterisks, collapse
    whitespace, casefold, drop one trailing full stop."""
    normalised = unicodedata.normalize("NFKC", text).strip()
    normalised = _ANSWER_LABEL.sub("", normalised)
    normalised = normalised.strip(_WRAPPING_CHARS)
    normalised = " ".join(normalised.split()).casefold()
    if normalised.endswith(".") and not normalised[:-1].endswith("."):
        normalised = normalised[:-1].rstrip()
    return normalised


def _numbers(text: str) -> list[Decimal]:
    found: list[Decimal] = []
    for token in _NUMBER.findall(text):
        try:
            found.append(Decimal(token.replace(",", "")))
        except InvalidOperation:
            continue
    return found


def _is_bare_number(text: str) -> bool:
    remainder = _BARE_NUMBER_EXTRAS.sub("", text)
    return bool(remainder) and _NUMBER.fullmatch(remainder) is not None


def answers_agree(claimed: str, executed: str) -> bool:
    """Does the executed output agree with the claimed answer? Decided here, never by a model.

    Both sides go through `normalise_answer`. They agree when:

    1. the normalised strings are equal; or
    2. each side holds exactly one number, at least one side is nothing but that number
       (a currency or percent sign is allowed), and the two numbers differ by at most
       1e-9 relative (1e-12 absolute). Thousands separators are ignored. So a claim of
       "24 legs" agrees with an output of "24", and "0.3" with "0.30000000000000004".

    Anything else disagrees, including two sentences that differ only in wording.
    """
    claimed_normalised = normalise_answer(claimed)
    executed_normalised = normalise_answer(executed)
    if claimed_normalised == executed_normalised:
        return True
    claimed_numbers = _numbers(claimed_normalised)
    executed_numbers = _numbers(executed_normalised)
    if len(claimed_numbers) != 1 or len(executed_numbers) != 1:
        return False
    if not (_is_bare_number(claimed_normalised) or _is_bare_number(executed_normalised)):
        return False
    difference = abs(claimed_numbers[0] - executed_numbers[0])
    allowed = max(NUMERIC_ABSOLUTE_TOLERANCE, NUMERIC_RELATIVE_TOLERANCE * abs(executed_numbers[0]))
    return difference <= allowed


def classify_attempt(result: sandbox.SandboxResult, claimed: str | None) -> tuple[AttemptClass, str | None]:
    """Turn one sandbox result into an attempt class. Returns (class, failure reason or None).

    | sandbox result                                   | class          |
    |--------------------------------------------------|----------------|
    | ok, but nothing on stdout                        | program_failed |
    | ok, the model gave no claim                      | no_claim       |
    | ok, output agrees with the claim                 | agree          |
    | ok, output disagrees with the claim              | disagree       |
    | timeout, nonzero_exit, output_too_large          | program_failed |
    | any other error (sandbox or gate, not the program)| infra_failed  |
    """
    if result.ok:
        printed = executed_answer(result.stdout)
        if printed is None:
            return "program_failed", REASON_NO_OUTPUT
        if claimed is None:
            return "no_claim", None
        return ("agree" if answers_agree(claimed, printed) else "disagree"), None
    if result.error_type in sandbox.PROGRAM_ERRORS:
        return "program_failed", result.error_type
    return "infra_failed", result.error_type


# --- the decision table ---


@dataclass(frozen=True)
class ExecuteDecision:
    action: Literal["return", "repair"]
    # Set when action is "return".
    status: ExecutionStatus | None = None
    answer_from: AnswerFrom | None = None


def _claim_or_failed(*candidates: tuple[AnswerFrom, bool]) -> ExecuteDecision:
    """Unverified with the first claim that exists, in the order given; failed when none does."""
    for answer_from, has_claim in candidates:
        if has_claim:
            return ExecuteDecision("return", "unverified", answer_from)
    return ExecuteDecision("return", "failed", None)


def decide(
    first: AttemptClass,
    repair: RepairState,
    repairs_allowed: bool,
    first_has_claim: bool = True,
    repair_has_claim: bool = True,
) -> ExecuteDecision:
    """Decision table. First match wins.

    | #  | first run      | repair                       | result                              |
    |----|----------------|------------------------------|-------------------------------------|
    | 1  | agree          | -                            | claim_matched, the executed output  |
    | 2  | disagree       | -                            | corrected, the executed output wins |
    | 3  | no_claim       | -                            | executed, the executed output       |
    | 4  | infra_failed   | -                            | unverified, the first claim (*)     |
    | 5  | program_failed | not attempted, allowed       | attempt the one repair              |
    | 6  | program_failed | not attempted, not allowed   | unverified, the first claim (*)     |
    | 7  | program_failed | agree                        | claim_matched, the repair's output  |
    | 8  | program_failed | disagree                     | corrected, the repair's output wins |
    | 9  | program_failed | no_claim                     | executed, the repair's output       |
    | 10 | program_failed | program_failed, infra_failed | unverified, the repair's claim,     |
    |    |                |                              | else the first claim (*)            |
    | 11 | program_failed | draft_failed                 | unverified, the first claim (*)     |

    (*) The model may decline to claim an answer (`first_has_claim`, `repair_has_claim`).
    An unverified result needs a claim to return. Where the row has none, the result is
    `failed`: no answer, and the run is an error. Nothing is made up.

    Rows 3 and 9: the program ran cleanly and there is no claim to compare it with. That
    is `executed`, never `claim_matched`: nothing agreed with the output. No row of this
    table returns `verified`: that needs two programs (`decide_cross_check`).
    Row 4: the sandbox or the gate failed, the program was not at fault, so a rewritten
    program would not help and no repair is attempted. Row 11: the repair call failed or
    returned invalid output, so the only claim there can be is the first one.
    """
    if first == "agree":
        return ExecuteDecision("return", "claim_matched", "executed_first")
    if first == "disagree":
        return ExecuteDecision("return", "corrected", "executed_first")
    if first == "no_claim":
        return ExecuteDecision("return", "executed", "executed_first")
    if first == "infra_failed":
        return _claim_or_failed(("claim_first", first_has_claim))
    if repair == "not_attempted":
        if repairs_allowed:
            return ExecuteDecision("repair")
        return _claim_or_failed(("claim_first", first_has_claim))
    if repair == "agree":
        return ExecuteDecision("return", "claim_matched", "executed_repair")
    if repair == "disagree":
        return ExecuteDecision("return", "corrected", "executed_repair")
    if repair == "no_claim":
        return ExecuteDecision("return", "executed", "executed_repair")
    if repair in ("program_failed", "infra_failed"):
        return _claim_or_failed(("claim_repair", repair_has_claim), ("claim_first", first_has_claim))
    return _claim_or_failed(("claim_first", first_has_claim))


# --- control flow ---


@dataclass
class SandboxAttempt:
    """One program run: what was run, what came back and how it was classed."""

    attempt: Literal["first", "repair"]
    program: str
    claimed_answer: str | None
    result: sandbox.SandboxResult
    # An AttemptClass, or in a cross-check a run state: ran | program_failed | infra_failed.
    outcome: str
    failure_reason: str | None
    executed_answer: str | None
    # Which of the two programs of a cross-check this is; "a" outside a cross-check.
    program_label: str = "a"


@dataclass
class ExecuteOutcome:
    tier: str
    status: str = STATUS_ERROR
    answer: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    final_effort: str | None = None
    execution_status: ExecutionStatus = "skipped"
    claimed_answer: str | None = None
    executed_answer: str | None = None
    execution_note: str | None = None
    # What the returned program printed before its answer line (None for a claim).
    executed_explanation: str | None = None
    repaired: bool = False
    repair_error_type: str | None = None
    # Cross-check only. `executed_answer` is program A's output, `executed_answer_b` program
    # B's (None for a program that did not run cleanly); `cross_check_rule` is how the two
    # were compared (integer | hex | decimal | number_list | string), None when not both ran.
    cross_check: bool = False
    cross_check_tier: str | None = None
    executed_answer_b: str | None = None
    cross_check_rule: str | None = None
    steps: list[ChainStep] = field(default_factory=list)
    attempts: list[SandboxAttempt] = field(default_factory=list)


def draft_json_schema() -> dict[str, Any]:
    schema = ExecuteDraft.model_json_schema()
    # claimed_answer stays required so the field is always present; its value may be null.
    schema["required"] = ["working", "program", "claimed_answer"]
    return schema


def parse_draft(call_result: CallResult) -> tuple[ExecuteDraft | None, str | None]:
    """The schema gate. Returns (draft, None) or (None, why it was rejected)."""
    try:
        return ExecuteDraft.model_validate_json(call_result.text), None
    except ValidationError as validation_error:
        return None, " ".join(str(validation_error).split())[:MAX_VALIDATION_MESSAGE_CHARS]


def build_draft_messages(conversation: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"role": "system", "content": EXECUTE_SYSTEM_PROMPT}] + conversation


def describe_failure(attempt: SandboxAttempt) -> str:
    """What the repair call is told about the failed run."""
    result = attempt.result
    if attempt.failure_reason == REASON_NO_OUTPUT:
        happened = "The program exited with code 0 but printed nothing to stdout."
    elif attempt.failure_reason == sandbox.ERROR_TIMEOUT:
        happened = "The program was stopped: it did not finish within the time limit."
    elif attempt.failure_reason == sandbox.ERROR_OUTPUT_TOO_LARGE:
        happened = "The program was stopped: it printed far too much output."
    else:
        happened = f"The program exited with code {result.exit_code}."
    return (
        f"{happened}\n\nstderr (last {REPAIR_STDERR_CHARS} characters):\n"
        f"{result.stderr[-REPAIR_STDERR_CHARS:] or '(empty)'}\n\n"
        f"stdout (first {REPAIR_STDOUT_CHARS} characters):\n{result.stdout[:REPAIR_STDOUT_CHARS] or '(empty)'}"
    )


def build_repair_messages(
    question: str, attempt: SandboxAttempt, system_prompt: str = REPAIR_SYSTEM_PROMPT
) -> list[dict[str, str]]:
    user_content = (
        f"Question:\n{question}\n\nProgram that failed:\n{attempt.program}\n\n"
        f"What happened when it ran:\n{describe_failure(attempt)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _why_unverified(attempts: list[SandboxAttempt], repair_error_type: str | None) -> str:
    parts = []
    for attempt in attempts:
        label = "the program" if attempt.attempt == "first" else "the repaired program"
        if attempt.failure_reason is None:
            continue
        if attempt.outcome == "infra_failed":
            parts.append(f"{label} could not be run in the sandbox ({attempt.failure_reason})")
        else:
            parts.append(f"{label} failed ({attempt.failure_reason})")
    if repair_error_type is not None:
        parts.append(f"the repair call failed ({repair_error_type})")
    return "; ".join(parts)


def executed_explanation(stdout: str) -> str | None:
    """What a clean run printed before its answer line, capped. None when there is nothing.

    The prompt asks the program to print its reasoning there, built from values it computed.
    The text is shown as printed: no model rewrites it, so every value in it was executed.
    """
    lines = stdout.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    explanation = "\n".join(lines[:-1]).strip()
    if not explanation:
        return None
    if len(explanation) > EXPLANATION_MAX_CHARS:
        explanation = explanation[:EXPLANATION_MAX_CHARS].rstrip() + "\n" + EXPLANATION_TRUNCATED_MARK
    return explanation


def strip_answer_label(answer: str) -> str:
    """Drop an 'Answer:' label the program printed itself, so the answer line carries one label."""
    stripped = _OWN_ANSWER_LABEL.sub("", answer.strip())
    return stripped if stripped else answer.strip()


def format_answer(answer: str, note: str, explanation: str | None = None) -> str:
    """The note first, then what the program printed as its explanation, then the answer on
    the last line: the answer keeps final position, with exactly one 'Answer:' label."""
    parts = [note]
    if explanation:
        parts.append(f"{EXPLANATION_HEADING}\n{explanation}")
    parts.append(f"Answer: {strip_answer_label(answer)}")
    return "\n\n".join(parts)


def run_execute(
    question: str,
    conversation: list[dict[str, str]],
    tier_name: str,
    max_repairs: int,
    effort_override: str | None,
    call_tier: TierCaller,
    run_in_sandbox: Callable[[str], sandbox.SandboxResult],
    simulated_answer_failure: Callable[[], CallResult] | None = None,
) -> ExecuteOutcome:
    outcome = ExecuteOutcome(tier=tier_name)

    def draft_call(role: str, messages: list[dict[str, str]]) -> ExecuteDraft | None:
        """One model call through the schema gate. Records the step; None when it failed."""
        if role == ROLE_DRAFT and simulated_answer_failure is not None:
            call_result, effort = simulated_answer_failure(), effort_override
        else:
            call_result, effort = call_tier(tier_name, messages, draft_json_schema(), effort_override)
        outcome.final_effort = effort
        if not call_result.ok:
            outcome.steps.append(ChainStep(role, tier_name, effort, call_result))
            outcome.error_type, outcome.error_message = call_result.error_type, call_result.error_message
            return None
        draft, rejection = parse_draft(call_result)
        outcome.steps.append(ChainStep(role, tier_name, effort, call_result, invalid=draft is None))
        if draft is None:
            outcome.error_type, outcome.error_message = ERROR_DRAFT_SCHEMA_INVALID, rejection
        return draft

    def run_attempt(label: Literal["first", "repair"], draft: ExecuteDraft) -> SandboxAttempt:
        result = run_in_sandbox(draft.program)
        attempt_class, failure_reason = classify_attempt(result, draft.claimed_answer)
        attempt = SandboxAttempt(
            attempt=label,
            program=draft.program,
            claimed_answer=draft.claimed_answer,
            result=result,
            outcome=attempt_class,
            failure_reason=failure_reason,
            executed_answer=executed_answer(result.stdout) if result.ok else None,
        )
        outcome.attempts.append(attempt)
        return attempt

    first_draft = draft_call(ROLE_DRAFT, build_draft_messages(conversation))
    if first_draft is None:
        return outcome  # status error: there is no claim to return
    first_attempt = run_attempt("first", first_draft)

    repair_state: RepairState = "not_attempted"
    repair_draft: ExecuteDraft | None = None
    repair_attempt: SandboxAttempt | None = None
    first_has_claim = first_draft.claimed_answer is not None
    decision = decide(first_attempt.outcome, repair_state, max_repairs > 0, first_has_claim)
    if decision.action == "repair":
        repair_draft = draft_call(ROLE_REPAIR, build_repair_messages(question, first_attempt))
        if repair_draft is None:
            repair_state = "draft_failed"
            outcome.repair_error_type = outcome.error_type
        else:
            repair_attempt = run_attempt("repair", repair_draft)
            repair_state = repair_attempt.outcome
        decision = decide(
            first_attempt.outcome,
            repair_state,
            repairs_allowed=False,
            first_has_claim=first_has_claim,
            repair_has_claim=repair_draft is not None and repair_draft.claimed_answer is not None,
        )
    # A failed repair call is recorded as repair_error_type, not as the run's error.
    outcome.error_type = outcome.error_message = None

    used_repair = decision.answer_from in ("executed_repair", "claim_repair")
    final_draft = repair_draft if used_repair else first_draft
    final_attempt = repair_attempt if used_repair else first_attempt
    outcome.repaired = decision.answer_from == "executed_repair"
    outcome.claimed_answer = final_draft.claimed_answer
    outcome.executed_answer = final_attempt.executed_answer
    outcome.execution_status = decision.status

    if decision.status == "failed":
        # No program ran cleanly and the model claimed nothing: fail closed, no answer.
        why = _why_unverified(outcome.attempts, outcome.repair_error_type)
        outcome.claimed_answer = None
        outcome.execution_note = f"FAILED: no answer. {why}, and the model gave no claimed answer to fall back on."
        outcome.error_type, outcome.error_message = ERROR_NO_ANSWER, outcome.execution_note
        return outcome

    if decision.status == "claim_matched":
        answer = final_attempt.executed_answer
        note = (
            "Matches the model's claim: a Python program computed this in an isolated sandbox and its "
            "output agrees with the answer the same model claimed. No second program checked it."
        )
        if outcome.repaired:
            note += " The first program failed; this is the repaired program."
    elif decision.status == "corrected":
        answer = final_attempt.executed_answer
        note = (
            f"Corrected by execution: the model claimed \"{final_draft.claimed_answer}\", but the "
            "program it wrote printed a different result when run in an isolated sandbox. The "
            "executed output is returned."
        )
        if outcome.repaired:
            note += " The first program failed; this is the repaired program."
    elif decision.status == "executed":
        answer = final_attempt.executed_answer
        note = (
            "Computed by execution, not cross-checked: a Python program computed this in an isolated "
            "sandbox. The model gave no answer of its own to compare with, so this is the program's "
            "output and nothing independent agreed with it."
        )
        if outcome.repaired:
            note += " The first program failed; this is the repaired program."
    else:
        answer = final_draft.claimed_answer
        note = (
            "UNVERIFIED: this answer was not confirmed by execution ("
            f"{_why_unverified(outcome.attempts, outcome.repair_error_type)}). It is the model's own claim."
        )
    outcome.execution_note = note
    if decision.status != "unverified":
        outcome.executed_explanation = executed_explanation(final_attempt.result.stdout)
    outcome.answer = format_answer(answer, note, outcome.executed_explanation)
    outcome.status = STATUS_OK
    return outcome


# --- cross-check: two programs written independently, compared in code ---

PROGRAM_A = "a"
PROGRAM_B = "b"
ROLE_DRAFT_B = "draft_b"
ROLE_REPAIR_B = "repair_b"
_TRACK_ROLES = {PROGRAM_A: (ROLE_DRAFT, ROLE_REPAIR), PROGRAM_B: (ROLE_DRAFT_B, ROLE_REPAIR_B)}

RunState = Literal["ran", "program_failed", "infra_failed"]
TrackState = Literal["ran", "program_failed", "infra_failed", "draft_failed"]
TrackRepairState = Literal["not_attempted", "ran", "program_failed", "infra_failed", "draft_failed"]

# Added to program B's system prompt when both programs are written by the same model: two
# drafts from one model share its habits, so B is asked for a different method. It says
# nothing about program A, because B's prompt never has anything of A to say.
INDEPENDENT_METHOD_RULE = """

This program is one of two written separately for the same question, and their answers will be compared. You are not shown the other program. Where the question can be solved in more than one sound way, do not take the most obvious route: pick a different method (for example enumerate or simulate instead of using a closed formula, or parse and process the data in a different order or representation), so that one mistake is not made in both programs. The answer and its format must still be exactly what the question asks for."""

# Added to both programs' draft and repair prompts in a cross-check. `values_match` compares
# the two last lines exactly, so two programs that reach one conclusion and word it
# differently are disputed. The fix is at the source: the last line is the bare value in a
# fixed form, and everything else goes in the explanation. The comparison is not loosened.
BARE_ANSWER_LINE_RULE = """

The last line is compared exactly, by code, with the last line of a second program written separately for the same question. It must therefore hold only the bare value the question asks for, in its shortest standard form. Every reason, qualification and supporting figure belongs in the explanation lines above it, never on the last line:
- If the question states a format for the answer, follow that format exactly.
- A number: the number alone, with no thousands separators, no currency sign, no unit and no words, unless the question's stated format requires them.
- A question that asks whether something is true, correct or safe, or asks you to confirm something: exactly Yes or exactly No, and nothing else on the line.
- A question that asks for a value that does not exist or must not be given (for example the input is invalid and must be rejected, or what the question takes for granted is false): exactly None, and nothing else on the line. The explanation says why.
- Anything else: the exact value alone (an identifier, a code, a name, a date as YYYY-MM-DD), with no sentence around it."""

# Tolerance of the decimal rule in `values_match`.
CROSS_CHECK_RELATIVE_TOLERANCE = Decimal("1e-9")

_CURRENCY_PREFIX = re.compile(r"^([-+]?)\s*[$\u20ac\u00a3]\s*")
_INTEGER_TEXT = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)")
_HEX_TEXT = re.compile(r"[-+]?0x[0-9a-f]+")
_DECIMAL_TEXT = re.compile(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?")
_LIST_SEPARATORS = re.compile(r"[\s\[\](){},;]+")


@dataclass(frozen=True)
class ValueComparison:
    match: bool
    # integer | hex | decimal | number_list | string: the rule that decided.
    rule: str


def _scalar(text: str) -> tuple[str, int | Decimal] | None:
    """A normalised answer as one number: ("integer" | "hex", int) or ("decimal", Decimal).
    None when it is anything else. A leading currency sign is dropped; a percent sign,
    a unit or any other text makes it a string."""
    bare = _CURRENCY_PREFIX.sub(r"\1", text)
    if _HEX_TEXT.fullmatch(bare):
        return "hex", int(bare, 16)
    if _INTEGER_TEXT.fullmatch(bare):
        return "integer", int(bare.replace(",", ""))
    if _DECIMAL_TEXT.fullmatch(bare):
        try:
            return "decimal", Decimal(bare.replace(",", ""))
        except InvalidOperation:
            return None
    return None


def _decimal_places(value: Decimal) -> int:
    exponent = value.as_tuple().exponent
    return -exponent if isinstance(exponent, int) and exponent < 0 else 0


def _scalars_match(left: tuple[str, int | Decimal], right: tuple[str, int | Decimal]) -> ValueComparison:
    left_kind, left_value = left
    right_kind, right_value = right
    if left_kind != "decimal" and right_kind != "decimal":
        rule = "hex" if "hex" in (left_kind, right_kind) else "integer"
        return ValueComparison(left_value == right_value, rule)
    left_decimal, right_decimal = Decimal(left_value), Decimal(right_value)
    allowed = CROSS_CHECK_RELATIVE_TOLERANCE * max(abs(left_decimal), abs(right_decimal))
    if left_kind == "decimal" and right_kind == "decimal":
        left_places, right_places = _decimal_places(left_decimal), _decimal_places(right_decimal)
        if left_places != right_places and min(left_places, right_places) >= 1:
            # One side was rounded to fewer places: half a unit in its last place.
            allowed = max(allowed, Decimal(5).scaleb(-(min(left_places, right_places) + 1)))
    return ValueComparison(abs(left_decimal - right_decimal) <= allowed, "decimal")


def _number_list(text: str) -> list[tuple[str, int | Decimal]] | None:
    """A normalised answer as two or more numbers separated only by brackets, commas,
    semicolons and whitespace. None for anything else."""
    tokens = [token for token in _LIST_SEPARATORS.split(text) if token]
    if len(tokens) < 2:
        return None
    scalars = [_scalar(token) for token in tokens]
    if any(scalar is None for scalar in scalars):
        return None
    return scalars  # type: ignore[return-value]


def values_match(answer_a: str, answer_b: str) -> ValueComparison:
    """Do two programs' answer lines hold the same value? Decided here, never by a model.

    Both lines go through `normalise_answer` (NFKC, an 'Answer:' label dropped, wrapping
    quotes and asterisks stripped, whitespace collapsed, casefolded, one trailing full
    stop dropped). Then the first rule that applies decides:

    | both sides are                         | rule        | match when                         |
    |----------------------------------------|-------------|------------------------------------|
    | integers (decimal digits or 0x hex)    | integer/hex | the integers are equal, exactly    |
    | numbers, at least one with a decimal   | decimal     | within the tolerance below         |
    |   point or an exponent                 |             |                                    |
    | lists of two or more numbers           | number_list | same length, each pair matches by  |
    |                                        |             | the two rules above                |
    | anything else                          | string      | the normalised strings are equal   |

    Decimal tolerance, stated: the two values may differ by 1e-9 relative to the larger
    one. When both carry decimal places and one carries fewer, that one is taken to be
    rounded and the difference may be up to half a unit in its last place (1580.17
    matches 1580.1687; 1580.17 does not match 1580.18, and 1580 does not match 1580.17).

    A leading currency sign and thousands separators are ignored in a number. A percent
    sign, a unit or any other text makes the side a string, so "5%" does not match "5",
    "24 legs" does not match "24", and two sentences that differ in wording do not match.
    """
    normalised_a, normalised_b = normalise_answer(answer_a), normalise_answer(answer_b)
    scalar_a, scalar_b = _scalar(normalised_a), _scalar(normalised_b)
    if scalar_a is not None and scalar_b is not None:
        return _scalars_match(scalar_a, scalar_b)
    list_a, list_b = _number_list(normalised_a), _number_list(normalised_b)
    if list_a is not None and list_b is not None:
        same = len(list_a) == len(list_b) and all(
            _scalars_match(left, right).match for left, right in zip(list_a, list_b)
        )
        return ValueComparison(same, "number_list")
    return ValueComparison(bool(normalised_a) and normalised_a == normalised_b, "string")


def classify_run(result: sandbox.SandboxResult) -> tuple[RunState, str | None]:
    """One sandbox result of a cross-check as a run state. Returns (state, failure reason).
    A claim plays no part: only what the program printed counts.

    | sandbox result                                    | state          |
    |---------------------------------------------------|----------------|
    | ok, with at least one non-empty line on stdout    | ran            |
    | ok, but nothing on stdout                         | program_failed |
    | timeout, nonzero_exit, output_too_large           | program_failed |
    | any other error (sandbox or gate, not the program)| infra_failed   |
    """
    if result.ok:
        if executed_answer(result.stdout) is None:
            return "program_failed", REASON_NO_OUTPUT
        return "ran", None
    if result.error_type in sandbox.PROGRAM_ERRORS:
        return "program_failed", result.error_type
    return "infra_failed", result.error_type


@dataclass(frozen=True)
class TrackDecision:
    action: Literal["return", "repair"]
    # Set when action is "return".
    state: TrackState | None = None
    use: Literal["first", "repair"] | None = None


def decide_track(first: RunState, repair: TrackRepairState, repairs_allowed: bool) -> TrackDecision:
    """One program's own course, the same for A and for B. First match wins.

    | # | first run      | repair                     | result                         |
    |---|----------------|----------------------------|--------------------------------|
    | 1 | ran            | -                          | ran, the first program         |
    | 2 | infra_failed   | -                          | infra_failed                   |
    | 3 | program_failed | not attempted, allowed     | attempt the one repair         |
    | 4 | program_failed | not attempted, not allowed | program_failed                 |
    | 5 | program_failed | ran                        | ran, the repaired program      |
    | 6 | program_failed | infra_failed               | infra_failed                   |
    | 7 | program_failed | program_failed             | program_failed                 |
    | 8 | program_failed | draft_failed               | program_failed                 |

    A repair is this program's own: its prompt holds the question, this program and what
    happened when this program ran. Nothing of the other program is available here.
    """
    if first == "ran":
        return TrackDecision("return", "ran", "first")
    if first == "infra_failed":
        return TrackDecision("return", "infra_failed", None)
    if repair == "not_attempted":
        return TrackDecision("repair") if repairs_allowed else TrackDecision("return", "program_failed", None)
    if repair == "ran":
        return TrackDecision("return", "ran", "repair")
    if repair == "infra_failed":
        return TrackDecision("return", "infra_failed", None)
    return TrackDecision("return", "program_failed", None)


@dataclass(frozen=True)
class CrossCheckDecision:
    status: ExecutionStatus
    # Whose output is the answer; None when no answer is returned as confirmed.
    answer_from: Literal["a", "b"] | None


def decide_cross_check(a: TrackState, b: TrackState, match: bool | None) -> CrossCheckDecision:
    """The cross-check verdict. `match` is `values_match` of the two outputs, None when
    they were not compared. First match wins.

    | # | program A | program B | outputs match | result                                   |
    |---|-----------|-----------|---------------|------------------------------------------|
    | 1 | ran       | ran       | yes           | verified, A's output                     |
    | 2 | ran       | ran       | no or unknown | disputed, both outputs shown, no answer  |
    | 3 | ran       | not ran   | -             | one_sided, A's output, not confirmed     |
    | 4 | not ran   | ran       | -             | one_sided, B's output, not confirmed     |
    | 5 | not ran   | not ran   | -             | failed, no answer, the run is an error   |

    "not ran" is program_failed, infra_failed or draft_failed after the program's own
    repair. Row 1 is the only way to `verified`. Row 2 with an unknown comparison is
    disputed: an absent comparison never counts as agreement.
    """
    if a == "ran" and b == "ran":
        if match is True:
            return CrossCheckDecision("verified", "a")
        return CrossCheckDecision("disputed", None)
    if a == "ran":
        return CrossCheckDecision("one_sided", "a")
    if b == "ran":
        return CrossCheckDecision("one_sided", "b")
    return CrossCheckDecision("failed", None)


@dataclass
class ProgramTrack:
    """One program of a cross-check, from its draft to its last run."""

    label: str
    tier: str
    state: TrackState = "draft_failed"
    steps: list[ChainStep] = field(default_factory=list)
    attempts: list[SandboxAttempt] = field(default_factory=list)
    final_attempt: SandboxAttempt | None = None
    repaired: bool = False
    effort: str | None = None
    # Why a draft or repair call of this track failed, when one did.
    draft_error_type: str | None = None
    repair_error_type: str | None = None

    @property
    def answer(self) -> str | None:
        return self.final_attempt.executed_answer if self.state == "ran" and self.final_attempt else None


def track_system_prompt(independent_method: bool) -> str:
    return EXECUTE_SYSTEM_PROMPT + BARE_ANSWER_LINE_RULE + (INDEPENDENT_METHOD_RULE if independent_method else "")


def track_repair_system_prompt() -> str:
    return REPAIR_SYSTEM_PROMPT + BARE_ANSWER_LINE_RULE


def run_program_track(
    label: str,
    question: str,
    conversation: list[dict[str, str]],
    tier_name: str,
    system_prompt: str,
    max_repairs: int,
    effort_override: str | None,
    call_tier: TierCaller,
    run_in_sandbox: Callable[[str], sandbox.SandboxResult],
    simulated_draft_failure: Callable[[], CallResult] | None = None,
) -> ProgramTrack:
    """Draft one program from the question, run it, repair it at most once from its own failure.

    Independence is structural: every message this function sends is built from its
    arguments, and none of them carries anything of the other program. `run_cross_check`
    calls it once per program with the same question and conversation.
    """
    track = ProgramTrack(label=label, tier=tier_name)
    draft_role, repair_role = _TRACK_ROLES[label]

    def draft_call(role: str, messages: list[dict[str, str]]) -> tuple[ExecuteDraft | None, str | None]:
        if role == draft_role and simulated_draft_failure is not None:
            call_result, effort = simulated_draft_failure(), effort_override
        else:
            call_result, effort = call_tier(tier_name, messages, draft_json_schema(), effort_override)
        track.effort = effort
        if not call_result.ok:
            track.steps.append(ChainStep(role, tier_name, effort, call_result))
            return None, call_result.error_type
        draft, _rejection = parse_draft(call_result)
        track.steps.append(ChainStep(role, tier_name, effort, call_result, invalid=draft is None))
        return draft, (None if draft is not None else ERROR_DRAFT_SCHEMA_INVALID)

    def run_attempt(attempt_label: Literal["first", "repair"], draft: ExecuteDraft) -> SandboxAttempt:
        result = run_in_sandbox(draft.program)
        state, failure_reason = classify_run(result)
        attempt = SandboxAttempt(
            attempt=attempt_label,
            program=draft.program,
            claimed_answer=draft.claimed_answer,
            result=result,
            outcome=state,
            failure_reason=failure_reason,
            executed_answer=executed_answer(result.stdout) if result.ok else None,
            program_label=label,
        )
        track.attempts.append(attempt)
        return attempt

    first_draft, track.draft_error_type = draft_call(
        draft_role, [{"role": "system", "content": system_prompt}] + conversation
    )
    if first_draft is None:
        return track
    first_attempt = run_attempt("first", first_draft)
    repair_state: TrackRepairState = "not_attempted"
    repair_attempt: SandboxAttempt | None = None
    decision = decide_track(first_attempt.outcome, repair_state, max_repairs > 0)
    if decision.action == "repair":
        repair_draft, track.repair_error_type = draft_call(
            repair_role, build_repair_messages(question, first_attempt, track_repair_system_prompt())
        )
        if repair_draft is None:
            repair_state = "draft_failed"
        else:
            repair_attempt = run_attempt("repair", repair_draft)
            repair_state = repair_attempt.outcome
        decision = decide_track(first_attempt.outcome, repair_state, repairs_allowed=False)
    track.state = decision.state
    if decision.use is not None:
        track.final_attempt = repair_attempt if decision.use == "repair" else first_attempt
        track.repaired = decision.use == "repair"
    return track


def _why_track_failed(track: ProgramTrack) -> str:
    if track.draft_error_type is not None:
        return f"the call that should have written it failed ({track.draft_error_type})"
    why = _why_unverified(track.attempts, track.repair_error_type)
    return why or "it did not run"


def _labelled_explanation(label: str, stdout: str, max_chars: int) -> str | None:
    explanation = executed_explanation(stdout)
    if explanation is None:
        return None
    if len(explanation) > max_chars:
        explanation = explanation[:max_chars].rstrip() + "\n" + EXPLANATION_TRUNCATED_MARK
    return f"Explanation printed by program {label.upper()}:\n{explanation}"


def run_cross_check(
    question: str,
    conversation: list[dict[str, str]],
    tier_a: str,
    tier_b: str,
    writer_a: str,
    writer_b: str,
    independent_method: bool,
    max_repairs: int,
    effort_override: str | None,
    call_tier_a: TierCaller,
    call_tier_b: TierCaller,
    run_in_sandbox: Callable[[str], sandbox.SandboxResult],
    simulated_answer_failure: Callable[[], CallResult] | None = None,
    parallel: bool = False,
) -> ExecuteOutcome:
    """Two programs from the question alone, each in its own sandbox, compared in code.

    `writer_a` and `writer_b` name who wrote each program (tier and model) for the answer
    text. `independent_method` adds INDEPENDENT_METHOD_RULE to B's prompt; the router sets
    it when both tiers run the same model. With `parallel` the two tracks run at the same
    time: they share no state, so the order cannot matter.
    """

    def track_a() -> ProgramTrack:
        return run_program_track(
            PROGRAM_A, question, conversation, tier_a, track_system_prompt(False), max_repairs,
            effort_override, call_tier_a, run_in_sandbox, simulated_answer_failure,
        )

    def track_b() -> ProgramTrack:
        return run_program_track(
            PROGRAM_B, question, conversation, tier_b, track_system_prompt(independent_method),
            max_repairs, effort_override, call_tier_b, run_in_sandbox,
        )

    if parallel:
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a, future_b = pool.submit(track_a), pool.submit(track_b)
            a, b = future_a.result(), future_b.result()
    else:
        a, b = track_a(), track_b()

    comparison = values_match(a.answer, b.answer) if a.answer is not None and b.answer is not None else None
    decision = decide_cross_check(a.state, b.state, comparison.match if comparison else None)

    outcome = ExecuteOutcome(tier=tier_a, cross_check=True, cross_check_tier=tier_b)
    outcome.steps = a.steps + b.steps
    outcome.attempts = a.attempts + b.attempts
    outcome.final_effort = a.effort
    outcome.execution_status = decision.status
    outcome.executed_answer, outcome.executed_answer_b = a.answer, b.answer
    outcome.cross_check_rule = comparison.rule if comparison else None
    outcome.repaired = any(track.repaired for track in (a, b) if track.state == "ran")
    outcome.repair_error_type = a.repair_error_type or b.repair_error_type

    if decision.status == "failed":
        outcome.execution_note = (
            f"FAILED: no answer. Neither program produced one. Program A ({writer_a}): "
            f"{_why_track_failed(a)}. Program B ({writer_b}): {_why_track_failed(b)}."
        )
        outcome.error_type, outcome.error_message = ERROR_NO_ANSWER, outcome.execution_note
        return outcome

    outcome.status = STATUS_OK
    if decision.status == "verified":
        note = (
            "Verified by two programs: two Python programs were written independently from the "
            f"question alone (A by {writer_a}, B by {writer_b}), each ran in its own isolated sandbox, "
            f"and they printed the same answer (compared as {comparison.rule}). Agreement does not "
            "rule out both reading the question wrongly in the same way."
        )
        if outcome.repaired:
            repaired = " and ".join(f"program {track.label.upper()}" for track in (a, b) if track.repaired)
            note += f" Repaired after a first failure, each from its own error only: {repaired}."
        outcome.execution_note = note
        outcome.executed_explanation = executed_explanation(a.final_attempt.result.stdout)
        outcome.answer = format_answer(a.answer, note, outcome.executed_explanation)
        return outcome

    if decision.status == "one_sided":
        ran, other = (a, b) if decision.answer_from == PROGRAM_A else (b, a)
        writers = {PROGRAM_A: writer_a, PROGRAM_B: writer_b}
        note = (
            "NOT CROSS-CHECKED: only one of the two programs produced an answer. Program "
            f"{ran.label.upper()} ({writers[ran.label]}) computed this in an isolated sandbox. Program "
            f"{other.label.upper()} ({writers[other.label]}) produced nothing to compare it with: "
            f"{_why_track_failed(other)}. Nothing independent agreed with this answer."
        )
        if ran.repaired:
            note += f" Program {ran.label.upper()} first failed and was repaired from its own error."
        outcome.execution_note = note
        outcome.executed_explanation = executed_explanation(ran.final_attempt.result.stdout)
        outcome.answer = format_answer(ran.answer, note, outcome.executed_explanation)
        return outcome

    # disputed: both outputs are shown and no answer line is given.
    note = (
        "DISPUTED: no answer is confirmed. Two Python programs were written independently from the "
        "question alone, each ran in its own isolated sandbox, and they printed different answers "
        f"(compared as {comparison.rule})."
    )
    outcome.execution_note = note
    explanations = [
        _labelled_explanation(track.label, track.final_attempt.result.stdout, EXPLANATION_MAX_CHARS // 2)
        for track in (a, b)
    ]
    outcome.executed_explanation = "\n\n".join(text for text in explanations if text) or None
    parts = [
        note,
        f"Program A ({writer_a}) printed: {strip_answer_label(a.answer)}\n"
        f"Program B ({writer_b}) printed: {strip_answer_label(b.answer)}",
    ]
    if outcome.executed_explanation:
        parts.append(outcome.executed_explanation)
    parts.append(DISPUTED_LAST_LINE)
    outcome.answer = "\n\n".join(parts)
    return outcome
