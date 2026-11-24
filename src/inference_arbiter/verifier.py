"""Chain-mode verifier: a second opinion on a draft answer.

Honest scope: the questions are closed-book. No verifier here has sources to check the
draft against, so a verdict is a second model's opinion, not grounding. It catches
answers that miss the question, swallow a false premise, contradict themselves or
state something a different model believes is wrong. It cannot catch an error both
models share. Grounding against fetched sources is a later step.

Two kinds of verifier tier:
- a chat model, which re-derives the answer independently and returns a VerifierVerdict;
- a decisions model (TypeSafe Jev), which is asked small typed judgments. Jev is never
  asked for the verdict: that is the truth table in `decide_verdict`, in plain Python.

All prompt wording, Jev question wording, thresholds and the truth table live in this
one module so they can be audited and edited together.

Fail-closed: a failed call, a timeout or an invalid verdict is a verifier error, never
a guessed verdict. The chain decides what to do with a verifier error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from inference_arbiter.clients import (
    ERROR_TIMEOUT,
    ChatCall,
    ChatClient,
    DecisionsCall,
    OpenRouterDecisionsClient,
)
from inference_arbiter.config import TierConfig
from inference_arbiter.schemas import CallResult, DraftAnswer, VerifierVerdict

ERROR_VERDICT_INVALID = "verdict_invalid"

VerifierStatus = Literal["ok", "timeout", "error", "invalid"]


@dataclass
class VerifierOutcome:
    status: VerifierStatus
    verdict: VerifierVerdict | None
    call: CallResult


# --- Chat verifier -----------------------------------------------------------------

VERIFIER_SYSTEM_PROMPT = """You are a strict verifier. You are given a question and another model's draft answer to it. You have no sources: check the draft against the question using your own knowledge and reasoning.

Work in this order:
1. Read the question alone and work out your own answer first. Redo any arithmetic or logic step by step.
2. Check whether the question rests on a false premise, can be read in two ways that give different answers, or leaves out information the answer depends on (a rate, a date, a unit, which thing is meant). If so, the draft must say so. A draft that silently picks one reading, or fills the gap with a value of its own, has not answered the question asked.
3. Compare the draft's answer and each of its key claims with your own result.

Verdicts:
- supported: the draft answers the question asked, and you find no error in the answer or its key claims.
- unsupported: the draft is wrong, answers a different question, accepts a false premise, answers an ambiguous or underspecified question without flagging it, or contradicts its own key claims.
- uncertain: you cannot tell, or you and the draft disagree and you are not sure which is right.

Do not reward confident wording. A missing detail that does not change the answer is not an error.

Return JSON with:
- verdict: one of supported, unsupported, uncertain
- critique: what is wrong and what the correct result is, specific enough to fix the answer; one short sentence when supported
- failed_claims: the key claims (or parts of the answer) you found wrong, quoted; an empty list when supported"""


def build_verifier_user_text(question: str, draft: DraftAnswer) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"DRAFT ANSWER (JSON):\n{draft.model_dump_json(indent=2)}"
    )


def build_verifier_call(verifier_tier: TierConfig, question: str, draft: DraftAnswer) -> ChatCall:
    if verifier_tier.think == "effort":
        think_setting: bool | str | None = verifier_tier.effort
    else:
        think_setting = verifier_tier.think
    temperature = verifier_tier.temperature if verifier_tier.temperature is not None else 0.0
    return ChatCall(
        model=verifier_tier.model,
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": build_verifier_user_text(question, draft)},
        ],
        timeout_s=verifier_tier.timeout_s,
        effort=verifier_tier.effort,
        think=think_setting,
        num_ctx=verifier_tier.num_ctx,
        keep_alive=verifier_tier.keep_alive,
        temperature=temperature,
        max_output_tokens=verifier_tier.max_output_tokens,
        json_schema=VerifierVerdict.model_json_schema(),
        input_price_per_mtok=verifier_tier.input_price_per_mtok,
        output_price_per_mtok=verifier_tier.output_price_per_mtok,
    )


def _failed_outcome(call_result: CallResult) -> VerifierOutcome:
    failed_status: VerifierStatus = "timeout" if call_result.error_type == ERROR_TIMEOUT else "error"
    return VerifierOutcome(status=failed_status, verdict=None, call=call_result)


def _invalid_outcome(call_result: CallResult, detail: str) -> VerifierOutcome:
    marked_call = call_result.model_copy(
        update={"error_type": ERROR_VERDICT_INVALID, "error_message": detail[:300]}
    )
    return VerifierOutcome(status="invalid", verdict=None, call=marked_call)


def interpret_chat_verdict(call_result: CallResult) -> VerifierOutcome:
    if not call_result.ok:
        return _failed_outcome(call_result)
    try:
        verdict = VerifierVerdict.model_validate_json(call_result.text)
    except ValidationError as validation_error:
        return _invalid_outcome(call_result, f"verifier output failed validation: {validation_error}")
    return VerifierOutcome(status="ok", verdict=verdict, call=call_result)


# --- Jev (decisions model) verifier ------------------------------------------------

# State sent to Jev. Each question refers to these fields by path.
STATE_QUESTION = "user_question"
STATE_DRAFT = "draft"

QUESTION_ANSWERS = "answers_question"
QUESTION_FALSE_PREMISE = "unflagged_false_premise"
QUESTION_CONTRADICTION = "self_contradiction"

JEV_QUESTIONS: dict[str, dict[str, Any]] = {
    QUESTION_ANSWERS: {
        "type": "noul",
        "instructions": (
            f"Does `{STATE_DRAFT}.answer` directly answer what `{STATE_QUESTION}` asks?"
        ),
        "criteria": {
            "true": "The answer addresses the exact question asked and commits to a result.",
            "false": (
                "The answer addresses a different question, dodges, or leaves the question "
                "unanswered."
            ),
        },
    },
    QUESTION_FALSE_PREMISE: {
        "type": "noul",
        "instructions": (
            f"Does `{STATE_QUESTION}` rest on a false premise that `{STATE_DRAFT}.answer` "
            "fails to point out?"
        ),
        "criteria": {
            "true": (
                "The question assumes something untrue and the answer goes along with it "
                "without correcting it."
            ),
            "false": (
                "The question has no false premise, or the answer explicitly corrects the "
                "false premise."
            ),
        },
    },
    QUESTION_CONTRADICTION: {
        "type": "noul",
        "instructions": (
            f"Does `{STATE_DRAFT}.answer` contradict any item in `{STATE_DRAFT}.key_claims`?"
        ),
        "criteria": {
            "true": "The answer states or implies something that conflicts with a listed key claim.",
            "false": "The answer is consistent with every listed key claim.",
        },
    },
}

# Starting thresholds for a yes/no judgment; the band between them is "not settled".
NOUL_YES = 0.7
NOUL_NO = 0.3

CRITIQUE_NOT_ANSWERED = "The draft does not directly answer the question that was asked."
CRITIQUE_FALSE_PREMISE = (
    "The question rests on a false premise and the draft does not point it out."
)
CRITIQUE_CONTRADICTION = "The draft answer contradicts its own key claims."
CRITIQUE_SUPPORTED = "No objection from the typed checks."


def build_jev_state(question: str, draft: DraftAnswer) -> dict[str, Any]:
    return {STATE_QUESTION: question, STATE_DRAFT: draft.model_dump()}


def decide_verdict(
    answers_probability: float, false_premise_probability: float, contradiction_probability: float
) -> VerifierVerdict:
    """Truth table: three judgments in, one verdict out. First match wins.

    1. Any settled defect (answer misses the question, unflagged false premise,
       self-contradiction) is `unsupported`.
    2. Otherwise any judgment inside the band between NOUL_NO and NOUL_YES is `uncertain`.
    3. Otherwise `supported`.

    The critique names every check that did not pass cleanly, with its probability.
    Jev cannot say which claim is wrong, so failed_claims is always empty.
    """
    defect_notes: list[str] = []
    unsettled_notes: list[str] = []

    if answers_probability <= NOUL_NO:
        defect_notes.append(f"{CRITIQUE_NOT_ANSWERED} (answers_question p={answers_probability:.2f})")
    elif answers_probability < NOUL_YES:
        unsettled_notes.append(
            f"Not settled whether the draft answers the question (p={answers_probability:.2f})."
        )

    if false_premise_probability >= NOUL_YES:
        defect_notes.append(
            f"{CRITIQUE_FALSE_PREMISE} (unflagged_false_premise p={false_premise_probability:.2f})"
        )
    elif false_premise_probability > NOUL_NO:
        unsettled_notes.append(
            "Not settled whether the question has a false premise the draft misses "
            f"(p={false_premise_probability:.2f})."
        )

    if contradiction_probability >= NOUL_YES:
        defect_notes.append(
            f"{CRITIQUE_CONTRADICTION} (self_contradiction p={contradiction_probability:.2f})"
        )
    elif contradiction_probability > NOUL_NO:
        unsettled_notes.append(
            "Not settled whether the draft contradicts its key claims "
            f"(p={contradiction_probability:.2f})."
        )

    if defect_notes:
        return VerifierVerdict(
            verdict="unsupported", critique=" ".join(defect_notes + unsettled_notes), failed_claims=[]
        )
    if unsettled_notes:
        return VerifierVerdict(verdict="uncertain", critique=" ".join(unsettled_notes), failed_claims=[])
    return VerifierVerdict(verdict="supported", critique=CRITIQUE_SUPPORTED, failed_claims=[])


def _read_noul(answers: dict[str, Any], question_name: str) -> float:
    probability = float(answers[question_name]["noul"])
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{question_name} probability {probability} is out of range")
    return probability


def interpret_jev_answers(call_result: CallResult) -> VerifierOutcome:
    """Turn a decisions call into a VerifierOutcome. Fail-closed: no guessed verdict."""
    if not call_result.ok:
        return _failed_outcome(call_result)
    try:
        answers = json.loads(call_result.text)
        verdict = decide_verdict(
            _read_noul(answers, QUESTION_ANSWERS),
            _read_noul(answers, QUESTION_FALSE_PREMISE),
            _read_noul(answers, QUESTION_CONTRADICTION),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as parse_error:
        return _invalid_outcome(call_result, f"decisions answers unusable: {parse_error!r}")
    return VerifierOutcome(status="ok", verdict=verdict, call=call_result)


# --- Entry point -------------------------------------------------------------------


def run_verifier(
    verifier_tier: TierConfig,
    client: ChatClient | OpenRouterDecisionsClient,
    question: str,
    draft: DraftAnswer,
) -> VerifierOutcome:
    """Verify with the given verifier tier: a chat model or a decisions model."""
    if isinstance(client, OpenRouterDecisionsClient):
        decisions_call = DecisionsCall(
            model=verifier_tier.model,
            state=build_jev_state(question, draft),
            questions=JEV_QUESTIONS,
            timeout_s=verifier_tier.timeout_s,
        )
        return interpret_jev_answers(client.decide(decisions_call))

    return interpret_chat_verdict(client.chat(build_verifier_call(verifier_tier, question, draft)))
