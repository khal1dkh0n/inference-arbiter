"""Triage with a decisions model (TypeSafe Jev through OpenRouter's Decisions API).

All question wording, thresholds and the truth table live in this one module so they
can be audited and edited together.

Jev is asked for two small judgments about the question text. It is never asked which
tier to use: that decision is the truth table in `decide_route`, in plain Python.
"""

from __future__ import annotations

import json
from typing import Any

from inference_arbiter.schemas import CallResult, TriageDecision, TriageOutcome

# The question text is sent as a named field so each question can refer to it by path.
STATE_FIELD = "user_question"

QUESTION_IS_CODE = "is_code"
QUESTION_DEPTH = "reasoning_depth"

# Depth levels, lowest first. Each one is written to make sense on its own.
DEPTH_LEVELS = [
    "Answerable by recalling one well-known fact or definition.",
    "Needs a short explanation or two or three simple steps.",
    "Needs several reasoning steps, careful calculation, or weighing more than one consideration.",
    "Needs long, subtle expert reasoning where one small mistake changes the answer.",
]
DEPTH_SIMPLE_MAX = 1  # levels 0-1 are everyday questions
DEPTH_MULTI_STEP = 2
DEPTH_EXPERT = 3

QUESTIONS: dict[str, dict[str, Any]] = {
    QUESTION_IS_CODE: {
        "type": "noul",
        "instructions": (
            f"Is `{STATE_FIELD}` asking for program code, SQL or shell commands to be "
            "written, explained, run in the head, or debugged?"
        ),
        "criteria": {
            "true": "The question contains or asks for code, a query, a command or a stack trace.",
            "false": "The question is about facts, concepts or reasoning and involves no code.",
        },
    },
    QUESTION_DEPTH: {
        "type": "score",
        "instructions": f"How much reasoning does a correct answer to `{STATE_FIELD}` need?",
        "criteria": DEPTH_LEVELS,
    },
}

# Comparison variant only (scripts/compare_triage.py): ask Jev to pick the tier directly.
# The harness does not route on this; it exists to measure it against the truth table.
QUESTION_DIRECT_TIER = "direct_tier"
DIRECT_TIER_QUESTION: dict[str, Any] = {
    "type": "choice",
    "instructions": f"Which kind of assistant is the cheapest that can answer `{STATE_FIELD}` well?",
    "criteria": {
        "local": "A small general model: everyday facts, definitions, short explanations.",
        "code": "A coding model: writing, explaining or debugging code, SQL or shell commands.",
        "mid": "A stronger general model: several reasoning steps, calculation or analysis.",
        "frontier": "The strongest model: long, subtle expert reasoning that is easy to get wrong.",
    },
}

# Starting thresholds for a yes/no judgment; the band between them is "not settled".
NOUL_YES = 0.7
NOUL_NO = 0.3


def build_state(triage_text: str) -> dict[str, str]:
    return {STATE_FIELD: triage_text}


def _top_depth_level(depth_answer: dict[str, Any]) -> tuple[int, float]:
    """Most probable depth level and its probability, read from the distribution."""
    probabilities = depth_answer["probabilities"]
    level_key = max(probabilities, key=lambda key: float(probabilities[key]))
    return int(level_key), float(probabilities[level_key])


def decide_route(is_code_probability: float, depth_level: int, depth_probability: float) -> TriageDecision:
    """Truth table: two judgments in, one route label out.

    Confidence is the probability of the judgment that picked the label. When the code
    judgment is not settled, confidence is lowered so the router's low-confidence rule
    escalates the question.
    """
    reason = f"jev: is_code={is_code_probability:.2f}, depth={depth_level} (p={depth_probability:.2f})"

    if depth_level >= DEPTH_EXPERT:
        return TriageDecision(label="frontier", confidence=depth_probability, reason=reason)

    if is_code_probability >= NOUL_YES:
        return TriageDecision(label="code", confidence=is_code_probability, reason=reason)

    label = "mid" if depth_level >= DEPTH_MULTI_STEP else "local"
    if is_code_probability > NOUL_NO:
        # Could be code, could be not: report the weaker of the two judgments.
        unsettled_confidence = min(depth_probability, 1.0 - is_code_probability)
        return TriageDecision(label=label, confidence=unsettled_confidence, reason=reason)

    return TriageDecision(label=label, confidence=depth_probability, reason=reason)


def interpret_answers(call_result: CallResult) -> TriageOutcome:
    """Turn a decisions call into a TriageOutcome. Fail-closed: no guessed label."""
    if not call_result.ok:
        failed_status = "timeout" if call_result.error_type == "timeout" else "error"
        return TriageOutcome(status=failed_status, call=call_result)
    try:
        answers = json.loads(call_result.text)
        is_code_probability = float(answers[QUESTION_IS_CODE]["noul"])
        depth_level, depth_probability = _top_depth_level(answers[QUESTION_DEPTH])
        if not 0 <= depth_level < len(DEPTH_LEVELS):
            raise ValueError(f"depth level {depth_level} is out of range")
        decision = decide_route(is_code_probability, depth_level, depth_probability)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return TriageOutcome(status="invalid", call=call_result)
    return TriageOutcome(status="ok", decision=decision, call=call_result)


def interpret_direct_tier(call_result: CallResult) -> TriageOutcome:
    """Read the comparison-only direct tier choice from the same decisions call."""
    if not call_result.ok:
        failed_status = "timeout" if call_result.error_type == "timeout" else "error"
        return TriageOutcome(status=failed_status, call=call_result)
    try:
        answer = json.loads(call_result.text)[QUESTION_DIRECT_TIER]
        label = answer["choice"]
        probability = float(answer["probabilities"][label])
        decision = TriageDecision(
            label=label, confidence=probability, reason=f"jev direct choice (p={probability:.2f})"
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return TriageOutcome(status="invalid", call=call_result)
    return TriageOutcome(status="ok", decision=decision, call=call_result)
