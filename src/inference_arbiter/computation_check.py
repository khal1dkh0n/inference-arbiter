"""Computation check: can a short program compute the answer to this question?

One typed yes/no judgment with a confidence, from one tier, asked by auto mode before its
closed-book call. The model judges; `decide` turns the judgment into a verdict in code.

Fail-closed: a timeout, client error, invalid output or a yes below the confidence
threshold is never a yes. There is no retry; the router then makes the closed-book call
it would have made without the check.
"""

from __future__ import annotations

from pydantic import ValidationError

from inference_arbiter.clients import ERROR_TIMEOUT, ChatCall, ChatClient
from inference_arbiter.config import HarnessConfig, TierConfig
from inference_arbiter.schemas import (
    CallResult,
    ComputationCheckDecision,
    ComputationCheckOutcome,
    ComputationVerdict,
    Message,
    TriageStatus,
)
from inference_arbiter.triage import build_triage_text

CHECK_SYSTEM_PROMPT = """You are a classifier. Do not answer the question. Decide one thing: can a short Python 3 program that uses only the standard library compute the answer to the user's question?

Say yes (computable: true) when the answer is a value that follows from data and rules given in the question itself: arithmetic, counting, enumeration or search, dates, parsing or decoding a string given in the question, checksums and hashes, or the output of a Python or SQL snippet given in the question.

Say no (computable: false) when the answer has to be known or judged, not computed: facts, definitions, names, what a standard or a product specifies, explanations, opinions, advice, or a question that depends on something that is not given (an attachment, a current price or rate, an unstated place).

Return JSON with:
- reason: at most 15 words
- computable: true or false
- confidence: a number from 0 to 1 for how sure you are"""


def build_check_call(check_tier: TierConfig, question_text: str) -> ChatCall:
    # Thinking is forced off whenever the tier declares a think setting, as for triage.
    think_setting = False if check_tier.think is not None else None
    temperature = check_tier.temperature if check_tier.temperature is not None else 0.0
    return ChatCall(
        model=check_tier.model,
        messages=[
            {"role": "system", "content": CHECK_SYSTEM_PROMPT},
            {"role": "user", "content": question_text},
        ],
        timeout_s=check_tier.timeout_s,
        effort=None,
        think=think_setting,
        num_ctx=check_tier.num_ctx,
        keep_alive=check_tier.keep_alive,
        temperature=temperature,
        max_output_tokens=check_tier.max_output_tokens,
        json_schema=ComputationCheckDecision.model_json_schema(),
        input_price_per_mtok=check_tier.input_price_per_mtok,
        output_price_per_mtok=check_tier.output_price_per_mtok,
    )


def interpret_check_result(call_result: CallResult) -> ComputationCheckOutcome:
    if not call_result.ok:
        failed_status = "timeout" if call_result.error_type == ERROR_TIMEOUT else "error"
        return ComputationCheckOutcome(status=failed_status, call=call_result)
    try:
        decision = ComputationCheckDecision.model_validate_json(call_result.text)
    except ValidationError:
        return ComputationCheckOutcome(status="invalid", call=call_result)
    return ComputationCheckOutcome(status="ok", decision=decision, call=call_result)


def run_computation_check(
    config: HarnessConfig, client: ChatClient, messages: list[Message]
) -> ComputationCheckOutcome:
    """Ask auto.computation_check_tier about the last user message. Never raises for a failed call."""
    check_tier = config.tiers[config.auto.computation_check_tier]
    question_text = build_triage_text(messages, config.triage_max_chars)
    return interpret_check_result(client.chat(build_check_call(check_tier, question_text)))


def decide(
    status: TriageStatus, computable: bool | None, confidence: float | None, threshold: float
) -> ComputationVerdict:
    """Decision table. First match wins.

    | # | check call status          | computable | confidence   | verdict        |
    |---|----------------------------|------------|--------------|----------------|
    | 1 | skipped                    | -          | -            | not_run        |
    | 2 | timeout, error or invalid  | -          | -            | failed         |
    | 3 | ok                         | missing    | -            | failed         |
    | 4 | ok                         | no         | -            | no             |
    | 5 | ok                         | yes        | missing      | failed         |
    | 6 | ok                         | yes        | < threshold  | low_confidence |
    | 7 | ok                         | yes        | >= threshold | yes            |

    Only `yes` sends a question to execute mode (auto.decide_mode). Rows 3 and 5 cannot
    happen after schema validation; they are here so a missing value is never a yes.
    """
    if status == "skipped":
        return "not_run"
    if status != "ok" or computable is None:
        return "failed"
    if not computable:
        return "no"
    if confidence is None:
        return "failed"
    if confidence < threshold:
        return "low_confidence"
    return "yes"


def verdict_of(outcome: ComputationCheckOutcome, threshold: float) -> ComputationVerdict:
    decision = outcome.decision
    return decide(
        outcome.status,
        decision.computable if decision else None,
        decision.confidence if decision else None,
        threshold,
    )
