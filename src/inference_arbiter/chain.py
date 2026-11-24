"""Phase 2 verification chain: draft, schema gate, verify, at most one repair.

Plain Python control flow with a fixed upper bound of four model calls after triage:
at most two draft attempts, one verifier call and one repair call. There is no loop.

1. Draft on the tier the router selected. The model must return a DraftAnswer.
2. Schema gate. Output that is not a valid DraftAnswer never moves on. The draft gets
   ONE second attempt: on the `escalation` tier (or `fallback_tier`) when the output was
   invalid, or on the tier's `on_error` target when the call itself failed. If the
   second attempt also fails, the run fails with an explicit error.
3. Verify with the configured verifier tier (see verifier.py for what a verdict means).
4. Repair at most once, on `chain.repair_tier`, when the verdict is unsupported or
   uncertain (or the verifier failed and `on_verifier_error` is `repair`). The repair
   output passes the same schema gate with no second attempt and is not re-verified.
   The repair call never uses the `on_error` hop: a weaker tier's output is never
   presented as a repair. If the repair fails, the draft is returned and the failure
   is recorded.

This module does not record telemetry. It returns every model call as a ChainStep and
the router turns those into call rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import ValidationError

from inference_arbiter import verifier
from inference_arbiter.clients import AnyClient
from inference_arbiter.config import ChainConfig, HarnessConfig
from inference_arbiter.schemas import (
    ERROR_SIMULATED_ERROR,
    STATUS_ERROR,
    STATUS_OK,
    AskRequest,
    CallResult,
    DraftAnswer,
    VerificationStatus,
)

ROLE_DRAFT = "draft"
ROLE_VERIFY = "verify"
ROLE_REPAIR = "repair"

ERROR_SCHEMA_INVALID = "schema_invalid"
ERROR_SIMULATED_INVALID = "simulated_invalid"
ERROR_DRAFT_SCHEMA_INVALID = "draft_schema_invalid"

MAX_VALIDATION_MESSAGE_CHARS = 300

DRAFT_SYSTEM_PROMPT = """Answer the user's question.

Return JSON with these fields, in this order:
- working: work the question out here first, step by step: do the arithmetic, trace the code, or recall the relevant facts. One line is enough for a simple fact.
- answer: your complete answer, written for the user as plain text, consistent with your working
- key_claims: the factual or computed claims your answer rests on, each one a short sentence that can be checked on its own
- assumptions: anything you assumed that the question did not state; an empty list if nothing"""

REPAIR_SYSTEM_PROMPT = """You are repairing a draft answer. You are given the question, a draft answer from another model, and a verifier's findings about that draft.

Work the question out yourself. The verifier can be wrong: keep what is correct in the draft, fix what is wrong, and do not adopt a criticism you find mistaken. If the question rests on a false premise, say so in the answer.

Return JSON with these fields, in this order:
- working: work the question out here first, step by step
- answer: the corrected complete answer, written for the user as plain text; do not mention the draft or the verifier
- key_claims: the factual or computed claims your answer rests on, each one a short sentence that can be checked on its own
- assumptions: anything you assumed that the question did not state; an empty list if nothing"""

SAME_MODEL_NOTE = (
    "not verified: the drafting tier and the verifier run the same model, and no "
    "alt_verifier_tier is configured"
)

NO_VERDICT_NOTE = (
    "The verifier was unavailable, so there are no findings. Check the draft yourself from scratch."
)

# (tier_name, messages, json_schema or None for a plain answer, effort_override)
#   -> (call result, effort used)
TierCaller = Callable[
    [str, list[dict[str, str]], dict[str, Any] | None, str | None], tuple[CallResult, str | None]
]
# failed_tier -> (fallback tier or None, skipped reason or None)
FallbackResolver = Callable[[str], tuple[str | None, str | None]]


@dataclass
class ChainStep:
    """One model call made by the chain, in order."""

    role: str  # draft | verify | repair
    tier: str
    effort: str | None
    result: CallResult
    invalid: bool = False  # the call returned, but its output failed validation


@dataclass
class ChainOutcome:
    draft_tier: str
    final_tier: str
    verifier_tier: str
    status: str = STATUS_ERROR
    answer: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    final_effort: str | None = None
    fallback_used: bool = False
    fallback_skipped_reason: str | None = None
    schema_escalated: bool = False
    verification_status: VerificationStatus = "skipped"
    verification_critique: str | None = None
    repaired: bool = False
    repair_error_type: str | None = None
    steps: list[ChainStep] = field(default_factory=list)


def draft_json_schema() -> dict[str, Any]:
    """Schema sent to draft and repair models: `working` is required and comes first."""
    schema = DraftAnswer.model_json_schema()
    required = [name for name in schema.get("required", []) if name != "working"]
    schema["required"] = ["working", *required]
    return schema


def parse_draft(call_result: CallResult) -> tuple[DraftAnswer | None, str | None]:
    """The schema gate. Returns (draft, None) or (None, why it was rejected)."""
    try:
        return DraftAnswer.model_validate_json(call_result.text), None
    except ValidationError as validation_error:
        return None, " ".join(str(validation_error).split())[:MAX_VALIDATION_MESSAGE_CHARS]


def build_draft_messages(conversation: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"role": "system", "content": DRAFT_SYSTEM_PROMPT}] + conversation


def build_repair_messages(
    question: str, draft: DraftAnswer, critique: str | None, failed_claims: list[str]
) -> list[dict[str, str]]:
    if critique is None:
        findings = NO_VERDICT_NOTE
    else:
        findings = f"Critique: {critique}"
        if failed_claims:
            findings += "\nClaims the verifier rejected:\n" + "\n".join(
                f"- {claim}" for claim in failed_claims
            )
    user_text = (
        f"QUESTION:\n{question}\n\n"
        f"DRAFT ANSWER (JSON):\n{draft.model_dump_json(indent=2)}\n\n"
        f"VERIFIER FINDINGS:\n{findings}"
    )
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def _simulated_invalid_draft() -> CallResult:
    return CallResult(
        text="simulated draft that is not valid JSON",
        status=STATUS_OK,
        error_message="invalid draft injected by --simulate; no model was called",
        latency_ms=0.0,
    )


def _simulated_verifier_failure() -> CallResult:
    return CallResult(
        status=STATUS_ERROR,
        error_type=ERROR_SIMULATED_ERROR,
        error_message="failure injected by --simulate; no model was called",
        latency_ms=0.0,
    )


def _gated_call(
    role: str,
    tier_name: str,
    call_result: CallResult,
    effort: str | None,
    invalid_error_type: str = ERROR_SCHEMA_INVALID,
) -> tuple[ChainStep, DraftAnswer | None]:
    """Wrap a draft or repair call result in a step and run it through the schema gate."""
    if not call_result.ok:
        return ChainStep(role=role, tier=tier_name, effort=effort, result=call_result), None

    draft, rejection = parse_draft(call_result)
    if draft is None:
        marked_result = call_result.model_copy(
            update={
                "error_type": invalid_error_type,
                "error_message": call_result.error_message or rejection,
            }
        )
        return ChainStep(role=role, tier=tier_name, effort=effort, result=marked_result, invalid=True), None

    return ChainStep(role=role, tier=tier_name, effort=effort, result=call_result), draft


def run_chain(
    config: HarnessConfig,
    chain_config: ChainConfig,
    clients: dict[str, AnyClient],
    request: AskRequest,
    question: str,
    conversation: list[dict[str, str]],
    draft_tier: str,
    verifier_tier_name: str,
    call_tier: TierCaller,
    simulated_answer_failure: Callable[[], CallResult],
    resolve_fallback: FallbackResolver,
) -> ChainOutcome:
    outcome = ChainOutcome(draft_tier=draft_tier, final_tier=draft_tier, verifier_tier=verifier_tier_name)
    draft_schema = draft_json_schema()
    draft_messages = build_draft_messages(conversation)

    # Step 1: draft.
    if request.simulate == "answer-error":
        first_step, draft = _gated_call(
            ROLE_DRAFT, draft_tier, simulated_answer_failure(), config.tiers[draft_tier].effort
        )
    elif request.simulate == "draft-invalid":
        first_step, draft = _gated_call(
            ROLE_DRAFT,
            draft_tier,
            _simulated_invalid_draft(),
            config.tiers[draft_tier].effort,
            invalid_error_type=ERROR_SIMULATED_INVALID,
        )
    else:
        first_result, first_effort = call_tier(draft_tier, draft_messages, draft_schema, request.effort)
        first_step, draft = _gated_call(ROLE_DRAFT, draft_tier, first_result, first_effort)
    outcome.steps.append(first_step)
    outcome.final_effort = first_step.effort

    # Step 2: schema gate. One second attempt, then the draft stage is over.
    if draft is None:
        if first_step.invalid:
            second_tier: str | None = config.escalation.get(draft_tier, config.fallback_tier)
            outcome.schema_escalated = True
        else:
            second_tier, outcome.fallback_skipped_reason = resolve_fallback(draft_tier)
            if second_tier is None:
                outcome.error_type = first_step.result.error_type
                outcome.error_message = first_step.result.error_message
                return outcome
            outcome.fallback_used = True

        second_result, second_effort = call_tier(second_tier, draft_messages, draft_schema, request.effort)
        second_step, draft = _gated_call(ROLE_DRAFT, second_tier, second_result, second_effort)
        outcome.steps.append(second_step)
        outcome.final_tier = second_tier
        outcome.final_effort = second_effort

        if draft is None:
            if second_step.invalid:
                outcome.error_type = ERROR_DRAFT_SCHEMA_INVALID
                outcome.error_message = (
                    f"draft from tier '{second_tier}' failed schema validation after one "
                    f"second attempt: {second_step.result.error_message}"
                )
            else:
                outcome.error_type = second_step.result.error_type
                outcome.error_message = second_step.result.error_message
            return outcome

    # From here on there is a valid draft, so the run has an answer whatever happens next.
    outcome.status = STATUS_OK
    outcome.answer = draft.answer

    # Step 3: verify. A model never verifies its own draft.
    drafting_model = config.tiers[outcome.final_tier].model
    if config.tiers[verifier_tier_name].model == drafting_model:
        alt_verifier = chain_config.alt_verifier_tier
        if alt_verifier is None or config.tiers[alt_verifier].model == drafting_model:
            outcome.verification_status = "unavailable"
            outcome.verification_critique = SAME_MODEL_NOTE
            if chain_config.on_verifier_error != "repair" or chain_config.max_repairs < 1:
                return outcome
            return _repair(outcome, chain_config, call_tier, question, conversation, draft, None, [])
        verifier_tier_name = alt_verifier
        outcome.verifier_tier = alt_verifier

    verifier_tier = config.tiers[verifier_tier_name]
    if request.simulate == "verifier-error":
        verifier_outcome = verifier.VerifierOutcome(
            status="error", verdict=None, call=_simulated_verifier_failure()
        )
    else:
        verifier_outcome = verifier.run_verifier(
            verifier_tier, clients[verifier_tier.provider], question, draft
        )
    outcome.steps.append(
        ChainStep(
            role=ROLE_VERIFY,
            tier=verifier_tier_name,
            effort=verifier_tier.effort,
            result=verifier_outcome.call,
            invalid=verifier_outcome.status == "invalid",
        )
    )

    verdict = verifier_outcome.verdict
    if verdict is None:
        outcome.verification_status = "unavailable"
        needs_repair = chain_config.on_verifier_error == "repair"
        critique: str | None = None
        failed_claims: list[str] = []
    else:
        outcome.verification_status = verdict.verdict
        outcome.verification_critique = verdict.critique
        needs_repair = verdict.verdict != "supported"
        critique = verdict.critique
        failed_claims = verdict.failed_claims

    if not needs_repair or chain_config.max_repairs < 1:
        return outcome

    return _repair(outcome, chain_config, call_tier, question, conversation, draft, critique, failed_claims)


def _repair(
    outcome: ChainOutcome,
    chain_config: ChainConfig,
    call_tier: TierCaller,
    question: str,
    conversation: list[dict[str, str]],
    draft: DraftAnswer,
    critique: str | None,
    failed_claims: list[str],
) -> ChainOutcome:
    """Step 4: one call to the repair tier. No on_error hop, no second attempt, no re-verification.

    If the call fails the draft stands, and final_tier stays on the tier that wrote it.
    """
    repair_tier = chain_config.repair_tier

    if chain_config.repair_style == "escalate":
        # Clean escalation: the stronger model sees only the original question, so it is
        # not anchored on the rejected draft and the call costs the same as a direct one.
        result, effort = call_tier(repair_tier, conversation, None, None)
        answer_text = result.text.strip() if result.ok else ""
        outcome.steps.append(ChainStep(role=ROLE_REPAIR, tier=repair_tier, effort=effort, result=result))
        if not answer_text:
            outcome.repair_error_type = result.error_type or "empty"
            return outcome
    else:
        repair_messages = build_repair_messages(question, draft, critique, failed_claims)
        result, effort = call_tier(repair_tier, repair_messages, draft_json_schema(), None)
        repair_step, repaired_draft = _gated_call(ROLE_REPAIR, repair_tier, result, effort)
        outcome.steps.append(repair_step)
        if repaired_draft is None:
            outcome.repair_error_type = repair_step.result.error_type
            return outcome
        answer_text = repaired_draft.answer

    outcome.repaired = True
    outcome.answer = answer_text
    outcome.final_tier = repair_tier
    outcome.final_effort = effort
    return outcome
