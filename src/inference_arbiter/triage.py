"""Text-only triage: classify the last user message into a route label.

Fail-closed: a timeout, client error or invalid output yields a failed status.
There is no retry and no guessed label; the router owns the fallback.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from inference_arbiter import jev_triage
from inference_arbiter.clients import (
    ERROR_TIMEOUT,
    ChatCall,
    ChatClient,
    DecisionsCall,
    OpenRouterDecisionsClient,
)
from inference_arbiter.config import HarnessConfig, TierConfig
from inference_arbiter.schemas import CallResult, Message, TriageDecision, TriageOutcome

TRUNCATION_MARKER = "\n[... truncated for triage ...]\n"

TRIAGE_SYSTEM_PROMPT = """You are a routing classifier. Do not answer the question. Pick the cheapest tier that can answer it well.

Labels:
- local: everyday questions, definitions, simple facts, short rewrites or summaries.
- code: writing, explaining or debugging code, SQL, shell commands, stack traces.
- mid: multi-step reasoning, comparisons, analysis, or anything needing broad or recent knowledge.
- frontier: high-ambiguity or high-stakes problems, long subtle reasoning, expert-level design or proofs.

Return JSON with:
- label: one of local, code, mid, frontier
- confidence: a number from 0 to 1 for how sure you are of the label
- reason: at most 15 words"""


def extract_text(content: str | list[dict[str, Any]]) -> str:
    """Return only the text of a message, dropping images and other attachments."""
    if isinstance(content, str):
        return content
    text_parts: list[str] = []
    for part in content:
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
    return "\n".join(text_parts)


def last_user_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return extract_text(message.content)
    return ""


def truncate_head_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return text[:head_chars] + TRUNCATION_MARKER + text[-tail_chars:]


def build_triage_text(messages: list[Message], max_chars: int) -> str:
    return truncate_head_tail(last_user_text(messages), max_chars)


def build_triage_call(triage_tier: TierConfig, triage_text: str) -> ChatCall:
    # Thinking is forced off for triage whenever the tier declares a think setting;
    # models without thinking support must leave think unset in the config.
    think_setting = False if triage_tier.think is not None else None
    temperature = triage_tier.temperature if triage_tier.temperature is not None else 0.0
    return ChatCall(
        model=triage_tier.model,
        messages=[
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
            {"role": "user", "content": triage_text},
        ],
        timeout_s=triage_tier.timeout_s,
        effort=None,
        think=think_setting,
        num_ctx=triage_tier.num_ctx,
        keep_alive=triage_tier.keep_alive,
        temperature=temperature,
        max_output_tokens=triage_tier.max_output_tokens,
        json_schema=TriageDecision.model_json_schema(),
        input_price_per_mtok=triage_tier.input_price_per_mtok,
        output_price_per_mtok=triage_tier.output_price_per_mtok,
    )


def interpret_triage_result(call_result: CallResult) -> TriageOutcome:
    if not call_result.ok:
        failed_status = "timeout" if call_result.error_type == ERROR_TIMEOUT else "error"
        return TriageOutcome(status=failed_status, call=call_result)
    try:
        decision = TriageDecision.model_validate_json(call_result.text)
    except ValidationError:
        return TriageOutcome(status="invalid", call=call_result)
    return TriageOutcome(status="ok", decision=decision, call=call_result)


def run_triage(
    config: HarnessConfig,
    client: ChatClient | OpenRouterDecisionsClient,
    messages: list[Message],
) -> TriageOutcome:
    """Classify with the configured triage tier: a chat model or a decisions model."""
    triage_tier = config.tiers[config.triage_tier]
    triage_text = build_triage_text(messages, config.triage_max_chars)

    if isinstance(client, OpenRouterDecisionsClient):
        decisions_call = DecisionsCall(
            model=triage_tier.model,
            state=jev_triage.build_state(triage_text),
            questions=jev_triage.QUESTIONS,
            timeout_s=triage_tier.timeout_s,
        )
        return jev_triage.interpret_answers(client.decide(decisions_call))

    call_result = client.chat(build_triage_call(triage_tier, triage_text))
    return interpret_triage_result(call_result)
