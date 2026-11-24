"""Auto mode: which strategy answers a question when the caller did not pick one.

The whole decision is `decide_mode`, a pure function. It encodes what the Phase 2 evals
measured (eval/results/2026-11-22-phase2-final.md): a single mid-tier call matched or
beat the chain closed-book, a stuffed call matched the grounded pipeline whenever the
sources fit one context window, and the pipeline is the only thing that works when they
do not fit. With the computation check on, a question without sources that a short program
can compute goes to execute mode (eval/results/2026-11-24-auto-execute.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from inference_arbiter.schemas import ComputationVerdict

AutoMode = Literal["route", "stuffed", "openbook", "execute"]

REASON_NO_SOURCES = "auto: no sources, one closed-book call"
REASON_COMPUTABLE = "auto: no sources, the computation check said a program can compute the answer"
REASON_NOT_COMPUTABLE = "auto: no sources, the computation check said no, one closed-book call"
REASON_CHECK_LOW_CONFIDENCE = (
    "auto: no sources, the computation check said yes below the confidence threshold, one closed-book call"
)
REASON_CHECK_FAILED = "auto: no sources, the computation check failed, one closed-book call"
REASON_CITATIONS_REQUIRED = "auto: citations required"
REASON_NO_WINDOW = "auto: the answering tier has no context_window_tokens, treated as not fitting"
REASON_LOAD_FAILED = "auto: sources could not be loaded"


@dataclass(frozen=True)
class AutoDecision:
    mode: AutoMode
    reason: str


def format_token_count(tokens: int) -> str:
    if tokens < 1000:
        return str(tokens)
    if tokens < 1_000_000:
        return f"{tokens / 1000:.0f}k"
    return f"{tokens / 1_000_000:.1f}M"


def describe_check(tier: str, verdict: ComputationVerdict, confidence: float | None) -> str:
    """The check as it is appended to the auto reason, so telemetry's `auto_reason` holds it."""
    if confidence is None:
        return f"check on {tier}: {verdict}"
    return f"check on {tier}: {verdict}, confidence {confidence:.2f}"


def execute_fallback_reason(error_type: str | None) -> str:
    """Why an auto run that the check sent to execute mode ended as a closed-book call."""
    return (
        f"{REASON_COMPUTABLE}; execution failed ({error_type or 'no answer'}), "
        "one closed-book call, not verified"
    )


def decide_mode(
    has_sources: bool,
    estimated_tokens: int,
    context_window_tokens: int | None,
    stuffed_fill_ratio: float,
    require_citations: bool,
    computation: ComputationVerdict = "not_run",
) -> AutoDecision:
    """Decision table. First match wins.

    | # | sources | computation check | require_citations | stuffed prompt vs window * ratio | mode     |
    |---|---------|-------------------|-------------------|----------------------------------|----------|
    | 1 | no      | yes               | -                 | -                                | execute  |
    | 2 | no      | not_run           | -                 | -                                | route    |
    | 3 | no      | no                | -                 | -                                | route    |
    | 4 | no      | low_confidence    | -                 | -                                | route    |
    | 5 | no      | failed            | -                 | -                                | route    |
    | 6 | yes     | -                 | yes               | -                                | openbook |
    | 7 | yes     | -                 | no                | window unknown                   | openbook |
    | 8 | yes     | -                 | no                | fits                             | stuffed  |
    | 9 | yes     | -                 | no                | does not fit                     | openbook |

    `route` here is one call to the closed-book tier with no triage. `computation` is the
    verdict of computation_check.decide; rows 2 to 5 differ only in the reason they report.
    With sources the verdict is not read: the router does not run the check for them.
    `estimated_tokens` is the estimate for the whole stuffed prompt (sources plus
    question). The request validator has already refused require_citations without sources.
    """
    if not has_sources:
        if computation == "yes":
            return AutoDecision("execute", REASON_COMPUTABLE)
        if computation == "no":
            return AutoDecision("route", REASON_NOT_COMPUTABLE)
        if computation == "low_confidence":
            return AutoDecision("route", REASON_CHECK_LOW_CONFIDENCE)
        if computation == "failed":
            return AutoDecision("route", REASON_CHECK_FAILED)
        return AutoDecision("route", REASON_NO_SOURCES)
    if require_citations:
        return AutoDecision("openbook", REASON_CITATIONS_REQUIRED)
    if context_window_tokens is None:
        return AutoDecision("openbook", REASON_NO_WINDOW)
    token_limit = int(context_window_tokens * stuffed_fill_ratio)
    sizes = f"{format_token_count(estimated_tokens)} of {format_token_count(token_limit)} tokens"
    if estimated_tokens <= token_limit:
        return AutoDecision("stuffed", f"auto: sources fit, {sizes}")
    return AutoDecision("openbook", f"auto: sources exceed context window, {sizes}")
