"""Auto mode: which strategy answers a question when the caller did not pick one.

The whole decision is `decide_mode`, a pure function. It encodes what the Phase 2 evals
measured (eval/results/2026-11-22-phase2-final.md): a single mid-tier call matched or
beat the chain closed-book, a stuffed call matched the grounded pipeline whenever the
sources fit one context window, and the pipeline is the only thing that works when they
do not fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AutoMode = Literal["route", "stuffed", "openbook"]

REASON_NO_SOURCES = "auto: no sources, one closed-book call"
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


def decide_mode(
    has_sources: bool,
    estimated_tokens: int,
    context_window_tokens: int | None,
    stuffed_fill_ratio: float,
    require_citations: bool,
) -> AutoDecision:
    """Decision table. First match wins.

    | # | sources | require_citations | stuffed prompt vs window * ratio | mode     |
    |---|---------|-------------------|----------------------------------|----------|
    | 1 | no      | -                 | -                                | route    |
    | 2 | yes     | yes               | -                                | openbook |
    | 3 | yes     | no                | window unknown                   | openbook |
    | 4 | yes     | no                | fits                             | stuffed  |
    | 5 | yes     | no                | does not fit                     | openbook |

    `route` here is one call to the closed-book tier with no triage. `estimated_tokens`
    is the estimate for the whole stuffed prompt (sources plus question). The request
    validator has already refused require_citations without sources.
    """
    if not has_sources:
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
