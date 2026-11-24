"""Five-question live smoke test for the Phase 1 router.

Runs against the real local Ollama daemon, so it is not part of pytest.
It never forces local_heavy or frontier; triage may still pick frontier,
which fails over to mid when no OpenRouter key is set.

Usage: uv run python scripts/smoke.py [--config PATH] [--db PATH]
"""

from __future__ import annotations

import argparse
import sys

from inference_arbiter.config import DEFAULT_CONFIG_PATH
from inference_arbiter.router import DEFAULT_DB_PATH, build_harness
from inference_arbiter.schemas import AskRequest

ANSWER_PREVIEW_CHARS = 70

SMOKE_CASES = [
    {"name": "simple fact", "question": "What is the capital of Japan?"},
    {"name": "code", "question": "Write a Python function that reverses a string."},
    {
        "name": "reasoning",
        "question": "Compare optimistic and pessimistic locking and say when each fits.",
    },
    {"name": "forced mid", "question": "Name three prime numbers.", "tier": "mid"},
    {
        "name": "triage timeout",
        "question": "What is two plus two?",
        "simulate": "triage-timeout",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 router smoke test")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    arguments = parser.parse_args()

    harness = build_harness(arguments.config, arguments.db)
    failed_count = 0

    for case in SMOKE_CASES:
        request = AskRequest.from_question(
            case["question"], tier=case.get("tier"), simulate=case.get("simulate"), mode="route"
        )
        result = harness.answer(request)
        if result.status != "ok":
            failed_count += 1
        preview = (result.answer or f"ERROR {result.error_type}").replace("\n", " ")
        print(
            f"{case['name']:<15} {result.status:<5} route={result.selected_tier:<8} "
            f"reason={result.route_reason:<14} final={result.final_tier:<8} "
            f"fallback={str(result.fallback_used):<5} {result.total_latency_ms:>9.1f} ms  "
            f"{preview[:ANSWER_PREVIEW_CHARS]}"
        )

    print(f"{len(SMOKE_CASES) - failed_count}/{len(SMOKE_CASES)} runs succeeded")
    if failed_count:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
