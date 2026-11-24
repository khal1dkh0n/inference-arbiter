"""Compare triage tiers on the golden question set. Runs triage only: no answers.

For each golden case and each triage tier it records the label, the confidence, the
tier the router would actually pick (after the low-confidence rule), latency and cost,
then scores them against the case's expected_route.

    uv run python scripts/compare_triage.py
    uv run python scripts/compare_triage.py --tiers triage triage_jev --out data/triage.json
    uv run python scripts/compare_triage.py --routes data/eval/measured_routes.json

With --routes it also scores against the measured routes written by scripts/run_eval.py
(the cheapest tier that actually answered each case correctly) and reports both scores.

A decisions tier (Jev) is read two ways from one request: the harness truth table, and
a comparison-only direct tier choice, reported as "<tier>:direct".
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import yaml

from inference_arbiter import jev_triage
from inference_arbiter.clients import DecisionsCall, OpenRouterDecisionsClient, build_clients
from inference_arbiter.config import DEFAULT_CONFIG_PATH, HarnessConfig, load_config, load_env_file
from inference_arbiter.router import choose_route
from inference_arbiter.schemas import Message, TriageOutcome
from inference_arbiter.triage import build_triage_text, run_triage

DEFAULT_GOLDEN_PATH = Path("eval/golden_questions.yaml")
# How strong each answer tier is. local and code are the same strength, different skill.
TIER_STRENGTH = {"local": 0, "code": 0, "mid": 1, "frontier": 2}
# Measured-route statuses that carry a route to score against (see scripts/run_eval.py).
SCORABLE_ROUTE_STATUSES = ("measured", "provisional")


def load_measured_routes(path: str | Path) -> dict[str, dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["routes"]


def score_against_measured(routed_tier: str, measured: dict | None) -> dict:
    """Score one routed tier against one measured-route entry. None fields mean not scorable."""
    if measured is None or measured["status"] not in SCORABLE_ROUTE_STATUSES:
        return {
            "measured_status": measured["status"] if measured else "absent",
            "measured_route": None,
            "measured_acceptable": [],
            "measured_correct": None,
            "measured_under": None,
            "measured_over": None,
        }
    measured_strength = TIER_STRENGTH[measured["route"]]
    return {
        "measured_status": measured["status"],
        "measured_route": measured["route"],
        "measured_acceptable": measured["acceptable"],
        "measured_correct": routed_tier in measured["acceptable"],
        "measured_under": TIER_STRENGTH[routed_tier] < measured_strength,
        "measured_over": TIER_STRENGTH[routed_tier] > measured_strength,
    }


def triage_once(config: HarnessConfig, clients: dict, tier_name: str, question: str) -> dict[str, TriageOutcome]:
    """Run one triage tier on one question. Returns outcomes keyed by candidate name."""
    config.triage_tier = tier_name
    tier = config.tiers[tier_name]
    client = clients[tier.provider]
    messages = [Message(role="user", content=question)]

    if not isinstance(client, OpenRouterDecisionsClient):
        return {tier_name: run_triage(config, client, messages)}

    questions = dict(jev_triage.QUESTIONS)
    questions[jev_triage.QUESTION_DIRECT_TIER] = jev_triage.DIRECT_TIER_QUESTION
    call_result = client.decide(
        DecisionsCall(
            model=tier.model,
            state=jev_triage.build_state(build_triage_text(messages, config.triage_max_chars)),
            questions=questions,
            timeout_s=tier.timeout_s,
        )
    )
    return {
        tier_name: jev_triage.interpret_answers(call_result),
        f"{tier_name}:direct": jev_triage.interpret_direct_tier(call_result),
    }


def score_row(
    config: HarnessConfig,
    case: dict,
    candidate: str,
    outcome: TriageOutcome,
    measured_routes: dict[str, dict] | None = None,
) -> dict:
    routed_tier, route_reason = choose_route(config, None, outcome)
    expected = case["expected_route"]
    decision = outcome.decision
    call = outcome.call
    measured_fields: dict = {}
    if measured_routes is not None:
        measured_fields = score_against_measured(routed_tier, measured_routes.get(case["id"]))
    return {
        **measured_fields,
        "case_id": case["id"],
        "category": case["category"],
        "candidate": candidate,
        "expected": expected,
        "status": outcome.status,
        "label": decision.label if decision else None,
        "confidence": round(decision.confidence, 3) if decision else None,
        "reason": decision.reason if decision else (call.error_message if call else None),
        "routed": routed_tier,
        "route_reason": route_reason,
        "label_correct": bool(decision and decision.label == expected),
        "routed_correct": routed_tier == expected,
        "under_routed": TIER_STRENGTH[routed_tier] < TIER_STRENGTH[expected],
        "over_routed": TIER_STRENGTH[routed_tier] > TIER_STRENGTH[expected],
        "latency_ms": call.latency_ms if call else None,
        "cost_usd": call.cost_usd if call else None,
    }


def summarise(rows: list[dict], candidate: str) -> dict:
    candidate_rows = [row for row in rows if row["candidate"] == candidate]
    succeeded = [row for row in candidate_rows if row["status"] == "ok"]
    latencies = [row["latency_ms"] for row in succeeded if row["latency_ms"] is not None]
    total = len(candidate_rows)
    measured_summary: dict = {}
    if candidate_rows and "measured_correct" in candidate_rows[0]:
        scored = [row for row in candidate_rows if row["measured_correct"] is not None]
        measured_summary = {
            "measured_scored": len(scored),
            "measured_unscored": total - len(scored),
            "measured_provisional": sum(row["measured_status"] == "provisional" for row in scored),
            "measured_correct": sum(row["measured_correct"] for row in scored),
            "measured_under": sum(row["measured_under"] for row in scored),
            "measured_over": sum(row["measured_over"] for row in scored),
        }
    return {
        **measured_summary,
        "candidate": candidate,
        "cases": total,
        "triage_failed": total - len(succeeded),
        "label_correct": sum(row["label_correct"] for row in candidate_rows),
        "routed_correct": sum(row["routed_correct"] for row in candidate_rows),
        "under_routed": sum(row["under_routed"] for row in candidate_rows),
        "over_routed": sum(row["over_routed"] for row in candidate_rows),
        "median_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
        "total_cost_usd": round(sum(row["cost_usd"] or 0.0 for row in candidate_rows), 6),
    }


def print_table(headers: list[str], table_rows: list[list]) -> None:
    widths = [max(len(str(value)) for value in column) for column in zip(headers, *table_rows)]
    for line in [headers, ["-" * width for width in widths], *table_rows]:
        print("  ".join(str(value).ljust(width) for value, width in zip(line, widths)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare triage tiers on the golden set")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN_PATH))
    parser.add_argument("--tiers", nargs="+", help="triage tiers to compare (default: all)")
    parser.add_argument("--out", help="write every row and the summaries to this JSON file")
    parser.add_argument(
        "--routes",
        help="measured routes JSON from scripts/run_eval.py; score against it as well as expected_route",
    )
    arguments = parser.parse_args(argv)

    load_env_file(".env")
    config = load_config(arguments.config)
    clients = build_clients(config)
    tier_names = arguments.tiers or config.triage_tier_names()
    unknown = [name for name in tier_names if name not in config.triage_tier_names()]
    if unknown:
        print(f"error: not triage tiers: {unknown}; choose from {config.triage_tier_names()}", file=sys.stderr)
        return 2

    cases = yaml.safe_load(Path(arguments.golden).read_text(encoding="utf-8"))["cases"]
    measured_routes = load_measured_routes(arguments.routes) if arguments.routes else None
    rows: list[dict] = []
    candidates: list[str] = []
    started_at = time.monotonic()
    for tier_name in tier_names:
        for case in cases:
            for candidate, outcome in triage_once(config, clients, tier_name, case["question"]).items():
                if candidate not in candidates:
                    candidates.append(candidate)
                rows.append(score_row(config, case, candidate, outcome, measured_routes))

    print(f"\nPer-case routing ({len(cases)} cases, threshold {config.confidence_threshold})\n")
    by_case = {(row["case_id"], row["candidate"]): row for row in rows}
    case_table = []
    for case in cases:
        line = [case["id"], case["expected_route"]]
        if measured_routes is not None:
            measured = measured_routes.get(case["id"])
            if measured is None:
                line.append("absent")
            elif measured["status"] in SCORABLE_ROUTE_STATUSES:
                suffix = " (provisional)" if measured["status"] == "provisional" else ""
                line.append("/".join(measured["acceptable"]) + suffix)
            else:
                line.append(measured["status"])
        for candidate in candidates:
            row = by_case[(case["id"], candidate)]
            if row["status"] != "ok":
                cell = f"{row['status']} -> {row['routed']}"
            else:
                cell = f"{row['label']} {row['confidence']:.2f} -> {row['routed']}"
            marks = "" if row["routed_correct"] else "  x"
            if row.get("measured_correct") is False:
                marks += "  m"
            line.append(cell + marks)
        case_table.append(line)
    route_headers = ["case", "expected"] + (["measured"] if measured_routes is not None else [])
    print_table([*route_headers, *candidates], case_table)
    if measured_routes is not None:
        print("\nx = differs from the hand-written expected_route, m = differs from the measured route")

    summaries = [summarise(rows, candidate) for candidate in candidates]
    print("\nSummary against the hand-written expected_route (routed = the tier the router picks after the low-confidence rule)\n")
    print_table(
        ["candidate", "failed", "label ok", "routed ok", "under", "over", "median ms", "cost usd"],
        [
            [
                summary["candidate"],
                summary["triage_failed"],
                f"{summary['label_correct']}/{summary['cases']}",
                f"{summary['routed_correct']}/{summary['cases']}",
                summary["under_routed"],
                summary["over_routed"],
                summary["median_latency_ms"],
                f"{summary['total_cost_usd']:.6f}",
            ]
            for summary in summaries
        ],
    )
    if measured_routes is not None:
        print(
            f"\nAgainst measured routes ({arguments.routes}): the cheapest tier that answered correctly. "
            "Under = routed to a weaker tier than was needed, over = stronger than needed.\n"
        )
        print_table(
            ["candidate", "scored", "unscored", "provisional", "routed ok", "under", "over"],
            [
                [
                    summary["candidate"],
                    summary["measured_scored"],
                    summary["measured_unscored"],
                    summary["measured_provisional"],
                    f"{summary['measured_correct']}/{summary['measured_scored']}",
                    summary["measured_under"],
                    summary["measured_over"],
                ]
                for summary in summaries
            ],
        )
        print(
            "\nUnscored cases have no measured route yet (no tier answered correctly, or a tier is "
            "missing); provisional ones may still move to a cheaper tier."
        )
    print(f"\nelapsed {time.monotonic() - started_at:.1f}s")

    if arguments.out:
        out_path = Path(arguments.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summaries": summaries, "rows": rows}, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
