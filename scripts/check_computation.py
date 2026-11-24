"""Measure the computation check on its own: no answer is written, nothing is executed.

For every case of the question files that carries a `computable` label (eval/README.md),
ask the check tier whether a short program can compute the answer and compare the verdict
with the label.

    uv run python scripts/check_computation.py
    uv run python scripts/check_computation.py --tier local --out data/eval/check_local.json

"Positive" is a verdict of yes: the question would be sent to execute mode. A yes below
the confidence threshold and a failed call are not a yes. Precision is the share of the
questions sent that are labelled computable; recall is the share of the computable
questions that were sent. The result file keeps one record per case and the prompt text
the run used, so every figure can be recomputed from it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from inference_arbiter import computation_check
from inference_arbiter.clients import build_clients
from inference_arbiter.config import DEFAULT_CONFIG_PATH, HarnessConfig, load_config, load_env_file
from inference_arbiter.schemas import Message

DEFAULT_QUESTION_PATHS = (Path("eval/golden_questions.yaml"), Path("eval/hard_questions.yaml"))
DEFAULT_EVAL_DIR = Path("data/eval")
LABEL_FIELD = "computable"
VERDICT_YES = "yes"


def load_labelled_cases(paths: list[Path]) -> list[dict[str, Any]]:
    """Every case with a `computable` label, tagged with the file it came from."""
    labelled: list[dict[str, Any]] = []
    for path in paths:
        for case in yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]:
            if isinstance(case.get(LABEL_FIELD), bool):
                labelled.append({**case, "question_file": str(path)})
    return labelled


def check_case(config: HarnessConfig, clients: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    check_tier = config.tiers[config.auto.computation_check_tier]
    outcome = computation_check.run_computation_check(
        config, clients[check_tier.provider], [Message(role="user", content=case["question"])]
    )
    decision = outcome.decision
    call = outcome.call
    return {
        "case_id": case["id"],
        "question_file": case["question_file"],
        "category": case["category"],
        "expected_computable": case[LABEL_FIELD],
        "status": outcome.status,
        "computable": decision.computable if decision else None,
        "confidence": decision.confidence if decision else None,
        "reason": decision.reason if decision else None,
        "verdict": computation_check.verdict_of(outcome, config.auto.computation_check_threshold),
        "latency_ms": call.latency_ms if call else None,
        "load_duration_ms": call.ollama_load_duration_ms if call else None,
        "error_type": call.error_type if call else None,
        "cost_usd": (call.cost_usd or 0.0) if call else 0.0,
    }


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    def cases_where(expected: bool, said_yes: bool) -> list[str]:
        return [
            record["case_id"]
            for record in records
            if record["expected_computable"] is expected and (record["verdict"] == VERDICT_YES) is said_yes
        ]

    true_positive = cases_where(True, True)
    false_positive = cases_where(False, True)
    false_negative = cases_where(True, False)
    true_negative = cases_where(False, False)
    sent = len(true_positive) + len(false_positive)
    computable = len(true_positive) + len(false_negative)
    latencies = [record["latency_ms"] for record in records if record["latency_ms"] is not None]
    verdicts: dict[str, int] = {}
    for record in records:
        verdicts[record["verdict"]] = verdicts.get(record["verdict"], 0) + 1
    false_positive_by_category: dict[str, int] = {}
    for record in records:
        if record["case_id"] in false_positive:
            category = record["category"]
            false_positive_by_category[category] = false_positive_by_category.get(category, 0) + 1
    return {
        "cases": len(records),
        "true_positive": len(true_positive),
        "false_positive": len(false_positive),
        "false_negative": len(false_negative),
        "true_negative": len(true_negative),
        "precision": len(true_positive) / sent if sent else None,
        "recall": len(true_positive) / computable if computable else None,
        "false_positive_cases": false_positive,
        "false_negative_cases": false_negative,
        "false_positive_by_category": false_positive_by_category,
        "verdicts": verdicts,
        "latency_median_ms": statistics.median(latencies) if latencies else None,
        "latency_max_ms": max(latencies) if latencies else None,
    }


def _ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_summary(summary: dict[str, Any], tier_name: str, model: str, threshold: float) -> None:
    print(f"\nComputation check on {tier_name} ({model}), threshold {threshold}, {summary['cases']} cases\n")
    print(f"  sent and computable (TP)         {summary['true_positive']}")
    print(f"  sent, not computable (FP)        {summary['false_positive']}  {','.join(summary['false_positive_cases'])}")
    print(f"  computable, not sent (FN)        {summary['false_negative']}  {','.join(summary['false_negative_cases'])}")
    print(f"  not computable, not sent (TN)    {summary['true_negative']}")
    print(f"  precision {_ratio(summary['precision'])}, recall {_ratio(summary['recall'])}")
    print(f"  verdicts {summary['verdicts']}")
    median = summary["latency_median_ms"]
    print(f"  check latency: median {'n/a' if median is None else f'{median:.0f} ms'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the computation check against the case labels")
    parser.add_argument("--questions", nargs="+", default=[str(path) for path in DEFAULT_QUESTION_PATHS],
                        help="question files whose cases carry a `computable` label")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--tier", help="tier to ask (default: config auto.computation_check_tier)")
    parser.add_argument("--threshold", type=float,
                        help="confidence a yes needs (default: config auto.computation_check_threshold)")
    parser.add_argument("--out", help="result JSON path (default: data/eval/check_<utc timestamp>.json)")
    arguments = parser.parse_args(argv)

    load_env_file(".env")
    config = load_config(arguments.config)
    if arguments.tier is not None:
        tier = config.tiers.get(arguments.tier)
        if tier is None or tier.role not in ("triage", "answer"):
            print(f"error: '{arguments.tier}' is not a triage tier or an answer tier", file=sys.stderr)
            return 2
        if config.providers[tier.provider].kind == "openrouter_decisions":
            print(f"error: '{arguments.tier}' uses a decisions provider; the check needs a chat model", file=sys.stderr)
            return 2
        config.auto.computation_check_tier = arguments.tier
    if arguments.threshold is not None:
        config.auto.computation_check_threshold = arguments.threshold
    tier_name = config.auto.computation_check_tier
    model = config.tiers[tier_name].model

    cases = load_labelled_cases([Path(path) for path in arguments.questions])
    if not cases:
        print("error: no case carries a `computable` label", file=sys.stderr)
        return 2

    clients = build_clients(config)
    started_utc = datetime.now(timezone.utc)
    records: list[dict[str, Any]] = []
    for case in cases:
        record = check_case(config, clients, case)
        records.append(record)
        latency = "-" if record["latency_ms"] is None else f"{record['latency_ms']:.0f}ms"
        print(
            f"  {record['case_id']:<6} expected {'yes' if record['expected_computable'] else 'no':<3} "
            f"verdict {record['verdict']:<14} {latency:>8}  {record['reason'] or record['error_type'] or ''}"[:160],
            file=sys.stderr,
            flush=True,
        )

    summary = summarise(records)
    out_path = Path(arguments.out) if arguments.out else (
        DEFAULT_EVAL_DIR / f"check_{started_utc.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = computation_check.CHECK_SYSTEM_PROMPT
    out_path.write_text(
        json.dumps(
            {
                "started_utc": started_utc.isoformat(timespec="seconds"),
                "question_files": arguments.questions,
                "config_sha256": config.sha256,
                "check_tier": tier_name,
                "check_model": model,
                "threshold": config.auto.computation_check_threshold,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt": prompt,
                "summary": summary,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print_summary(summary, tier_name, model, config.auto.computation_check_threshold)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
