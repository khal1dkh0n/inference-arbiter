#!/usr/bin/env python3
"""Cost, latency, routing, chain and open-book report over the harness telemetry database.

Usage: python scripts/report.py --db path/to/telemetry.db [--since ISO_TS] [--top N]
Standard library only; opens the database read-only.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB_PATH = "data/telemetry.db"
DEFAULT_TOP_N = 10
EARLIEST_TS = ""
TRIAGE_FAILURE_STATUSES = ("timeout", "error", "invalid")
NONE_LABEL = "(none)"


def open_read_only(db_path: str | Path) -> sqlite3.Connection:
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile; fraction is 0.5 for p50, 0.95 for p95."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def tier_model_breakdown(conn: sqlite3.Connection, since: str = EARLIEST_TS) -> list[dict]:
    """Run count, total cost and p50/p95 total latency per (final_tier, final_model)."""
    rows = conn.execute(
        """
        SELECT final_tier, final_model, total_latency_ms, total_cost_usd
        FROM runs
        WHERE ts_utc >= ?
        """,
        (since,),
    ).fetchall()
    groups: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["final_tier"] or NONE_LABEL, row["final_model"] or NONE_LABEL)
        group = groups.setdefault(key, {"latencies": [], "cost": 0.0, "unpriced": 0})
        group["latencies"].append(row["total_latency_ms"])
        if row["total_cost_usd"] is None:
            group["unpriced"] += 1
        else:
            group["cost"] += row["total_cost_usd"]
    breakdown = []
    for (tier, model), group in sorted(groups.items()):
        breakdown.append(
            {
                "final_tier": tier,
                "final_model": model,
                "runs": len(group["latencies"]),
                "total_cost_usd": group["cost"],
                "runs_without_cost": group["unpriced"],
                "p50_latency_ms": percentile(group["latencies"], 0.5),
                "p95_latency_ms": percentile(group["latencies"], 0.95),
            }
        )
    return breakdown


def route_reason_counts(conn: sqlite3.Connection, since: str = EARLIEST_TS) -> list[dict]:
    rows = conn.execute(
        """
        SELECT route_reason, COUNT(*) AS runs
        FROM runs
        WHERE ts_utc >= ?
        GROUP BY route_reason
        ORDER BY runs DESC, route_reason
        """,
        (since,),
    ).fetchall()
    return [dict(row) for row in rows]


def triage_status_counts(conn: sqlite3.Connection, since: str = EARLIEST_TS) -> list[dict]:
    rows = conn.execute(
        """
        SELECT triage_status, COUNT(*) AS runs
        FROM runs
        WHERE ts_utc >= ?
        GROUP BY triage_status
        ORDER BY runs DESC, triage_status
        """,
        (since,),
    ).fetchall()
    return [dict(row) for row in rows]


def failure_rates(conn: sqlite3.Connection, since: str = EARLIEST_TS) -> dict:
    """Fallback rate, triage failure rate and run error rate over all runs."""
    placeholders = ", ".join("?" for _ in TRIAGE_FAILURE_STATUSES)
    row = conn.execute(
        f"""
        SELECT
            COUNT(*) AS runs,
            COALESCE(SUM(fallback_used), 0) AS fallback_runs,
            COALESCE(SUM(triage_status IN ({placeholders})), 0) AS triage_failed_runs,
            COALESCE(SUM(status = 'error'), 0) AS error_runs
        FROM runs
        WHERE ts_utc >= ?
        """,
        (*TRIAGE_FAILURE_STATUSES, since),
    ).fetchone()
    rates = dict(row)
    total_runs = rates["runs"]
    for count_key, rate_key in (
        ("fallback_runs", "fallback_rate"),
        ("triage_failed_runs", "triage_failure_rate"),
        ("error_runs", "error_rate"),
    ):
        if total_runs == 0:
            rates[rate_key] = None
        else:
            rates[rate_key] = rates[count_key] / total_runs
    return rates


def slowest_ollama_loads(
    conn: sqlite3.Connection, limit: int = DEFAULT_TOP_N, since: str = EARLIEST_TS
) -> list[dict]:
    """Ollama calls with the highest load_duration: model-reload thrashing suspects."""
    rows = conn.execute(
        """
        SELECT runs.ts_utc, calls.trace_id, calls.seq, calls.role, calls.tier,
               calls.model, calls.status, calls.latency_ms,
               calls.ollama_load_duration_ms, calls.ollama_total_duration_ms
        FROM calls
        JOIN runs ON runs.trace_id = calls.trace_id
        WHERE calls.ollama_load_duration_ms IS NOT NULL AND runs.ts_utc >= ?
        ORDER BY calls.ollama_load_duration_ms DESC, calls.id
        LIMIT ?
        """,
        (since, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def has_chain_columns(conn: sqlite3.Connection) -> bool:
    """False for a version 1 database that the harness has not opened (and migrated) yet."""
    run_columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
    return "mode" in run_columns


def mode_breakdown(conn: sqlite3.Connection, since: str = EARLIEST_TS) -> list[dict]:
    """Run count, error count, total cost and p50/p95 total latency per mode."""
    rows = conn.execute(
        """
        SELECT mode, status, total_latency_ms, total_cost_usd
        FROM runs
        WHERE ts_utc >= ?
        """,
        (since,),
    ).fetchall()
    groups: dict[str, dict] = {}
    for row in rows:
        group = groups.setdefault(row["mode"], {"latencies": [], "cost": 0.0, "errors": 0})
        group["latencies"].append(row["total_latency_ms"])
        group["cost"] += row["total_cost_usd"] or 0.0
        if row["status"] == "error":
            group["errors"] += 1
    breakdown = []
    for mode, group in sorted(groups.items()):
        run_count = len(group["latencies"])
        breakdown.append(
            {
                "mode": mode,
                "runs": run_count,
                "error_runs": group["errors"],
                "total_cost_usd": group["cost"],
                "mean_cost_usd": group["cost"] / run_count,
                "p50_latency_ms": percentile(group["latencies"], 0.5),
                "p95_latency_ms": percentile(group["latencies"], 0.95),
            }
        )
    return breakdown


def chain_verdict_counts(conn: sqlite3.Connection, since: str = EARLIEST_TS) -> list[dict]:
    rows = conn.execute(
        """
        SELECT verification_status, COALESCE(verifier_tier, ?) AS verifier_tier, COUNT(*) AS runs,
               COALESCE(SUM(repaired), 0) AS repaired_runs
        FROM runs
        WHERE mode = 'chain' AND ts_utc >= ?
        GROUP BY verification_status, verifier_tier
        ORDER BY runs DESC, verification_status, verifier_tier
        """,
        (NONE_LABEL, since),
    ).fetchall()
    return [dict(row) for row in rows]


def chain_rates(conn: sqlite3.Connection, since: str = EARLIEST_TS) -> dict:
    """Repair, repair-failure and schema-escalation rates over chain runs."""
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS runs,
            COALESCE(SUM(repaired), 0) AS repaired_runs,
            COALESCE(SUM(schema_escalated), 0) AS schema_escalated_runs,
            COALESCE(SUM(status = 'error'), 0) AS error_runs
        FROM runs
        WHERE mode = 'chain' AND ts_utc >= ?
        """,
        (since,),
    ).fetchone()
    rates = dict(row)
    attempted_row = conn.execute(
        """
        SELECT COUNT(DISTINCT calls.trace_id) AS repair_attempted_runs
        FROM calls
        JOIN runs ON runs.trace_id = calls.trace_id
        WHERE calls.role = 'repair' AND runs.mode = 'chain' AND runs.ts_utc >= ?
        """,
        (since,),
    ).fetchone()
    rates["repair_attempted_runs"] = attempted_row["repair_attempted_runs"]
    rates["repair_failed_runs"] = rates["repair_attempted_runs"] - rates["repaired_runs"]
    total_runs = rates["runs"]
    for count_key, rate_key in (
        ("repaired_runs", "repair_rate"),
        ("repair_failed_runs", "repair_failure_rate"),
        ("schema_escalated_runs", "schema_escalation_rate"),
        ("error_runs", "error_rate"),
    ):
        if total_runs == 0:
            rates[rate_key] = None
        else:
            rates[rate_key] = rates[count_key] / total_runs
    return rates


def render_chain_section(conn: sqlite3.Connection, since: str = EARLIEST_TS) -> list[str]:
    lines = ["Chain mode"]
    if not has_chain_columns(conn):
        lines.append("  (this database predates chain mode; it is migrated the next time the harness opens it)")
        return lines

    lines.append("  Cost and latency by mode")
    lines.append(
        format_table(
            ["mode", "runs", "errors", "cost_usd", "mean_cost_usd", "p50_ms", "p95_ms"],
            [
                [
                    entry["mode"],
                    entry["runs"],
                    entry["error_runs"],
                    f"{entry['total_cost_usd']:.6f}",
                    f"{entry['mean_cost_usd']:.6f}",
                    entry["p50_latency_ms"],
                    entry["p95_latency_ms"],
                ]
                for entry in mode_breakdown(conn, since)
            ],
        )
    )

    rates = chain_rates(conn, since)
    chain_runs = rates["runs"]
    lines.append("  Chain runs by verdict and verifier")
    lines.append(
        format_table(
            ["verification_status", "verifier_tier", "runs", "repaired"],
            [
                [entry["verification_status"], entry["verifier_tier"], entry["runs"], entry["repaired_runs"]]
                for entry in chain_verdict_counts(conn, since)
            ],
        )
    )
    lines.append(
        "  repair rate:            "
        + _format_rate(rates["repaired_runs"], rates["repair_rate"], chain_runs)
    )
    lines.append(
        "  repair failure rate:    "
        + _format_rate(rates["repair_failed_runs"], rates["repair_failure_rate"], chain_runs)
        + "  [repair attempted, draft returned]"
    )
    lines.append(
        "  schema-escalation rate: "
        + _format_rate(rates["schema_escalated_runs"], rates["schema_escalation_rate"], chain_runs)
    )
    lines.append(
        "  chain run error rate:   "
        + _format_rate(rates["error_runs"], rates["error_rate"], chain_runs)
    )
    return lines


def has_openbook_columns(conn: sqlite3.Connection) -> bool:
    """False for a version 1 or 2 database that the harness has not opened (and migrated) yet."""
    run_columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
    return "grounding_status" in run_columns


def openbook_status_counts(conn: sqlite3.Connection, since: str = EARLIEST_TS) -> list[dict]:
    """Openbook runs per grounding status and verifier, with escalations and abstentions."""
    rows = conn.execute(
        """
        SELECT grounding_status, COALESCE(grounding_verifier_tier, ?) AS grounding_verifier_tier,
               COUNT(*) AS runs,
               COALESCE(SUM(repaired), 0) AS escalated_runs,
               COALESCE(SUM(abstained), 0) AS abstained_runs
        FROM runs
        WHERE mode = 'openbook' AND ts_utc >= ?
        GROUP BY grounding_status, grounding_verifier_tier
        ORDER BY runs DESC, grounding_status, grounding_verifier_tier
        """,
        (NONE_LABEL, since),
    ).fetchall()
    return [dict(row) for row in rows]


def openbook_retrieval_counts(conn: sqlite3.Connection, since: str = EARLIEST_TS) -> list[dict]:
    rows = conn.execute(
        """
        SELECT COALESCE(retrieval_method, ?) AS retrieval_method, COUNT(*) AS runs,
               AVG(sources_count) AS mean_sources, AVG(passages_retrieved) AS mean_passages
        FROM runs
        WHERE mode = 'openbook' AND ts_utc >= ?
        GROUP BY retrieval_method
        ORDER BY runs DESC, retrieval_method
        """,
        (NONE_LABEL, since),
    ).fetchall()
    return [dict(row) for row in rows]


def openbook_rates(conn: sqlite3.Connection, since: str = EARLIEST_TS) -> dict:
    """Escalation, abstention and error rates over openbook runs, and the claim support ratio."""
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS runs,
            COALESCE(SUM(repaired), 0) AS escalated_runs,
            COALESCE(SUM(abstained), 0) AS abstained_runs,
            COALESCE(SUM(status = 'error'), 0) AS error_runs,
            COALESCE(SUM(claims_total), 0) AS claims_total,
            COALESCE(SUM(claims_supported), 0) AS claims_supported
        FROM runs
        WHERE mode = 'openbook' AND ts_utc >= ?
        """,
        (since,),
    ).fetchone()
    rates = dict(row)
    attempted_row = conn.execute(
        """
        SELECT COUNT(DISTINCT calls.trace_id) AS escalation_attempted_runs
        FROM calls
        JOIN runs ON runs.trace_id = calls.trace_id
        WHERE calls.role = 'escalate' AND runs.mode = 'openbook' AND runs.ts_utc >= ?
        """,
        (since,),
    ).fetchone()
    rates["escalation_attempted_runs"] = attempted_row["escalation_attempted_runs"]
    rates["escalation_failed_runs"] = rates["escalation_attempted_runs"] - rates["escalated_runs"]
    total_runs = rates["runs"]
    for count_key, rate_key in (
        ("escalated_runs", "escalation_rate"),
        ("escalation_failed_runs", "escalation_failure_rate"),
        ("abstained_runs", "abstention_rate"),
        ("error_runs", "error_rate"),
    ):
        if total_runs == 0:
            rates[rate_key] = None
        else:
            rates[rate_key] = rates[count_key] / total_runs
    if rates["claims_total"] == 0:
        rates["claim_support_rate"] = None
    else:
        rates["claim_support_rate"] = rates["claims_supported"] / rates["claims_total"]
    return rates


def render_openbook_section(conn: sqlite3.Connection, since: str = EARLIEST_TS) -> list[str]:
    lines = ["Open-book mode"]
    if not has_openbook_columns(conn):
        lines.append(
            "  (this database predates open-book mode; it is migrated the next time the harness opens it)"
        )
        return lines

    rates = openbook_rates(conn, since)
    openbook_runs = rates["runs"]
    lines.append("  Open-book runs by grounding status and verifier")
    lines.append(
        format_table(
            ["grounding_status", "verifier_tier", "runs", "escalated", "abstained"],
            [
                [
                    entry["grounding_status"],
                    entry["grounding_verifier_tier"],
                    entry["runs"],
                    entry["escalated_runs"],
                    entry["abstained_runs"],
                ]
                for entry in openbook_status_counts(conn, since)
            ],
        )
    )
    lines.append("  Retrieval")
    lines.append(
        format_table(
            ["retrieval_method", "runs", "mean_sources", "mean_passages"],
            [
                [entry["retrieval_method"], entry["runs"], entry["mean_sources"], entry["mean_passages"]]
                for entry in openbook_retrieval_counts(conn, since)
            ],
        )
    )
    lines.append(
        "  escalation rate:         "
        + _format_rate(rates["escalated_runs"], rates["escalation_rate"], openbook_runs)
    )
    lines.append(
        "  escalation failure rate: "
        + _format_rate(rates["escalation_failed_runs"], rates["escalation_failure_rate"], openbook_runs)
        + "  [escalation attempted, first draft returned]"
    )
    lines.append(
        "  abstention rate:         "
        + _format_rate(rates["abstained_runs"], rates["abstention_rate"], openbook_runs)
    )
    lines.append(
        "  open-book error rate:    "
        + _format_rate(rates["error_runs"], rates["error_rate"], openbook_runs)
    )
    lines.append(
        "  claims supported:        "
        + _format_rate(rates["claims_supported"], rates["claim_support_rate"], rates["claims_total"])
        + "  [claims of returned answers that passed the checks]"
    )
    return lines


def _format_cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def format_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "  (no rows)"
    text_rows = [[_format_cell(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for text_row in text_rows:
        for index, cell in enumerate(text_row):
            widths[index] = max(widths[index], len(cell))
    lines = ["  " + "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))]
    lines.append("  " + "  ".join("-" * width for width in widths))
    for text_row in text_rows:
        lines.append("  " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(text_row)))
    return "\n".join(lines)


def _format_rate(count: int, rate: float | None, total: int) -> str:
    if rate is None:
        return "n/a (no runs)"
    return f"{rate * 100:.1f}% ({count}/{total})"


def render_report(conn: sqlite3.Connection, since: str = EARLIEST_TS, top_n: int = DEFAULT_TOP_N) -> str:
    sections = []

    breakdown = tier_model_breakdown(conn, since)
    sections.append("Cost and latency by final tier and model")
    sections.append(
        format_table(
            ["final_tier", "final_model", "runs", "cost_usd", "no_cost", "p50_ms", "p95_ms"],
            [
                [
                    entry["final_tier"],
                    entry["final_model"],
                    entry["runs"],
                    f"{entry['total_cost_usd']:.6f}",
                    entry["runs_without_cost"],
                    entry["p50_latency_ms"],
                    entry["p95_latency_ms"],
                ]
                for entry in breakdown
            ],
        )
    )
    total_runs = sum(entry["runs"] for entry in breakdown)
    total_cost = sum(entry["total_cost_usd"] for entry in breakdown)
    sections.append(f"  total: {total_runs} runs, {total_cost:.6f} USD")

    sections.append("")
    sections.append("Runs by route_reason")
    sections.append(
        format_table(
            ["route_reason", "runs"],
            [[entry["route_reason"], entry["runs"]] for entry in route_reason_counts(conn, since)],
        )
    )

    rates = failure_rates(conn, since)
    sections.append("")
    sections.append("Fallback and triage failure")
    sections.append(
        "  fallback rate:       "
        + _format_rate(rates["fallback_runs"], rates["fallback_rate"], rates["runs"])
    )
    sections.append(
        "  triage failure rate: "
        + _format_rate(rates["triage_failed_runs"], rates["triage_failure_rate"], rates["runs"])
        + f"  [triage_status in {', '.join(TRIAGE_FAILURE_STATUSES)}]"
    )
    sections.append(
        "  run error rate:      "
        + _format_rate(rates["error_runs"], rates["error_rate"], rates["runs"])
    )
    sections.append(
        format_table(
            ["triage_status", "runs"],
            [[entry["triage_status"], entry["runs"]] for entry in triage_status_counts(conn, since)],
        )
    )

    sections.append("")
    sections.extend(render_chain_section(conn, since))

    sections.append("")
    sections.extend(render_openbook_section(conn, since))

    sections.append("")
    sections.append(f"Ollama calls with the highest load duration (top {top_n})")
    sections.append(
        format_table(
            ["ts_utc", "trace_id", "seq", "role", "tier", "model", "status", "load_ms", "total_ms", "latency_ms"],
            [
                [
                    entry["ts_utc"],
                    entry["trace_id"],
                    entry["seq"],
                    entry["role"],
                    entry["tier"],
                    entry["model"],
                    entry["status"],
                    entry["ollama_load_duration_ms"],
                    entry["ollama_total_duration_ms"],
                    entry["latency_ms"],
                ]
                for entry in slowest_ollama_loads(conn, top_n, since)
            ],
        )
    )
    return "\n".join(sections)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report over the inference arbiter telemetry database.")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help=f"SQLite database path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--since", default=EARLIEST_TS, help="only include runs with ts_utc >= this ISO 8601 timestamp")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N, help="number of slowest Ollama loads to list")
    args = parser.parse_args(argv)

    if not Path(args.db).is_file():
        print(f"error: database not found: {args.db}", file=sys.stderr)
        return 1
    try:
        conn = open_read_only(args.db)
        try:
            print(f"Telemetry report: {args.db}")
            if args.since:
                print(f"Runs since: {args.since}")
            print()
            print(render_report(conn, args.since, args.top))
        finally:
            conn.close()
    except sqlite3.Error as error:
        print(f"error: could not read {args.db}: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
