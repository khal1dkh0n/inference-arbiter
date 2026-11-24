import importlib.util
import sqlite3
from pathlib import Path

import pytest

from inference_arbiter.telemetry import SCHEMA_VERSION, CallRecord, RunRecord, Telemetry

REPORT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "report.py"


def load_report_module():
    spec = importlib.util.spec_from_file_location("harness_report", REPORT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_call(**overrides) -> CallRecord:
    values = dict(
        seq=0,
        role="answer",
        tier="local",
        provider="ollama",
        model="qwen3:14b",
        effort=None,
        timeout_s=30.0,
        status="ok",
        error_type=None,
        latency_ms=1200.0,
        prompt_tokens=100,
        completion_tokens=50,
        cached_tokens=None,
        cost_usd=0.0,
        cost_source="local_zero",
        ollama_load_duration_ms=40.0,
        ollama_total_duration_ms=1150.0,
        ollama_prompt_eval_ms=200.0,
        ollama_eval_ms=900.0,
    )
    values.update(overrides)
    return CallRecord(**values)


def make_run(trace_id: str, **overrides) -> RunRecord:
    values = dict(
        trace_id=trace_id,
        ts_utc="2026-11-21T10:00:00+00:00",
        prompt_sha256="a" * 64,
        prompt_text="What is 2 + 2?",
        config_sha256="b" * 64,
        triage_status="ok",
        triage_label="simple",
        triage_confidence=0.92,
        triage_reason="short factual question",
        route_reason="triage_simple",
        selected_tier="local",
        final_tier="local",
        final_model="qwen3:14b",
        effort=None,
        fallback_used=False,
        status="ok",
        error_type=None,
        total_latency_ms=1500.0,
        total_cost_usd=0.0,
    )
    values.update(overrides)
    return RunRecord(**values)


def read_all(db_path: Path, sql: str) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "nested" / "dir" / "telemetry.db"


def test_schema_creation(db_path: Path):
    telemetry = Telemetry(db_path)
    telemetry.close()

    assert db_path.is_file()
    conn = sqlite3.connect(db_path)
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        indexes = {
            (row[0], row[1])
            for row in conn.execute("SELECT name, tbl_name FROM sqlite_master WHERE type = 'index'")
        }
        assert {"runs", "calls", "schema_version"} <= tables
        assert ("idx_runs_ts_utc", "runs") in indexes
        assert ("idx_calls_trace_id", "calls") in indexes
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [(SCHEMA_VERSION,)]

        run_columns = {row[1]: row for row in conn.execute("PRAGMA table_info(runs)")}
        assert run_columns["trace_id"][5] == 1
        assert set(run_columns) == set(RunRecord.__dataclass_fields__)

        call_columns = {row[1]: row for row in conn.execute("PRAGMA table_info(calls)")}
        assert call_columns["id"][5] == 1
        assert set(call_columns) == {"id", "trace_id"} | set(CallRecord.__dataclass_fields__)
        foreign_keys = conn.execute("PRAGMA foreign_key_list(calls)").fetchall()
        assert [(row[2], row[3], row[4]) for row in foreign_keys] == [("runs", "trace_id", "trace_id")]
    finally:
        conn.close()


def test_reopen_keeps_single_schema_version_row_and_data(db_path: Path):
    first = Telemetry(db_path)
    first.record(make_run("t-1"), [make_call()])
    first.close()

    second = Telemetry(db_path)
    second.record(make_run("t-2"), [make_call()])
    second.close()

    assert len(read_all(db_path, "SELECT * FROM schema_version")) == 1
    assert len(read_all(db_path, "SELECT * FROM runs")) == 2


def test_round_trip_triage_answer_fallback(db_path: Path):
    run = make_run(
        "trace-fallback",
        route_reason="fallback_after_timeout",
        selected_tier="mid",
        final_tier="frontier",
        final_model="vendor/frontier-model",
        effort="high",
        fallback_used=True,
        total_latency_ms=42000.5,
        total_cost_usd=0.0321,
    )
    calls = [
        make_call(seq=0, role="triage", latency_ms=300.0, ollama_load_duration_ms=5200.0),
        make_call(
            seq=1,
            role="answer",
            tier="mid",
            provider="openrouter",
            model="openai/gpt-mini",
            effort="medium",
            timeout_s=20.0,
            status="timeout",
            error_type="TimeoutError",
            latency_ms=20000.0,
            prompt_tokens=None,
            completion_tokens=None,
            cost_usd=None,
            cost_source=None,
            ollama_load_duration_ms=None,
            ollama_total_duration_ms=None,
            ollama_prompt_eval_ms=None,
            ollama_eval_ms=None,
        ),
        make_call(
            seq=2,
            role="answer",
            tier="frontier",
            provider="openrouter",
            model="vendor/frontier-model",
            effort="high",
            timeout_s=120.0,
            latency_ms=21700.5,
            prompt_tokens=812,
            completion_tokens=640,
            cached_tokens=256,
            cost_usd=0.0321,
            cost_source="provider_reported",
            ollama_load_duration_ms=None,
            ollama_total_duration_ms=None,
            ollama_prompt_eval_ms=None,
            ollama_eval_ms=None,
        ),
    ]
    telemetry = Telemetry(db_path)
    telemetry.record(run, calls)
    telemetry.close()

    run_rows = read_all(db_path, "SELECT * FROM runs")
    assert len(run_rows) == 1
    stored_run = dict(run_rows[0])
    assert stored_run.pop("fallback_used") == 1
    expected_run = dict(run.__dict__)
    expected_run.pop("fallback_used")
    assert stored_run == expected_run

    call_rows = read_all(db_path, "SELECT * FROM calls ORDER BY seq")
    assert len(call_rows) == 3
    for stored, original in zip(call_rows, calls):
        stored_call = dict(stored)
        assert isinstance(stored_call.pop("id"), int)
        assert stored_call.pop("trace_id") == "trace-fallback"
        assert stored_call == original.__dict__
    assert [row["status"] for row in call_rows] == ["ok", "timeout", "ok"]
    assert [row["id"] for row in call_rows] == sorted(row["id"] for row in call_rows)


def test_none_handling(db_path: Path):
    run = make_run(
        "trace-none",
        prompt_text=None,
        triage_status="error",
        triage_label=None,
        triage_confidence=None,
        triage_reason=None,
        final_tier=None,
        final_model=None,
        effort=None,
        status="error",
        error_type="ConnectionError",
        total_cost_usd=None,
    )
    call = make_call(
        effort=None,
        timeout_s=None,
        status="error",
        error_type="ConnectionError",
        prompt_tokens=None,
        completion_tokens=None,
        cached_tokens=None,
        cost_usd=None,
        cost_source=None,
        ollama_load_duration_ms=None,
        ollama_total_duration_ms=None,
        ollama_prompt_eval_ms=None,
        ollama_eval_ms=None,
    )
    telemetry = Telemetry(db_path)
    telemetry.record(run, [call])
    telemetry.record(make_run("trace-no-calls"), [])
    telemetry.close()

    stored_run = dict(read_all(db_path, "SELECT * FROM runs WHERE trace_id = 'trace-none'")[0])
    for column in (
        "prompt_text", "triage_label", "triage_confidence", "triage_reason",
        "final_tier", "final_model", "effort", "total_cost_usd",
    ):
        assert stored_run[column] is None
    assert stored_run["fallback_used"] == 0

    stored_call = dict(read_all(db_path, "SELECT * FROM calls")[0])
    for column in (
        "effort", "timeout_s", "prompt_tokens", "completion_tokens", "cached_tokens",
        "cost_usd", "cost_source", "ollama_load_duration_ms", "ollama_total_duration_ms",
        "ollama_prompt_eval_ms", "ollama_eval_ms",
    ):
        assert stored_call[column] is None
    assert len(read_all(db_path, "SELECT * FROM runs WHERE trace_id = 'trace-no-calls'")) == 1


def test_record_does_not_raise_on_broken_path(tmp_path: Path, capsys):
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("plain file")
    telemetry = Telemetry(blocker / "sub" / "telemetry.db")
    telemetry.record(make_run("t-broken"), [make_call()])
    telemetry.close()

    warning_lines = capsys.readouterr().err.strip().splitlines()
    assert len(warning_lines) == 2
    assert all(line.startswith("telemetry warning:") for line in warning_lines)
    assert "t-broken" in warning_lines[1]


def test_record_does_not_raise_on_duplicate_trace_id(db_path: Path, capsys):
    telemetry = Telemetry(db_path)
    telemetry.record(make_run("dup"), [make_call(seq=0)])
    telemetry.record(make_run("dup", route_reason="second_write"), [make_call(seq=0), make_call(seq=1)])
    telemetry.record(make_run("after-dup"), [make_call()])
    telemetry.close()

    warning_lines = capsys.readouterr().err.strip().splitlines()
    assert len(warning_lines) == 1
    assert "trace_id=dup" in warning_lines[0]
    assert "IntegrityError" in warning_lines[0]

    runs = read_all(db_path, "SELECT trace_id, route_reason FROM runs ORDER BY trace_id")
    assert [(row["trace_id"], row["route_reason"]) for row in runs] == [
        ("after-dup", "triage_simple"),
        ("dup", "triage_simple"),
    ]
    assert len(read_all(db_path, "SELECT * FROM calls WHERE trace_id = 'dup'")) == 1


def test_failed_call_insert_rolls_back_run_row(db_path: Path, capsys):
    telemetry = Telemetry(db_path)
    telemetry.record(make_run("atomic"), [make_call(seq=0), make_call(seq=1, status=None)])
    telemetry.close()

    assert "trace_id=atomic" in capsys.readouterr().err
    assert read_all(db_path, "SELECT * FROM runs") == []
    assert read_all(db_path, "SELECT * FROM calls") == []


def test_record_does_not_raise_after_close_or_on_bad_input(db_path: Path, capsys):
    telemetry = Telemetry(db_path)
    telemetry.record(make_run("bad-calls"), None)
    telemetry.record(None, [])
    telemetry.close()
    telemetry.record(make_run("after-close"), [make_call()])
    telemetry.close()

    assert len(capsys.readouterr().err.strip().splitlines()) == 3
    assert read_all(db_path, "SELECT * FROM runs") == []


@pytest.fixture
def seeded_db(db_path: Path) -> Path:
    telemetry = Telemetry(db_path)
    local_latencies = [1000.0, 2000.0, 3000.0, 4000.0]
    for index, latency in enumerate(local_latencies):
        telemetry.record(
            make_run(
                f"local-{index}",
                ts_utc=f"2026-11-2{index}T10:00:00+00:00",
                total_latency_ms=latency,
            ),
            [
                make_call(seq=0, role="triage", ollama_load_duration_ms=10.0 * (index + 1)),
                make_call(seq=1, role="answer", ollama_load_duration_ms=1000.0 * (index + 1)),
            ],
        )
    telemetry.record(
        make_run(
            "frontier-0",
            ts_utc="2026-11-24T10:00:00+00:00",
            triage_status="timeout",
            triage_label=None,
            triage_confidence=None,
            route_reason="triage_failed_escalate",
            selected_tier="frontier",
            final_tier="frontier",
            final_model="vendor/frontier-model",
            total_latency_ms=9000.0,
            total_cost_usd=0.05,
        ),
        [
            make_call(seq=0, role="triage", status="timeout", ollama_load_duration_ms=None),
            make_call(
                seq=1, tier="frontier", provider="openrouter", model="vendor/frontier-model",
                cost_usd=0.05, cost_source="provider_reported", ollama_load_duration_ms=None,
            ),
        ],
    )
    telemetry.record(
        make_run(
            "frontier-1",
            ts_utc="2026-11-25T10:00:00+00:00",
            route_reason="fallback_after_timeout",
            selected_tier="mid",
            final_tier="frontier",
            final_model="vendor/frontier-model",
            fallback_used=True,
            total_latency_ms=11000.0,
            total_cost_usd=0.07,
        ),
        [make_call(seq=0, role="triage", ollama_load_duration_ms=7777.0)],
    )
    telemetry.record(
        make_run(
            "failed-0",
            ts_utc="2026-11-26T10:00:00+00:00",
            route_reason="fallback_after_timeout",
            selected_tier="mid",
            final_tier=None,
            final_model=None,
            fallback_used=True,
            status="error",
            error_type="AllTiersFailed",
            total_latency_ms=30000.0,
            total_cost_usd=None,
        ),
        [],
    )
    telemetry.close()
    return db_path


def test_report_tier_model_breakdown(seeded_db: Path):
    report = load_report_module()
    conn = report.open_read_only(seeded_db)
    try:
        breakdown = {
            (entry["final_tier"], entry["final_model"]): entry
            for entry in report.tier_model_breakdown(conn)
        }
    finally:
        conn.close()

    assert set(breakdown) == {
        ("local", "qwen3:14b"),
        ("frontier", "vendor/frontier-model"),
        (report.NONE_LABEL, report.NONE_LABEL),
    }
    local = breakdown[("local", "qwen3:14b")]
    assert local["runs"] == 4
    assert local["total_cost_usd"] == 0.0
    assert local["p50_latency_ms"] == 2000.0
    assert local["p95_latency_ms"] == 4000.0
    frontier = breakdown[("frontier", "vendor/frontier-model")]
    assert frontier["runs"] == 2
    assert frontier["total_cost_usd"] == pytest.approx(0.12)
    assert frontier["p50_latency_ms"] == 9000.0
    assert frontier["p95_latency_ms"] == 11000.0
    failed = breakdown[(report.NONE_LABEL, report.NONE_LABEL)]
    assert failed["runs"] == 1
    assert failed["runs_without_cost"] == 1


def test_report_route_reasons_and_rates(seeded_db: Path):
    report = load_report_module()
    conn = report.open_read_only(seeded_db)
    try:
        reasons = {entry["route_reason"]: entry["runs"] for entry in report.route_reason_counts(conn)}
        rates = report.failure_rates(conn)
        recent_rates = report.failure_rates(conn, since="2026-11-24T00:00:00+00:00")
    finally:
        conn.close()

    assert reasons == {
        "triage_simple": 4,
        "fallback_after_timeout": 2,
        "triage_failed_escalate": 1,
    }
    assert rates["runs"] == 7
    assert rates["fallback_rate"] == pytest.approx(2 / 7)
    assert rates["triage_failure_rate"] == pytest.approx(1 / 7)
    assert rates["error_rate"] == pytest.approx(1 / 7)
    assert recent_rates["runs"] == 3
    assert recent_rates["fallback_rate"] == pytest.approx(2 / 3)


def test_report_slowest_ollama_loads(seeded_db: Path):
    report = load_report_module()
    conn = report.open_read_only(seeded_db)
    try:
        slowest = report.slowest_ollama_loads(conn, limit=3)
    finally:
        conn.close()

    assert [entry["ollama_load_duration_ms"] for entry in slowest] == [7777.0, 4000.0, 3000.0]
    assert slowest[0]["trace_id"] == "frontier-1"
    assert slowest[0]["role"] == "triage"


def test_report_cli_output(seeded_db: Path, tmp_path: Path, capsys):
    report = load_report_module()
    assert report.main(["--db", str(seeded_db), "--top", "2"]) == 0
    output = capsys.readouterr().out
    assert "Cost and latency by final tier and model" in output
    assert "vendor/frontier-model" in output
    assert "fallback_after_timeout" in output
    assert "fallback rate:       28.6% (2/7)" in output
    assert "triage failure rate: 14.3% (1/7)" in output
    assert "7777.0" in output

    assert report.main(["--db", str(tmp_path / "missing.db")]) == 1
    assert "database not found" in capsys.readouterr().err


def test_report_on_empty_database(db_path: Path, capsys):
    Telemetry(db_path).close()
    report = load_report_module()
    assert report.main(["--db", str(db_path)]) == 0
    output = capsys.readouterr().out
    assert "(no rows)" in output
    assert "n/a (no runs)" in output


def test_percentile_nearest_rank():
    report = load_report_module()
    assert report.percentile([], 0.5) is None
    assert report.percentile([5.0], 0.95) == 5.0
    values = [float(number) for number in range(1, 101)]
    assert report.percentile(values, 0.5) == 50.0
    assert report.percentile(values, 0.95) == 95.0


# --- schema version 2: chain columns and migration from a version 1 database ---

V1_SCHEMA_STATEMENTS = (
    "CREATE TABLE schema_version (version INTEGER NOT NULL)",
    """
    CREATE TABLE runs (
        trace_id TEXT PRIMARY KEY,
        ts_utc TEXT NOT NULL,
        prompt_sha256 TEXT NOT NULL,
        prompt_text TEXT,
        config_sha256 TEXT NOT NULL,
        triage_status TEXT NOT NULL,
        triage_label TEXT,
        triage_confidence REAL,
        triage_reason TEXT,
        route_reason TEXT NOT NULL,
        selected_tier TEXT NOT NULL,
        final_tier TEXT,
        final_model TEXT,
        effort TEXT,
        fallback_used INTEGER NOT NULL,
        status TEXT NOT NULL,
        error_type TEXT,
        total_latency_ms REAL NOT NULL,
        total_cost_usd REAL
    )
    """,
    """
    CREATE TABLE calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trace_id TEXT NOT NULL REFERENCES runs(trace_id),
        seq INTEGER NOT NULL,
        role TEXT NOT NULL,
        tier TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        effort TEXT,
        timeout_s REAL,
        status TEXT NOT NULL,
        error_type TEXT,
        latency_ms REAL NOT NULL,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        cached_tokens INTEGER,
        cost_usd REAL,
        cost_source TEXT,
        ollama_load_duration_ms REAL,
        ollama_total_duration_ms REAL,
        ollama_prompt_eval_ms REAL,
        ollama_eval_ms REAL
    )
    """,
    "CREATE INDEX idx_runs_ts_utc ON runs(ts_utc)",
    "CREATE INDEX idx_calls_trace_id ON calls(trace_id)",
    "INSERT INTO schema_version (version) VALUES (1)",
    """
    INSERT INTO runs VALUES (
        'old-run', '2026-11-20T10:00:00+00:00', 'hash', 'an old question', 'cfg', 'ok', 'local',
        0.9, 'old reason', 'triage_label', 'local', 'local', 'qwen3:14b', NULL, 0, 'ok', NULL,
        1500.0, 0.0
    )
    """,
    """
    INSERT INTO calls (trace_id, seq, role, tier, provider, model, status, latency_ms)
    VALUES ('old-run', 1, 'answer', 'local', 'ollama', 'qwen3:14b', 'ok', 1400.0)
    """,
)


@pytest.fixture
def v1_db(tmp_path: Path) -> Path:
    v1_path = tmp_path / "v1.db"
    conn = sqlite3.connect(v1_path)
    try:
        with conn:
            for statement in V1_SCHEMA_STATEMENTS:
                conn.execute(statement)
    finally:
        conn.close()
    return v1_path


def test_schema_version_is_8():
    assert SCHEMA_VERSION == 8


def test_v1_database_is_migrated_in_place_and_old_rows_stay_readable(v1_db: Path):
    telemetry = Telemetry(v1_db)
    telemetry.record(
        make_run(
            "new-chain-run",
            mode="chain",
            verification_status="unsupported",
            verification_critique="wrong count",
            repaired=True,
            schema_escalated=True,
            verifier_tier="verify",
        ),
        [make_call(seq=1, role="draft"), make_call(seq=2, role="verify"), make_call(seq=3, role="repair")],
    )
    telemetry.close()

    assert [tuple(row) for row in read_all(v1_db, "SELECT version FROM schema_version")] == [(SCHEMA_VERSION,)]

    old_run = read_all(v1_db, "SELECT * FROM runs WHERE trace_id = 'old-run'")[0]
    assert old_run["prompt_text"] == "an old question"
    assert old_run["final_tier"] == "local"
    assert old_run["mode"] == "route"
    assert old_run["verification_status"] == "skipped"
    assert old_run["verification_critique"] is None
    assert old_run["repaired"] == 0
    assert old_run["schema_escalated"] == 0
    assert old_run["verifier_tier"] is None
    assert len(read_all(v1_db, "SELECT * FROM calls WHERE trace_id = 'old-run'")) == 1

    new_run = read_all(v1_db, "SELECT * FROM runs WHERE trace_id = 'new-chain-run'")[0]
    assert new_run["mode"] == "chain"
    assert new_run["verification_status"] == "unsupported"
    assert new_run["verification_critique"] == "wrong count"
    assert new_run["repaired"] == 1
    assert new_run["schema_escalated"] == 1
    assert new_run["verifier_tier"] == "verify"
    new_roles = read_all(v1_db, "SELECT role FROM calls WHERE trace_id = 'new-chain-run' ORDER BY seq")
    assert [row["role"] for row in new_roles] == ["draft", "verify", "repair"]


def test_migration_is_idempotent(v1_db: Path):
    Telemetry(v1_db).close()
    Telemetry(v1_db).close()
    reopened = Telemetry(v1_db)
    reopened.record(make_run("after-reopen"), [make_call()])
    reopened.close()

    assert [tuple(row) for row in read_all(v1_db, "SELECT version FROM schema_version")] == [(SCHEMA_VERSION,)]
    run_columns = [row["name"] for row in read_all(v1_db, "PRAGMA table_info(runs)")]
    assert len(run_columns) == len(set(run_columns))
    assert set(run_columns) == set(RunRecord.__dataclass_fields__)
    assert len(read_all(v1_db, "SELECT * FROM runs")) == 2


def test_run_record_defaults_describe_a_route_run(db_path: Path):
    telemetry = Telemetry(db_path)
    telemetry.record(make_run("route-run"), [make_call()])
    telemetry.close()

    row = read_all(db_path, "SELECT * FROM runs")[0]
    assert (row["mode"], row["verification_status"], row["repaired"], row["schema_escalated"]) == (
        "route",
        "skipped",
        0,
        0,
    )


@pytest.fixture
def chain_seeded_db(db_path: Path) -> Path:
    telemetry = Telemetry(db_path)
    for index in range(4):
        telemetry.record(
            make_run(f"route-{index}", total_latency_ms=1000.0 + index, total_cost_usd=0.0),
            [make_call()],
        )
    chain_runs = [
        ("chain-supported-0", "supported", False, False, "ok", 2000.0, 0.0, ["draft", "verify"]),
        ("chain-supported-1", "supported", False, True, "ok", 2500.0, 0.0, ["draft", "draft", "verify"]),
        ("chain-repaired", "unsupported", True, False, "ok", 9000.0, 0.04, ["draft", "verify", "repair"]),
        ("chain-repair-failed", "uncertain", False, False, "ok", 4000.0, 0.0, ["draft", "verify", "repair"]),
        ("chain-unavailable", "unavailable", False, False, "ok", 3000.0, 0.0, ["draft", "verify"]),
        ("chain-failed", "skipped", False, True, "error", 1500.0, 0.0, ["draft", "draft"]),
    ]
    for trace_id, verdict, repaired, escalated, status, latency_ms, cost_usd, roles in chain_runs:
        telemetry.record(
            make_run(
                trace_id,
                mode="chain",
                verification_status=verdict,
                repaired=repaired,
                schema_escalated=escalated,
                verifier_tier="verify",
                status=status,
                total_latency_ms=latency_ms,
                total_cost_usd=cost_usd,
            ),
            [make_call(seq=position + 1, role=role) for position, role in enumerate(roles)],
        )
    telemetry.close()
    return db_path


def test_report_chain_rates_and_verdict_counts(chain_seeded_db: Path):
    report = load_report_module()
    conn = report.open_read_only(chain_seeded_db)
    try:
        rates = report.chain_rates(conn)
        verdicts = report.chain_verdict_counts(conn)
        modes = {entry["mode"]: entry for entry in report.mode_breakdown(conn)}
    finally:
        conn.close()

    assert rates["runs"] == 6
    assert rates["repaired_runs"] == 1
    assert rates["repair_attempted_runs"] == 2
    assert rates["repair_failed_runs"] == 1
    assert rates["schema_escalated_runs"] == 2
    assert rates["repair_rate"] == pytest.approx(1 / 6)
    assert rates["schema_escalation_rate"] == pytest.approx(2 / 6)
    assert rates["error_rate"] == pytest.approx(1 / 6)

    counts = {entry["verification_status"]: entry["runs"] for entry in verdicts}
    assert counts == {"supported": 2, "unsupported": 1, "uncertain": 1, "unavailable": 1, "skipped": 1}
    assert all(entry["verifier_tier"] == "verify" for entry in verdicts)

    assert modes["route"]["runs"] == 4
    assert modes["route"]["total_cost_usd"] == 0.0
    assert modes["chain"]["runs"] == 6
    assert modes["chain"]["error_runs"] == 1
    assert modes["chain"]["total_cost_usd"] == pytest.approx(0.04)
    assert modes["chain"]["p50_latency_ms"] == 2500.0
    assert modes["chain"]["p95_latency_ms"] == 9000.0


def test_report_renders_the_chain_section(chain_seeded_db: Path, capsys):
    report = load_report_module()

    assert report.main(["--db", str(chain_seeded_db)]) == 0

    output = capsys.readouterr().out
    assert "Chain mode" in output
    assert "repair rate:            16.7% (1/6)" in output
    assert "schema-escalation rate: 33.3% (2/6)" in output
    assert "repair failure rate:    16.7% (1/6)" in output


def test_report_reads_an_unmigrated_v1_database(v1_db: Path, capsys):
    report = load_report_module()

    assert report.main(["--db", str(v1_db)]) == 0

    output = capsys.readouterr().out
    assert "this database predates chain mode" in output
    assert "Runs by route_reason" in output


# --- schema version 3: open-book columns, migration from version 1 and version 2 ---

V3_RUN_COLUMNS = {
    "sources_count",
    "passages_retrieved",
    "retrieval_method",
    "grounding_status",
    "abstained",
    "claims_total",
    "claims_supported",
    "grounding_verifier_tier",
}

V2_ADDED_COLUMNS = (
    "ALTER TABLE runs ADD COLUMN mode TEXT NOT NULL DEFAULT 'route'",
    "ALTER TABLE runs ADD COLUMN verification_status TEXT NOT NULL DEFAULT 'skipped'",
    "ALTER TABLE runs ADD COLUMN verification_critique TEXT",
    "ALTER TABLE runs ADD COLUMN repaired INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE runs ADD COLUMN schema_escalated INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE runs ADD COLUMN verifier_tier TEXT",
    "UPDATE schema_version SET version = 2",
    "UPDATE runs SET mode = 'chain', verification_status = 'supported', verifier_tier = 'verify'",
)


@pytest.fixture
def v2_db(v1_db: Path) -> Path:
    conn = sqlite3.connect(v1_db)
    try:
        with conn:
            for statement in V2_ADDED_COLUMNS:
                conn.execute(statement)
    finally:
        conn.close()
    return v1_db


def run_column_names(db_path: Path) -> set[str]:
    return {row["name"] for row in read_all(db_path, "PRAGMA table_info(runs)")}


def test_new_database_has_the_open_book_columns(db_path: Path):
    Telemetry(db_path).close()

    assert V3_RUN_COLUMNS <= run_column_names(db_path)


def test_v2_database_is_migrated_to_v3_and_old_rows_get_defaults(v2_db: Path):
    assert not V3_RUN_COLUMNS & run_column_names(v2_db)

    telemetry = Telemetry(v2_db)
    telemetry.record(
        make_run(
            "openbook-run",
            mode="openbook",
            sources_count=2,
            passages_retrieved=6,
            retrieval_method="hybrid",
            grounding_status="partly_grounded",
            abstained=True,
            repaired=True,
            claims_total=3,
            claims_supported=1,
            grounding_verifier_tier="ground_jev",
        ),
        [
            make_call(seq=1, role="embed"),
            make_call(seq=2, role="draft"),
            make_call(seq=3, role="ground"),
            make_call(seq=4, role="escalate"),
        ],
    )
    telemetry.close()

    assert V3_RUN_COLUMNS <= run_column_names(v2_db)
    assert [tuple(row) for row in read_all(v2_db, "SELECT version FROM schema_version")] == [(SCHEMA_VERSION,)]

    rows = {row["trace_id"]: dict(row) for row in read_all(v2_db, "SELECT * FROM runs")}
    old_row = rows["old-run"]
    assert old_row["prompt_text"] == "an old question"
    assert (old_row["mode"], old_row["verification_status"], old_row["verifier_tier"]) == (
        "chain",
        "supported",
        "verify",
    )
    assert old_row["sources_count"] == 0
    assert old_row["passages_retrieved"] == 0
    assert old_row["retrieval_method"] is None
    assert old_row["grounding_status"] == "skipped"
    assert old_row["abstained"] == 0
    assert old_row["claims_total"] == 0
    assert old_row["claims_supported"] == 0
    assert old_row["grounding_verifier_tier"] is None

    new_row = rows["openbook-run"]
    assert new_row["mode"] == "openbook"
    assert new_row["sources_count"] == 2
    assert new_row["passages_retrieved"] == 6
    assert new_row["retrieval_method"] == "hybrid"
    assert new_row["grounding_status"] == "partly_grounded"
    assert new_row["abstained"] == 1
    assert new_row["claims_total"] == 3
    assert new_row["claims_supported"] == 1
    assert new_row["grounding_verifier_tier"] == "ground_jev"
    call_roles = [row["role"] for row in read_all(v2_db, "SELECT role FROM calls WHERE trace_id = 'openbook-run' ORDER BY seq")]
    assert call_roles == ["embed", "draft", "ground", "escalate"]


def test_v1_database_is_migrated_straight_to_v3(v1_db: Path):
    Telemetry(v1_db).close()
    Telemetry(v1_db).close()

    assert V3_RUN_COLUMNS <= run_column_names(v1_db)
    assert "mode" in run_column_names(v1_db)
    assert [tuple(row) for row in read_all(v1_db, "SELECT version FROM schema_version")] == [(SCHEMA_VERSION,)]
    [old_row] = read_all(v1_db, "SELECT * FROM runs")
    assert old_row["grounding_status"] == "skipped"
    assert old_row["mode"] == "route"


def test_run_record_defaults_describe_a_run_without_sources(db_path: Path):
    telemetry = Telemetry(db_path)
    telemetry.record(make_run("plain"), [make_call()])
    telemetry.close()

    [row] = read_all(db_path, "SELECT * FROM runs")
    assert (row["sources_count"], row["passages_retrieved"], row["abstained"]) == (0, 0, 0)
    assert row["grounding_status"] == "skipped"
    assert row["retrieval_method"] is None


@pytest.fixture
def openbook_seeded_db(db_path: Path) -> Path:
    telemetry = Telemetry(db_path)
    telemetry.record(make_run("route-0"), [make_call()])
    telemetry.record(make_run("stuffed-0", mode="stuffed", sources_count=2, retrieval_method="none"), [make_call()])
    openbook_runs = [
        # trace_id, status, grounding, verifier, escalated, abstained, method, claims, supported, roles
        ("ob-grounded-0", "ok", "grounded", "ground_jev", False, False, "hybrid", 2, 2, ["embed", "draft", "ground"]),
        ("ob-grounded-1", "ok", "grounded", "ground_jev", False, False, "hybrid", 3, 3, ["draft", "ground"]),
        ("ob-partly", "ok", "partly_grounded", "ground_jev", True, False, "hybrid", 2, 2, ["draft", "ground", "escalate"]),
        ("ob-esc-failed", "ok", "partly_grounded", "ground_jev", False, False, "lexical", 2, 1, ["draft", "ground", "escalate"]),
        ("ob-abstained", "ok", "ungrounded", "ground_chat", True, True, "all", 1, 0, ["draft", "ground", "escalate"]),
        ("ob-agreed", "ok", "ungrounded", "ground_chat", False, True, "all", 0, 0, ["draft", "ground"]),
        ("ob-unavailable", "ok", "unavailable", "ground_jev", False, False, "hybrid", 2, 2, ["draft", "ground"]),
        ("ob-load-failed", "error", "skipped", None, False, False, None, 0, 0, []),
    ]
    for trace_id, status, grounding, verifier, escalated, abstained, method, claims, supported, roles in openbook_runs:
        telemetry.record(
            make_run(
                trace_id,
                mode="openbook",
                status=status,
                grounding_status=grounding,
                grounding_verifier_tier=verifier,
                repaired=escalated,
                abstained=abstained,
                retrieval_method=method,
                sources_count=2,
                passages_retrieved=6 if method else 0,
                claims_total=claims,
                claims_supported=supported,
            ),
            [make_call(seq=position + 1, role=role) for position, role in enumerate(roles)],
        )
    telemetry.close()
    return db_path


def test_report_open_book_rates_and_counts(openbook_seeded_db: Path):
    report = load_report_module()
    conn = report.open_read_only(openbook_seeded_db)
    try:
        rates = report.openbook_rates(conn)
        statuses = report.openbook_status_counts(conn)
        retrieval = {entry["retrieval_method"]: entry for entry in report.openbook_retrieval_counts(conn)}
        modes = {entry["mode"]: entry for entry in report.mode_breakdown(conn)}
    finally:
        conn.close()

    assert rates["runs"] == 8
    assert rates["escalated_runs"] == 2
    assert rates["escalation_attempted_runs"] == 3
    assert rates["escalation_failed_runs"] == 1
    assert rates["abstained_runs"] == 2
    assert rates["error_runs"] == 1
    assert rates["escalation_rate"] == pytest.approx(2 / 8)
    assert rates["abstention_rate"] == pytest.approx(2 / 8)
    assert (rates["claims_total"], rates["claims_supported"]) == (12, 10)
    assert rates["claim_support_rate"] == pytest.approx(10 / 12)

    by_status = {(entry["grounding_status"], entry["grounding_verifier_tier"]): entry for entry in statuses}
    assert by_status[("grounded", "ground_jev")]["runs"] == 2
    assert by_status[("partly_grounded", "ground_jev")]["escalated_runs"] == 1
    assert by_status[("ungrounded", "ground_chat")]["abstained_runs"] == 2
    assert by_status[("skipped", "(none)")]["runs"] == 1

    assert retrieval["hybrid"]["runs"] == 4
    assert retrieval["hybrid"]["mean_passages"] == 6.0
    assert retrieval["all"]["runs"] == 2
    assert retrieval["(none)"]["runs"] == 1

    assert modes["openbook"]["runs"] == 8
    assert modes["stuffed"]["runs"] == 1
    assert modes["route"]["runs"] == 1


def test_report_renders_the_open_book_section(openbook_seeded_db: Path, capsys):
    report = load_report_module()

    assert report.main(["--db", str(openbook_seeded_db)]) == 0

    output = capsys.readouterr().out
    assert "Open-book mode" in output
    assert "escalation rate:         25.0% (2/8)" in output
    assert "escalation failure rate: 12.5% (1/8)" in output
    assert "abstention rate:         25.0% (2/8)" in output
    assert "claims supported:        83.3% (10/12)" in output


def test_report_reads_a_v2_database_that_predates_open_book_mode(v2_db: Path, capsys):
    report = load_report_module()

    assert report.main(["--db", str(v2_db)]) == 0

    output = capsys.readouterr().out
    assert "this database predates open-book mode" in output
    assert "Chain mode" in output
    assert V3_RUN_COLUMNS.isdisjoint(run_column_names(v2_db))


# --- schema version 4: why a call stopped, and the grounding notes ---

V4_CALL_COLUMNS = {"finish_reason", "native_finish_reason", "reasoning_tokens"}


def call_column_names(db_path: Path) -> set[str]:
    return {row["name"] for row in read_all(db_path, "PRAGMA table_info(calls)")}


def test_new_database_has_the_v4_columns(db_path: Path):
    Telemetry(db_path).close()

    assert V4_CALL_COLUMNS <= call_column_names(db_path)
    assert "grounding_notes" in run_column_names(db_path)


def test_finish_fields_and_grounding_notes_round_trip(db_path: Path):
    telemetry = Telemetry(db_path)
    telemetry.record(
        make_run("trace-v4", grounding_notes='["passages answer the question: uncertain (p=0.45)"]'),
        [
            make_call(seq=0),
            make_call(
                seq=1,
                role="escalate",
                status="error",
                error_type="empty",
                finish_reason="length",
                native_finish_reason="max_tokens",
                reasoning_tokens=2048,
            ),
        ],
    )
    telemetry.close()

    [run_row] = read_all(db_path, "SELECT * FROM runs")
    assert run_row["grounding_notes"] == '["passages answer the question: uncertain (p=0.45)"]'
    first_call, second_call = read_all(db_path, "SELECT * FROM calls ORDER BY seq")
    assert first_call["finish_reason"] is None
    assert first_call["reasoning_tokens"] is None
    assert second_call["finish_reason"] == "length"
    assert second_call["native_finish_reason"] == "max_tokens"
    assert second_call["reasoning_tokens"] == 2048


def test_v3_database_is_migrated_to_v4_and_old_calls_stay_readable(v2_db: Path):
    telemetry = Telemetry(v2_db)
    telemetry.close()
    conn = sqlite3.connect(v2_db)
    try:
        with conn:
            conn.execute("ALTER TABLE calls DROP COLUMN finish_reason")
            conn.execute("ALTER TABLE calls DROP COLUMN native_finish_reason")
            conn.execute("ALTER TABLE calls DROP COLUMN reasoning_tokens")
            conn.execute("ALTER TABLE runs DROP COLUMN grounding_notes")
            conn.execute("UPDATE schema_version SET version = 3")
    finally:
        conn.close()
    calls_before = len(read_all(v2_db, "SELECT * FROM calls"))

    telemetry = Telemetry(v2_db)
    telemetry.record(make_run("trace-after-v4"), [make_call(seq=0, finish_reason="stop")])
    telemetry.close()

    assert V4_CALL_COLUMNS <= call_column_names(v2_db)
    assert "grounding_notes" in run_column_names(v2_db)
    assert [tuple(row) for row in read_all(v2_db, "SELECT version FROM schema_version")] == [(SCHEMA_VERSION,)]
    call_rows = read_all(v2_db, "SELECT * FROM calls ORDER BY id")
    assert len(call_rows) == calls_before + 1
    assert all(row["finish_reason"] is None for row in call_rows[:-1])
    assert call_rows[-1]["finish_reason"] == "stop"


# --- schema version 5: what the escalation was sent ---


def test_escalation_input_size_round_trips_and_defaults_to_null(db_path: Path):
    telemetry = Telemetry(db_path)
    telemetry.record(make_run("trace-plain"), [make_call(seq=0)])
    telemetry.record(
        make_run("trace-escalated", escalation_passages=9, escalation_tokens_estimated=2950),
        [make_call(seq=0)],
    )
    telemetry.close()

    rows = {row["trace_id"]: row for row in read_all(db_path, "SELECT * FROM runs")}
    assert rows["trace-plain"]["escalation_passages"] is None
    assert rows["trace-plain"]["escalation_tokens_estimated"] is None
    assert rows["trace-escalated"]["escalation_passages"] == 9
    assert rows["trace-escalated"]["escalation_tokens_estimated"] == 2950


def test_v4_database_is_migrated_to_v5(db_path: Path):
    Telemetry(db_path).close()
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.execute("ALTER TABLE runs DROP COLUMN escalation_passages")
            conn.execute("ALTER TABLE runs DROP COLUMN escalation_tokens_estimated")
            conn.execute("UPDATE schema_version SET version = 4")
    finally:
        conn.close()

    Telemetry(db_path).close()

    assert {"escalation_passages", "escalation_tokens_estimated"} <= run_column_names(db_path)
    assert [tuple(row) for row in read_all(db_path, "SELECT version FROM schema_version")] == [(SCHEMA_VERSION,)]


# --- schema version 6: why auto mode picked the mode ---


def test_auto_reason_round_trips_and_defaults_to_null(db_path: Path):
    telemetry = Telemetry(db_path)
    telemetry.record(make_run("trace-explicit"), [make_call(seq=0)])
    telemetry.record(
        make_run("trace-auto", mode="stuffed", auto_reason="auto: sources fit, 17k of 838k tokens"),
        [make_call(seq=0)],
    )
    telemetry.close()

    rows = {row["trace_id"]: row for row in read_all(db_path, "SELECT * FROM runs")}
    assert rows["trace-explicit"]["auto_reason"] is None
    assert rows["trace-auto"]["mode"] == "stuffed"
    assert rows["trace-auto"]["auto_reason"] == "auto: sources fit, 17k of 838k tokens"


def test_v5_database_is_migrated_to_v6(db_path: Path):
    Telemetry(db_path).close()
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.execute("ALTER TABLE runs DROP COLUMN auto_reason")
            conn.execute("UPDATE schema_version SET version = 5")
    finally:
        conn.close()

    Telemetry(db_path).close()

    assert "auto_reason" in run_column_names(db_path)
    assert [tuple(row) for row in read_all(db_path, "SELECT version FROM schema_version")] == [(SCHEMA_VERSION,)]


# --- schema version 7: execute mode, the sandbox_runs audit table, migration from version 6 ---

from inference_arbiter.telemetry import SandboxRunRecord  # noqa: E402

V7_RUN_COLUMNS = {"execution_status", "claimed_answer", "executed_answer"}
SANDBOX_RUN_COLUMNS = {
    "id", "trace_id", "seq", "attempt", "sandbox_name", "runner", "program", "claimed_answer",
    "executed_answer", "status", "error_type", "exit_code", "stdout", "stderr", "stdout_truncated",
    "stderr_truncated", "duration_ms", "total_ms", "timed_out", "network_policy", "environment_names",
    "sandbox_removed", "outcome", "program_label",
}


def make_sandbox_run(**overrides) -> SandboxRunRecord:
    fields = dict(
        seq=1, attempt="first", sandbox_name="inference-arbiter-0123456789ab", runner="python",
        program="print(17 * 23)", claimed_answer="391", executed_answer="391", status="ok",
        error_type=None, exit_code=0, stdout="391\n", stderr="", stdout_truncated=False,
        stderr_truncated=False, duration_ms=250.0, total_ms=3300.0, timed_out=False,
        network_policy="deny_all", environment_names='["PATH", "HOME", "LANG"]',
        sandbox_removed=True, outcome="agree",
    )
    fields.update(overrides)
    return SandboxRunRecord(**fields)


def table_column_names(db_path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


@pytest.fixture
def v6_db(db_path: Path) -> Path:
    """A version 6 database with one run: no sandbox_runs table, no execute columns."""
    telemetry = Telemetry(db_path)
    telemetry.record(make_run("old-v6-run"), [make_call()])
    telemetry.close()
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.execute("DROP INDEX idx_sandbox_runs_trace_id")
            conn.execute("DROP TABLE sandbox_runs")
            for column in V7_RUN_COLUMNS:
                conn.execute(f"ALTER TABLE runs DROP COLUMN {column}")
            conn.execute("UPDATE schema_version SET version = 6")
    finally:
        conn.close()
    assert "sandbox_runs" not in {row["name"] for row in read_all(db_path, "SELECT name FROM sqlite_master")}
    return db_path


def test_new_database_has_the_audit_table_and_the_execute_columns(db_path: Path):
    Telemetry(db_path).close()

    assert table_column_names(db_path, "sandbox_runs") == SANDBOX_RUN_COLUMNS
    assert V7_RUN_COLUMNS <= run_column_names(db_path)
    indexes = {row["name"] for row in read_all(db_path, "SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert "idx_sandbox_runs_trace_id" in indexes


def test_v6_database_is_migrated_to_v7_and_old_rows_get_defaults(v6_db: Path):
    Telemetry(v6_db).close()

    assert [tuple(row) for row in read_all(v6_db, "SELECT version FROM schema_version")] == [(SCHEMA_VERSION,)]
    assert table_column_names(v6_db, "sandbox_runs") == SANDBOX_RUN_COLUMNS
    old_run = read_all(v6_db, "SELECT * FROM runs WHERE trace_id = 'old-v6-run'")[0]
    assert old_run["execution_status"] == "skipped"
    assert old_run["claimed_answer"] is None and old_run["executed_answer"] is None
    assert len(read_all(v6_db, "SELECT * FROM calls")) == 1
    assert read_all(v6_db, "SELECT * FROM sandbox_runs") == []


def test_v6_to_v7_migration_is_idempotent_and_the_migrated_database_takes_audit_rows(v6_db: Path):
    Telemetry(v6_db).close()
    telemetry = Telemetry(v6_db)
    telemetry.record(
        make_run("after-migration", mode="execute", execution_status="claim_matched"), [make_call()], [make_sandbox_run()]
    )
    telemetry.close()

    assert [tuple(row) for row in read_all(v6_db, "SELECT version FROM schema_version")] == [(SCHEMA_VERSION,)]
    assert [row["trace_id"] for row in read_all(v6_db, "SELECT trace_id FROM sandbox_runs")] == ["after-migration"]
    assert len(read_all(v6_db, "SELECT * FROM runs")) == 2


def test_v1_database_is_migrated_straight_to_v7(v1_db: Path):
    Telemetry(v1_db).close()

    assert V7_RUN_COLUMNS <= run_column_names(v1_db)
    assert table_column_names(v1_db, "sandbox_runs") == SANDBOX_RUN_COLUMNS
    assert [tuple(row) for row in read_all(v1_db, "SELECT version FROM schema_version")] == [(SCHEMA_VERSION,)]


def test_sandbox_audit_rows_round_trip(db_path: Path):
    telemetry = Telemetry(db_path)
    failed = make_sandbox_run(
        seq=1, program="print(1/0)", claimed_answer="41", executed_answer=None, status="error",
        error_type="nonzero_exit", exit_code=1, stdout="", stderr="ZeroDivisionError\n", outcome="program_failed",
    )
    timed_out = make_sandbox_run(
        seq=2, attempt="repair", sandbox_name="inference-arbiter-ba9876543210", status="error", error_type="timeout",
        exit_code=None, stdout="x" * 20, stdout_truncated=True, timed_out=True, executed_answer=None,
        sandbox_removed=False, outcome="program_failed",
    )
    telemetry.record(
        make_run("exec-run", mode="execute", execution_status="unverified", claimed_answer="391"),
        [make_call()],
        [failed, timed_out],
    )
    telemetry.close()

    run = read_all(db_path, "SELECT * FROM runs")[0]
    assert (run["mode"], run["execution_status"], run["claimed_answer"], run["executed_answer"]) == (
        "execute", "unverified", "391", None
    )
    first, second = read_all(db_path, "SELECT * FROM sandbox_runs ORDER BY seq")
    assert (first["trace_id"], first["seq"], first["attempt"], first["runner"]) == ("exec-run", 1, "first", "python")
    assert (first["program"], first["exit_code"], first["stderr"]) == ("print(1/0)", 1, "ZeroDivisionError\n")
    assert (first["status"], first["error_type"], first["outcome"]) == ("error", "nonzero_exit", "program_failed")
    assert (first["network_policy"], first["environment_names"]) == ("deny_all", '["PATH", "HOME", "LANG"]')
    assert (first["timed_out"], first["sandbox_removed"], first["stdout_truncated"]) == (0, 1, 0)
    assert (first["duration_ms"], first["total_ms"]) == (250.0, 3300.0)
    assert (second["attempt"], second["exit_code"], second["timed_out"]) == ("repair", None, 1)
    assert (second["sandbox_removed"], second["stdout_truncated"], second["sandbox_name"]) == (
        0, 1, "inference-arbiter-ba9876543210"
    )


def test_a_run_without_sandbox_runs_writes_no_audit_rows_and_defaults_to_skipped(db_path: Path):
    telemetry = Telemetry(db_path)
    telemetry.record(make_run("plain"), [make_call()])
    telemetry.record(make_run("plain-none"), [make_call()], None)
    telemetry.close()

    assert read_all(db_path, "SELECT * FROM sandbox_runs") == []
    assert {row["execution_status"] for row in read_all(db_path, "SELECT * FROM runs")} == {"skipped"}


def test_a_bad_audit_row_rolls_back_the_whole_run(db_path: Path, capsys):
    telemetry = Telemetry(db_path)
    telemetry.record(make_run("atomic-exec"), [make_call()], [make_sandbox_run(program=None)])
    telemetry.record(make_run("after"), [make_call()], [make_sandbox_run()])
    telemetry.close()

    assert [row["trace_id"] for row in read_all(db_path, "SELECT trace_id FROM runs")] == ["after"]
    assert len(read_all(db_path, "SELECT * FROM calls")) == 1
    assert len(read_all(db_path, "SELECT * FROM sandbox_runs")) == 1
    assert "telemetry warning" in capsys.readouterr().err


def test_the_audit_table_has_no_column_for_environment_values():
    assert not any("value" in column or "secret" in column or "key" in column for column in SANDBOX_RUN_COLUMNS)


# --- schema version 8: the cross-check columns, and `verified` gets one meaning ---

V8_RUN_COLUMNS = {"cross_check", "cross_check_tier", "executed_answer_b", "cross_check_rule"}


@pytest.fixture
def v7_db(db_path: Path) -> Path:
    """A version 7 database: no cross-check columns, and `verified` still means claim-based."""
    telemetry = Telemetry(db_path)
    telemetry.record(
        make_run("old-claim-run", mode="execute", execution_status="claim_matched", claimed_answer="391"),
        [make_call()],
        [make_sandbox_run()],
    )
    telemetry.record(make_run("old-executed-run", mode="execute", execution_status="executed"), [make_call()])
    telemetry.close()
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            for column in V8_RUN_COLUMNS:
                conn.execute(f"ALTER TABLE runs DROP COLUMN {column}")
            conn.execute("ALTER TABLE sandbox_runs DROP COLUMN program_label")
            conn.execute("UPDATE runs SET execution_status = 'verified' WHERE trace_id = 'old-claim-run'")
            conn.execute("UPDATE schema_version SET version = 7")
    finally:
        conn.close()
    return db_path


def test_new_database_has_the_cross_check_columns(db_path: Path):
    Telemetry(db_path).close()

    assert V8_RUN_COLUMNS <= run_column_names(db_path)
    assert "program_label" in table_column_names(db_path, "sandbox_runs")


def test_v7_database_is_migrated_to_v8_and_old_rows_get_defaults(v7_db: Path):
    Telemetry(v7_db).close()

    assert [tuple(row) for row in read_all(v7_db, "SELECT version FROM schema_version")] == [(8,)]
    assert V8_RUN_COLUMNS <= run_column_names(v7_db)
    old_run = read_all(v7_db, "SELECT * FROM runs WHERE trace_id = 'old-executed-run'")[0]
    assert (old_run["cross_check"], old_run["cross_check_tier"], old_run["executed_answer_b"]) == (0, None, None)
    assert old_run["cross_check_rule"] is None and old_run["execution_status"] == "executed"
    assert [row["program_label"] for row in read_all(v7_db, "SELECT * FROM sandbox_runs")] == ["a"]


def test_the_v8_migration_renames_the_claim_based_verified_so_the_word_has_one_meaning(v7_db: Path):
    Telemetry(v7_db).close()

    statuses = {row["trace_id"]: row["execution_status"] for row in read_all(v7_db, "SELECT * FROM runs")}
    assert statuses == {"old-claim-run": "claim_matched", "old-executed-run": "executed"}


def test_the_v8_migration_runs_once_and_leaves_a_cross_check_verified_alone(v7_db: Path):
    Telemetry(v7_db).close()
    telemetry = Telemetry(v7_db)
    telemetry.record(
        make_run(
            "cross-run", mode="execute", execution_status="verified", cross_check=True,
            cross_check_tier="code", executed_answer="391", executed_answer_b="391", cross_check_rule="integer",
        ),
        [make_call()],
        [make_sandbox_run(outcome="ran"), make_sandbox_run(seq=2, outcome="ran", program_label="b")],
    )
    telemetry.close()
    Telemetry(v7_db).close()

    run = read_all(v7_db, "SELECT * FROM runs WHERE trace_id = 'cross-run'")[0]
    assert run["execution_status"] == "verified", "a version 8 row is not rewritten when the database is reopened"
    assert (run["cross_check"], run["cross_check_tier"], run["executed_answer_b"], run["cross_check_rule"]) == (
        1, "code", "391", "integer"
    )
    labels = [row["program_label"] for row in read_all(v7_db, "SELECT * FROM sandbox_runs WHERE trace_id = 'cross-run' ORDER BY seq")]
    assert labels == ["a", "b"]
