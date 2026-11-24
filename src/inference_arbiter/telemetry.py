"""SQLite telemetry for the inference arbiter.

One row per run in `runs`, one row per model call in `calls`, and one row per program
run in a sandbox in `sandbox_runs` (the audit log of execute mode). Standard
library only; this module must not import anything else from inference_arbiter.

Secrets, API keys and HTTP headers are never stored: the schema has no
column for them and callers must not place them in any text field.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# Version 2 adds the chain-mode columns on `runs`, version 3 the open-book columns.
# Version 4 adds why each call stopped (`calls.finish_reason`, `native_finish_reason`,
# `reasoning_tokens`) and the open-book grounding notes (`runs.grounding_notes`).
# Version 5 adds what an open-book escalation was sent (`runs.escalation_passages`,
# `runs.escalation_tokens_estimated`).
# Version 6 adds why auto mode picked the run's mode (`runs.auto_reason`).
# Version 7 adds execute mode: the audit table `sandbox_runs` (one row per program run
# in a sandbox) and `runs.execution_status`, `runs.claimed_answer`, `runs.executed_answer`.
# Version 8 adds the execute-mode cross-check: `runs.cross_check`, `runs.cross_check_tier`,
# `runs.executed_answer_b`, `runs.cross_check_rule` and `sandbox_runs.program_label`. It also
# renames a status: up to version 7 `execution_status = 'verified'` meant "the program's
# output matched the claim of the model that wrote it". From version 8 that is
# `claim_matched`, and `verified` means two independently written programs agreed. The
# migration rewrites the old rows, so `verified` has one meaning in every database.
# Codetest mode added no column and no version: its statuses (tests_passed, tests_failed,
# untested) are values of `runs.execution_status`, its calls have the roles `interface` and
# `tests`, and its sandbox runs are `sandbox_runs` rows.
# Older databases are migrated in place when opened: columns are added with defaults,
# so rows written by an earlier version stay readable.
SCHEMA_VERSION = 8


@dataclass
class CallRecord:
    seq: int
    role: str  # triage | answer | draft | verify | repair | embed | ground | escalate
    tier: str
    provider: str
    model: str
    effort: str | None
    timeout_s: float | None
    status: str  # 'ok' | 'timeout' | 'error' | 'invalid'
    error_type: str | None
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    cost_usd: float | None
    cost_source: str | None  # 'provider_reported' | 'estimated' | 'local_zero'
    ollama_load_duration_ms: float | None
    ollama_total_duration_ms: float | None
    ollama_prompt_eval_ms: float | None
    ollama_eval_ms: float | None
    # Added in schema version 4.
    finish_reason: str | None = None  # stop | length | content_filter | tool_calls | error
    native_finish_reason: str | None = None  # the provider's own value, when it differs
    reasoning_tokens: int | None = None  # part of completion_tokens spent on reasoning


@dataclass
class SandboxRunRecord:
    """One program run in a sandbox. The program text is stored in full; stdout and stderr
    are stored as the gate capped them. Only the NAMES of the environment variables are
    stored: the gate builds the environment itself and it holds no secret."""

    seq: int
    attempt: str  # first | repair
    sandbox_name: str | None  # NULL when the gate refused before creating a sandbox
    runner: str  # python
    program: str
    claimed_answer: str | None
    # Last non-empty stdout line of a clean run. Codetest mode: the test run's result line.
    executed_answer: str | None
    status: str  # 'ok' | 'error'
    error_type: str | None  # sandbox_unavailable | timeout | nonzero_exit | output_too_large | ...
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    duration_ms: float  # the program run only
    total_ms: float  # create, checks, run and removal
    timed_out: bool
    network_policy: str  # deny_all
    environment_names: str  # JSON list of the variable names the program was given
    sandbox_removed: bool
    # agree | disagree | no_claim | program_failed | infra_failed; in a cross-check
    # ran | program_failed | infra_failed; in codetest mode passed | vacuous | tests_failed |
    # solution_invalid | tests_invalid | not_finished | infra_failed
    outcome: str
    # Added in schema version 8: which program of a cross-check this is (a | b).
    program_label: str = "a"


@dataclass
class RunRecord:
    trace_id: str
    ts_utc: str  # ISO 8601
    prompt_sha256: str
    prompt_text: str | None
    config_sha256: str
    triage_status: str
    triage_label: str | None
    triage_confidence: float | None
    triage_reason: str | None
    route_reason: str
    selected_tier: str
    final_tier: str | None
    final_model: str | None
    effort: str | None
    fallback_used: bool
    status: str  # 'ok' | 'error'
    error_type: str | None
    total_latency_ms: float
    total_cost_usd: float | None
    # Added in schema version 2. Defaults describe a route-mode run.
    mode: str = "route"  # route | chain | openbook | stuffed | execute
    verification_status: str = "skipped"  # supported | unsupported | uncertain | unavailable | skipped
    verification_critique: str | None = None
    repaired: bool = False
    schema_escalated: bool = False
    verifier_tier: str | None = None
    # Added in schema version 3. Defaults describe a run without sources. In openbook
    # mode `repaired` means "escalated".
    sources_count: int = 0
    passages_retrieved: int = 0
    retrieval_method: str | None = None  # hybrid | embedding | lexical | all | none
    grounding_status: str = "skipped"  # grounded | partly_grounded | ungrounded | unavailable | skipped
    abstained: bool = False
    claims_total: int = 0
    claims_supported: int = 0
    grounding_verifier_tier: str | None = None
    # Added in schema version 4: JSON list of the per-claim and answerability notes.
    grounding_notes: str | None = None
    # Added in schema version 5: what the open-book escalation call was sent. NULL when
    # no escalation was attempted.
    escalation_passages: int | None = None
    escalation_tokens_estimated: int | None = None
    # Added in schema version 6: why auto mode picked `mode`. NULL when the caller picked it.
    auto_reason: str | None = None
    # Added in schema version 7. Defaults describe a run outside execute mode.
    # verified | disputed | one_sided | claim_matched | corrected | executed | unverified | failed | skipped
    execution_status: str = "skipped"
    claimed_answer: str | None = None
    executed_answer: str | None = None  # in a cross-check: program A's output
    # Added in schema version 8. Defaults describe a run without the cross-check.
    cross_check: bool = False
    cross_check_tier: str | None = None  # the tier that wrote program B
    executed_answer_b: str | None = None  # program B's output; NULL when it did not run cleanly
    cross_check_rule: str | None = None  # integer | hex | decimal | number_list | string


# Columns added to `runs` after version 1, as (name, column definition). Used both in
# CREATE TABLE for new databases and in ALTER TABLE for existing ones.
_RUN_COLUMNS_ADDED_IN_V2 = (
    ("mode", "TEXT NOT NULL DEFAULT 'route'"),
    ("verification_status", "TEXT NOT NULL DEFAULT 'skipped'"),
    ("verification_critique", "TEXT"),
    ("repaired", "INTEGER NOT NULL DEFAULT 0"),
    ("schema_escalated", "INTEGER NOT NULL DEFAULT 0"),
    ("verifier_tier", "TEXT"),
)

_RUN_COLUMNS_ADDED_IN_V3 = (
    ("sources_count", "INTEGER NOT NULL DEFAULT 0"),
    ("passages_retrieved", "INTEGER NOT NULL DEFAULT 0"),
    ("retrieval_method", "TEXT"),
    ("grounding_status", "TEXT NOT NULL DEFAULT 'skipped'"),
    ("abstained", "INTEGER NOT NULL DEFAULT 0"),
    ("claims_total", "INTEGER NOT NULL DEFAULT 0"),
    ("claims_supported", "INTEGER NOT NULL DEFAULT 0"),
    ("grounding_verifier_tier", "TEXT"),
)

_RUN_COLUMNS_ADDED_IN_V4 = (("grounding_notes", "TEXT"),)

_RUN_COLUMNS_ADDED_IN_V5 = (
    ("escalation_passages", "INTEGER"),
    ("escalation_tokens_estimated", "INTEGER"),
)

_RUN_COLUMNS_ADDED_IN_V6 = (("auto_reason", "TEXT"),)

_RUN_COLUMNS_ADDED_IN_V7 = (
    ("execution_status", "TEXT NOT NULL DEFAULT 'skipped'"),
    ("claimed_answer", "TEXT"),
    ("executed_answer", "TEXT"),
)

_RUN_COLUMNS_ADDED_IN_V8 = (
    ("cross_check", "INTEGER NOT NULL DEFAULT 0"),
    ("cross_check_tier", "TEXT"),
    ("executed_answer_b", "TEXT"),
    ("cross_check_rule", "TEXT"),
)

_SANDBOX_RUN_COLUMNS_ADDED_IN_V8 = (("program_label", "TEXT NOT NULL DEFAULT 'a'"),)

_CALL_COLUMNS_ADDED_IN_V4 = (
    ("finish_reason", "TEXT"),
    ("native_finish_reason", "TEXT"),
    ("reasoning_tokens", "INTEGER"),
)


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
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
    """,  # the columns of later versions are added by _migrate, for new and old databases alike
    """
    CREATE TABLE IF NOT EXISTS calls (
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
    # Added in schema version 7. CREATE TABLE IF NOT EXISTS is the migration: a version 6
    # database gets the table the first time it is opened.
    """
    CREATE TABLE IF NOT EXISTS sandbox_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trace_id TEXT NOT NULL REFERENCES runs(trace_id),
        seq INTEGER NOT NULL,
        attempt TEXT NOT NULL,
        sandbox_name TEXT,
        runner TEXT NOT NULL,
        program TEXT NOT NULL,
        claimed_answer TEXT,
        executed_answer TEXT,
        status TEXT NOT NULL,
        error_type TEXT,
        exit_code INTEGER,
        stdout TEXT NOT NULL,
        stderr TEXT NOT NULL,
        stdout_truncated INTEGER NOT NULL,
        stderr_truncated INTEGER NOT NULL,
        duration_ms REAL NOT NULL,
        total_ms REAL NOT NULL,
        timed_out INTEGER NOT NULL,
        network_policy TEXT NOT NULL,
        environment_names TEXT NOT NULL,
        sandbox_removed INTEGER NOT NULL,
        outcome TEXT NOT NULL
    )
    """,  # program_label (version 8) is added by _migrate, for new and old databases alike
    "CREATE INDEX IF NOT EXISTS idx_sandbox_runs_trace_id ON sandbox_runs(trace_id)",
    "CREATE INDEX IF NOT EXISTS idx_runs_ts_utc ON runs(ts_utc)",
    "CREATE INDEX IF NOT EXISTS idx_calls_trace_id ON calls(trace_id)",
)

_RUN_COLUMNS = tuple(field.name for field in fields(RunRecord))
_CALL_COLUMNS = ("trace_id",) + tuple(field.name for field in fields(CallRecord))
_SANDBOX_RUN_COLUMNS = ("trace_id",) + tuple(field.name for field in fields(SandboxRunRecord))
_SANDBOX_RUN_FLAGS = ("stdout_truncated", "stderr_truncated", "timed_out", "sandbox_removed")


def _insert_statement(table: str, columns: tuple[str, ...]) -> str:
    placeholders = ", ".join(f":{column}" for column in columns)
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"


_INSERT_RUN = _insert_statement("runs", _RUN_COLUMNS)
_INSERT_CALL = _insert_statement("calls", _CALL_COLUMNS)
_INSERT_SANDBOX_RUN = _insert_statement("sandbox_runs", _SANDBOX_RUN_COLUMNS)


def _warn(message: str) -> None:
    one_line = " ".join(str(message).split())
    try:
        print(f"telemetry warning: {one_line}", file=sys.stderr)
    except Exception:
        pass


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring both tables up to SCHEMA_VERSION. Safe to run on any version, any number of times."""
    added_columns = {
        "runs": _RUN_COLUMNS_ADDED_IN_V2 + _RUN_COLUMNS_ADDED_IN_V3
        + _RUN_COLUMNS_ADDED_IN_V4
        + _RUN_COLUMNS_ADDED_IN_V5
        + _RUN_COLUMNS_ADDED_IN_V6
        + _RUN_COLUMNS_ADDED_IN_V7
        + _RUN_COLUMNS_ADDED_IN_V8,
        "calls": _CALL_COLUMNS_ADDED_IN_V4,
        "sandbox_runs": _SANDBOX_RUN_COLUMNS_ADDED_IN_V8,
    }
    for table_name, table_columns in added_columns.items():
        existing_columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")}
        for column_name, column_definition in table_columns:
            if column_name not in existing_columns:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")

    existing_version = conn.execute("SELECT version FROM schema_version").fetchone()
    if existing_version is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    elif existing_version[0] < SCHEMA_VERSION:
        if existing_version[0] < 8:
            # Before version 8 `verified` was the claim-based status (see SCHEMA_VERSION).
            conn.execute(
                "UPDATE runs SET execution_status = 'claim_matched' WHERE execution_status = 'verified'"
            )
        conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))


class Telemetry:
    """Writes run and call records to SQLite. Never raises to the caller."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False plus the lock lets a router that offloads
            # logging to a worker thread share this one connection safely.
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA foreign_keys=ON")
                with conn:
                    for statement in _SCHEMA_STATEMENTS:
                        conn.execute(statement)
                    _migrate(conn)
            except Exception:
                conn.close()
                raise
            self._conn = conn
        except Exception as error:
            _warn(
                f"could not open {self.db_path}, telemetry disabled: "
                f"{type(error).__name__}: {error}"
            )

    def record(
        self,
        run: RunRecord,
        calls: list[CallRecord],
        sandbox_runs: list[SandboxRunRecord] | None = None,
    ) -> None:
        """Write the run row, its call rows and its sandbox audit rows in one transaction."""
        try:
            if self._conn is None:
                raise RuntimeError("database is not open")
            run_row = asdict(run)
            run_row["fallback_used"] = int(bool(run.fallback_used))
            run_row["repaired"] = int(bool(run.repaired))
            run_row["schema_escalated"] = int(bool(run.schema_escalated))
            run_row["abstained"] = int(bool(run.abstained))
            run_row["cross_check"] = int(bool(run.cross_check))
            call_rows = []
            for call in calls:
                call_row = asdict(call)
                call_row["trace_id"] = run.trace_id
                call_rows.append(call_row)
            sandbox_rows = []
            for sandbox_run in sandbox_runs or []:
                sandbox_row = asdict(sandbox_run)
                sandbox_row["trace_id"] = run.trace_id
                for flag in _SANDBOX_RUN_FLAGS:
                    sandbox_row[flag] = int(bool(sandbox_row[flag]))
                sandbox_rows.append(sandbox_row)
            with self._lock:
                # The connection context manager commits on success and rolls
                # back on any exception, so a run never lands without its calls.
                with self._conn:
                    self._conn.execute(_INSERT_RUN, run_row)
                    self._conn.executemany(_INSERT_CALL, call_rows)
                    self._conn.executemany(_INSERT_SANDBOX_RUN, sandbox_rows)
        except Exception as error:
            trace_id = getattr(run, "trace_id", "unknown")
            _warn(
                f"record failed for trace_id={trace_id}: "
                f"{type(error).__name__}: {error}"
            )

    def close(self) -> None:
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.close()
            except Exception as error:
                _warn(f"close failed: {type(error).__name__}: {error}")
            self._conn = None
