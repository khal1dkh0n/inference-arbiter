"""Host-side dispatch gate: run one untrusted program inside a throwaway `sbx` MicroVM.

`run_program` is the only way a program reaches a sandbox. It is plain control flow:

1. Refuse a runner that is not on the allow-list, or a program over the size cap. No
   sandbox is created for a refused program.
2. Create a fresh sandbox (`sbx create ... shell`) with no workspace and no host mount,
   and with a sandbox-scoped deny-all network rule (`--deny-network '**'`). The global
   policy is never changed.
3. Ask `sbx policy check` whether the new sandbox may reach a host the global policy
   allows. If the answer is not an explicit "denied", the program is not run.
4. Copy the program in over stdin (no bind mount) into a directory owned by `nobody`.
5. Run it as `nobody` under `env -i` with the environment in `SANDBOX_ENVIRONMENT`, and
   under `timeout` inside the VM. A second, longer wall-clock timeout runs on the host.
6. Remove the sandbox, whatever happened in steps 2 to 5.

What crosses the boundary: the program text goes in; exit code, capped stdout and capped
stderr come out. Nothing else. Every `sbx` subprocess gets an environment built from
scratch (`PATH` and `HOME` only), so no host variable, and so no API key, can be passed
through, not even by a `-e NAME` flag that reads the caller's environment.

Every failure is a typed `error_type` on the result and fails closed: a result is `ok`
only when the program ran to completion with exit code 0 inside the limits. The `sbx`
call sits behind `CommandRunner`, so tests use a fake and never start a sandbox.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Literal, Protocol

RUNNER_PYTHON = "python"
ALLOWED_RUNNERS = (RUNNER_PYTHON,)

NETWORK_DENY_ALL = "deny_all"
# The sbx pattern that matches every host (see `sbx policy allow network --help`).
SBX_ALL_HOSTS_PATTERN = "**"

SANDBOX_NAME_PREFIX = "inference-arbiter-"
SANDBOX_AGENT = "shell"
SANDBOX_RUN_DIR = "/sandbox-run"
SANDBOX_PROGRAM_NAME = "program.py"
SANDBOX_USER = "nobody"
SANDBOX_UID_GID = "65534:65534"

# The whole environment of the program. Built here, never copied from the host.
SANDBOX_ENVIRONMENT = (
    ("PATH", "/usr/bin:/bin"),
    ("HOME", SANDBOX_RUN_DIR),
    ("LANG", "C.UTF-8"),
)
# The only host variables an `sbx` subprocess is given: it needs them to find its daemon.
HOST_ENVIRONMENT_NAMES = ("PATH", "HOME")

# `timeout` sends TERM at the limit and KILL this many seconds later.
KILL_GRACE_S = 2
# The host-side timeout is the in-VM one plus this, so the in-VM one normally fires first.
HOST_TIMEOUT_MARGIN_S = 8
EXIT_TIMEOUT_TERM = 124
EXIT_KILLED = 137
EXIT_SBX_COULD_NOT_RUN = 125

MANAGEMENT_OUTPUT_BYTES = 20_000
MAX_ERROR_MESSAGE_CHARS = 300

ERROR_RUNNER_NOT_ALLOWED = "runner_not_allowed"
ERROR_PROGRAM_TOO_LARGE = "program_too_large"
ERROR_SANDBOX_UNAVAILABLE = "sandbox_unavailable"
ERROR_NETWORK_POLICY_NOT_APPLIED = "network_policy_not_applied"
ERROR_TIMEOUT = "timeout"
ERROR_NONZERO_EXIT = "nonzero_exit"
ERROR_OUTPUT_TOO_LARGE = "output_too_large"

# The program itself misbehaved: a repair by the model can help.
PROGRAM_ERRORS = (ERROR_TIMEOUT, ERROR_NONZERO_EXIT, ERROR_OUTPUT_TOO_LARGE)
# The gate or the sandbox could not do its job: the program was not at fault.
INFRASTRUCTURE_ERRORS = (
    ERROR_RUNNER_NOT_ALLOWED,
    ERROR_PROGRAM_TOO_LARGE,
    ERROR_SANDBOX_UNAVAILABLE,
    ERROR_NETWORK_POLICY_NOT_APPLIED,
)


@dataclass(frozen=True)
class CommandResult:
    """What one host command returned."""

    exit_code: int | None  # None when the command could not be started or was killed here
    stdout: str
    stderr: str
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    duration_s: float = 0.0
    # Set when the executable could not be started at all (for example sbx is not installed).
    start_error: str | None = None

    @property
    def output_truncated(self) -> bool:
        return self.stdout_truncated or self.stderr_truncated


class CommandRunner(Protocol):
    def run(
        self,
        argv: list[str],
        stdin_text: str | None,
        timeout_s: float,
        max_output_bytes: int,
    ) -> CommandResult: ...


def host_environment() -> dict[str, str]:
    """The environment every `sbx` subprocess gets: PATH and HOME, nothing else."""
    return {name: os.environ[name] for name in HOST_ENVIRONMENT_NAMES if name in os.environ}


class SubprocessRunner:
    """Runs a host command with a wall-clock timeout and a cap on each output stream."""

    def run(
        self,
        argv: list[str],
        stdin_text: str | None,
        timeout_s: float,
        max_output_bytes: int,
    ) -> CommandResult:
        started_at = time.monotonic()
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=host_environment(),
                start_new_session=True,
            )
        except OSError as start_error:
            return CommandResult(
                exit_code=None,
                stdout="",
                stderr="",
                duration_s=time.monotonic() - started_at,
                start_error=f"{type(start_error).__name__}: {start_error}",
            )

        collected: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
        overflowed: set[str] = set()

        def drain(stream_name: str, stream) -> None:
            buffer = collected[stream_name]
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    return
                room = max_output_bytes - len(buffer)
                if len(chunk) > room:
                    buffer.extend(chunk[: max(room, 0)])
                    overflowed.add(stream_name)
                    # Stop the producer; keep reading so the pipe drains and the thread ends.
                    _kill(process)
                else:
                    buffer.extend(chunk)

        def feed() -> None:
            try:
                process.stdin.write(stdin_text.encode("utf-8"))
            except OSError:
                pass
            finally:
                try:
                    process.stdin.close()
                except OSError:
                    pass

        threads = [
            threading.Thread(target=drain, args=("stdout", process.stdout), daemon=True),
            threading.Thread(target=drain, args=("stderr", process.stderr), daemon=True),
        ]
        if stdin_text is not None:
            threads.append(threading.Thread(target=feed, daemon=True))
        for thread in threads:
            thread.start()

        timed_out = False
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill(process)
            process.wait()
        for thread in threads:
            thread.join(timeout=5)

        was_cut = timed_out or bool(overflowed)
        return CommandResult(
            exit_code=None if was_cut else process.returncode,
            stdout=collected["stdout"].decode("utf-8", errors="replace"),
            stderr=collected["stderr"].decode("utf-8", errors="replace"),
            timed_out=timed_out,
            stdout_truncated="stdout" in overflowed,
            stderr_truncated="stderr" in overflowed,
            duration_s=time.monotonic() - started_at,
        )


def _kill(process: subprocess.Popen) -> None:
    try:
        process.kill()
    except OSError:
        pass


@dataclass(frozen=True)
class SandboxLimits:
    """Limits for one program run. The defaults are the shipped `execute:` block."""

    timeout_s: float = 20.0
    max_output_bytes: int = 20_000
    max_program_bytes: int = 20_000
    cpus: int = 2
    memory: str = "1g"
    # Covers a first run that has to pull the sandbox image.
    create_timeout_s: float = 180.0
    management_timeout_s: float = 30.0
    sbx_path: str = "sbx"
    # A host the global policy allows; the new sandbox must be denied it before a program runs.
    network_canary: str = "pypi.org:443"


@dataclass(frozen=True)
class SandboxResult:
    """The typed result of one program run. `ok` only for a clean exit inside the limits."""

    status: Literal["ok", "error"]
    error_type: str | None
    error_message: str | None
    runner: str
    sandbox_name: str | None  # None when the gate refused before creating one
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    timed_out: bool
    duration_ms: float  # the program run only
    total_ms: float  # create, checks, run and removal together
    network_policy: str
    environment_names: tuple[str, ...]
    sandbox_removed: bool

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def new_sandbox_name() -> str:
    return f"{SANDBOX_NAME_PREFIX}{uuid.uuid4().hex[:12]}"


def create_command(limits: SandboxLimits, sandbox_name: str) -> list[str]:
    # No PATH argument after the agent: no workspace and no host directory is mounted.
    return [
        limits.sbx_path, "--no-input", "create",
        "--name", sandbox_name,
        "--no-progress",
        "--deny-network", SBX_ALL_HOSTS_PATTERN,
        "--cpus", str(limits.cpus),
        "--memory", limits.memory,
        SANDBOX_AGENT,
    ]


def network_check_command(limits: SandboxLimits, sandbox_name: str) -> list[str]:
    return [
        limits.sbx_path, "--no-input", "policy", "check", "network",
        "--sandbox", sandbox_name, "--json", limits.network_canary,
    ]


def stage_command(limits: SandboxLimits, sandbox_name: str) -> list[str]:
    program_path = f"{SANDBOX_RUN_DIR}/{SANDBOX_PROGRAM_NAME}"
    script = (
        f"install -d -o 65534 -g 65534 -m 700 {SANDBOX_RUN_DIR} && cat > {program_path} "
        f"&& chown {SANDBOX_UID_GID} {program_path} && chmod 400 {program_path}"
    )
    return [limits.sbx_path, "--no-input", "exec", "-i", "-u", "root", sandbox_name, "sh", "-c", script]


def run_command(limits: SandboxLimits, sandbox_name: str) -> list[str]:
    environment = [f"{name}={value}" for name, value in SANDBOX_ENVIRONMENT]
    return [
        limits.sbx_path, "--no-input", "exec",
        "-u", SANDBOX_USER, "-w", SANDBOX_RUN_DIR, sandbox_name,
        "env", "-i", *environment,
        "timeout", "-k", str(KILL_GRACE_S), _seconds(limits.timeout_s),
        # -I: isolated mode, no PYTHON* variables and no user site. -B: no bytecode files.
        "python3", "-I", "-B", SANDBOX_PROGRAM_NAME,
    ]


def remove_command(limits: SandboxLimits, sandbox_name: str) -> list[str]:
    return [limits.sbx_path, "--no-input", "rm", "--force", sandbox_name]


def list_command(limits: SandboxLimits) -> list[str]:
    return [limits.sbx_path, "--no-input", "ls", "--quiet"]


def _seconds(value: float) -> str:
    return f"{value:g}"


def _one_line(text: str) -> str:
    return " ".join(text.split())[:MAX_ERROR_MESSAGE_CHARS]


def _command_failure(step: str, result: CommandResult) -> str:
    if result.start_error is not None:
        return f"{step}: could not start sbx ({_one_line(result.start_error)})"
    if result.timed_out:
        return f"{step}: sbx did not finish in time"
    return f"{step}: sbx exit {result.exit_code}: {_one_line(result.stderr or result.stdout)}"


def network_is_denied(check: CommandResult) -> bool:
    """True only for an explicit "not allowed" from `sbx policy check`. Anything else is not proof."""
    try:
        verdict = json.loads(check.stdout)
    except (ValueError, TypeError):
        return False
    return isinstance(verdict, dict) and verdict.get("allowed") is False


def classify_run(run: CommandResult, timeout_s: float) -> tuple[str | None, bool]:
    """Decide the typed error of the program run. Returns (error_type or None, timed_out).

    First match wins:

    | # | what the run command returned                                    | error_type          |
    |---|------------------------------------------------------------------|---------------------|
    | 1 | sbx could not be started, or sbx exit 125 (it could not run it)  | sandbox_unavailable |
    | 2 | the host-side timeout fired                                      | timeout             |
    | 3 | output over the cap                                              | output_too_large    |
    | 4 | exit 124 (in-VM `timeout` sent TERM)                             | timeout             |
    | 5 | exit 137 at or after the limit (in-VM `timeout` had to send KILL)| timeout             |
    | 6 | any other non-zero exit                                          | nonzero_exit        |
    | 7 | exit 0                                                           | none                |
    """
    if run.start_error is not None or run.exit_code == EXIT_SBX_COULD_NOT_RUN:
        return ERROR_SANDBOX_UNAVAILABLE, False
    if run.timed_out:
        return ERROR_TIMEOUT, True
    if run.output_truncated:
        return ERROR_OUTPUT_TOO_LARGE, False
    if run.exit_code == EXIT_TIMEOUT_TERM:
        return ERROR_TIMEOUT, True
    if run.exit_code == EXIT_KILLED and run.duration_s >= timeout_s:
        return ERROR_TIMEOUT, True
    if run.exit_code != 0:
        return ERROR_NONZERO_EXIT, False
    return None, False


def run_program(
    program: str,
    runner: str = RUNNER_PYTHON,
    limits: SandboxLimits | None = None,
    command_runner: CommandRunner | None = None,
) -> SandboxResult:
    """Run `program` (Python source, standard library only) in a fresh sandbox and remove it."""
    limits = limits or SandboxLimits()
    command_runner = command_runner or SubprocessRunner()
    started_at = time.monotonic()
    environment_names = tuple(name for name, _ in SANDBOX_ENVIRONMENT)

    def result(
        error_type: str | None,
        error_message: str | None,
        sandbox_name: str | None,
        run: CommandResult | None = None,
        timed_out: bool = False,
        sandbox_removed: bool = True,
    ) -> SandboxResult:
        return SandboxResult(
            status="ok" if error_type is None else "error",
            error_type=error_type,
            error_message=error_message,
            runner=runner,
            sandbox_name=sandbox_name,
            exit_code=run.exit_code if run else None,
            stdout=run.stdout if run else "",
            stderr=run.stderr if run else "",
            stdout_truncated=bool(run and run.stdout_truncated),
            stderr_truncated=bool(run and run.stderr_truncated),
            timed_out=timed_out,
            duration_ms=round(run.duration_s * 1000, 1) if run else 0.0,
            total_ms=round((time.monotonic() - started_at) * 1000, 1),
            network_policy=NETWORK_DENY_ALL,
            environment_names=environment_names,
            sandbox_removed=sandbox_removed,
        )

    if runner not in ALLOWED_RUNNERS:
        return result(
            ERROR_RUNNER_NOT_ALLOWED, f"runner '{runner}' is not allowed; allowed: {list(ALLOWED_RUNNERS)}", None
        )
    program_bytes = len(program.encode("utf-8"))
    if program_bytes > limits.max_program_bytes:
        return result(
            ERROR_PROGRAM_TOO_LARGE,
            f"program is {program_bytes} bytes, over the {limits.max_program_bytes} byte limit",
            None,
        )

    sandbox_name = new_sandbox_name()
    run: CommandResult | None = None
    error_type: str | None = None
    error_message: str | None = None
    timed_out = False
    try:
        created = command_runner.run(
            create_command(limits, sandbox_name), None, limits.create_timeout_s, MANAGEMENT_OUTPUT_BYTES
        )
        if created.exit_code != 0:
            error_type, error_message = ERROR_SANDBOX_UNAVAILABLE, _command_failure("create", created)
        if error_type is None:
            check = command_runner.run(
                network_check_command(limits, sandbox_name),
                None,
                limits.management_timeout_s,
                MANAGEMENT_OUTPUT_BYTES,
            )
            if not network_is_denied(check):
                error_type = ERROR_NETWORK_POLICY_NOT_APPLIED
                error_message = (
                    f"sbx did not confirm that {limits.network_canary} is denied for this sandbox; "
                    "the program was not run"
                )
        if error_type is None:
            staged = command_runner.run(
                stage_command(limits, sandbox_name), program, limits.management_timeout_s, MANAGEMENT_OUTPUT_BYTES
            )
            if staged.exit_code != 0:
                error_type, error_message = ERROR_SANDBOX_UNAVAILABLE, _command_failure("stage", staged)
        if error_type is None:
            run = command_runner.run(
                run_command(limits, sandbox_name),
                None,
                limits.timeout_s + KILL_GRACE_S + HOST_TIMEOUT_MARGIN_S,
                limits.max_output_bytes,
            )
            error_type, timed_out = classify_run(run, limits.timeout_s)
            if error_type == ERROR_SANDBOX_UNAVAILABLE:
                error_message = _command_failure("run", run)
            elif error_type == ERROR_TIMEOUT:
                error_message = f"the program did not finish within {_seconds(limits.timeout_s)} s"
            elif error_type == ERROR_OUTPUT_TOO_LARGE:
                error_message = f"the program wrote more than {limits.max_output_bytes} bytes to one stream"
            elif error_type == ERROR_NONZERO_EXIT:
                error_message = f"the program exited with code {run.exit_code}"
    except Exception as gate_error:  # a broken runner must not leave a sandbox behind
        error_type = ERROR_SANDBOX_UNAVAILABLE
        error_message = _one_line(f"{type(gate_error).__name__}: {gate_error}")
        run = None
    finally:
        sandbox_removed = _remove(command_runner, limits, sandbox_name)

    return result(error_type, error_message, sandbox_name, run, timed_out, sandbox_removed)


def _remove(command_runner: CommandRunner, limits: SandboxLimits, sandbox_name: str) -> bool:
    """Remove the sandbox; one retry. Never raises. Only ever names a sandbox with our prefix.

    Returns True when `sbx rm` succeeded, or when `sbx ls` shows the sandbox does not
    exist (a create that failed leaves nothing to remove).
    """
    if not sandbox_name.startswith(SANDBOX_NAME_PREFIX):
        return False
    for _ in range(2):
        try:
            removed = command_runner.run(
                remove_command(limits, sandbox_name), None, limits.management_timeout_s, MANAGEMENT_OUTPUT_BYTES
            )
            if removed.exit_code == 0:
                return True
            listed = command_runner.run(
                list_command(limits), None, limits.management_timeout_s, MANAGEMENT_OUTPUT_BYTES
            )
            if listed.exit_code == 0 and sandbox_name not in listed.stdout.split():
                return True
        except Exception:
            continue
    return False
