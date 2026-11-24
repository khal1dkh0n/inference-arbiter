"""The dispatch gate. A fake stands in for `sbx`; no sandbox is started here."""

from __future__ import annotations

import sys

import pytest

from inference_arbiter import sandbox
from inference_arbiter.sandbox import CommandResult, SandboxLimits, run_program

from conftest import FakeSandboxRunner, program_output

SECRET_NAMES = ("OPENROUTER_API_KEY", "AWS_SECRET_ACCESS_KEY", "GH_TOKEN")


@pytest.fixture
def fake() -> FakeSandboxRunner:
    return FakeSandboxRunner()


@pytest.fixture
def host_secrets(monkeypatch):
    for name in SECRET_NAMES:
        monkeypatch.setenv(name, f"value-of-{name}")


# ---- the happy path and what is sent to sbx ----


def test_clean_run_returns_a_typed_ok_result_and_removes_the_sandbox(fake):
    fake.script("run", program_output("391\n"))

    result = run_program("print(17 * 23)", command_runner=fake)

    assert fake.verbs() == ["create", "check", "stage", "run", "rm"]
    assert result.ok and result.status == "ok" and result.error_type is None
    assert (result.exit_code, result.stdout, result.stderr) == (0, "391\n", "")
    assert result.timed_out is False
    assert result.duration_ms == 200.0
    assert result.sandbox_name.startswith("inference-arbiter-")
    assert result.network_policy == "deny_all"
    assert result.environment_names == ("PATH", "HOME", "LANG")
    assert result.sandbox_removed is True
    assert fake.existing == set()


def test_every_run_gets_its_own_sandbox_name(fake):
    names = {run_program("print(1)", command_runner=fake).sandbox_name for _ in range(5)}
    assert len(names) == 5


def test_create_mounts_nothing_and_denies_all_network_for_this_sandbox_only(fake):
    result = run_program("print(1)", limits=SandboxLimits(cpus=1, memory="512m"), command_runner=fake)

    create = fake.argv_of("create")
    assert create[-1] == "shell", "no PATH argument may follow the agent: that would mount a workspace"
    assert create[create.index("--deny-network") + 1] == "**"
    assert create[create.index("--name") + 1] == result.sandbox_name
    assert create[create.index("--cpus") + 1] == "1"
    assert create[create.index("--memory") + 1] == "512m"
    for forbidden in ("--mount", "-v", "--volume", "-e", "--env", "--env-file", "--allow-network", "--publish"):
        assert forbidden not in create
    for verb in fake.verbs():
        argv = fake.argv_of(verb)
        assert not any(word in argv for word in ("allow", "deny", "reset", "prune", "settings", "secret", "daemon"))


def test_the_program_runs_as_nobody_with_an_environment_built_from_scratch(fake, host_secrets):
    run_program("print(1)", limits=SandboxLimits(timeout_s=7), command_runner=fake)

    run = fake.argv_of("run")
    assert run[run.index("-u") + 1] == "nobody"
    assert run[run.index("-w") + 1] == "/sandbox-run"
    env_at = run.index("env")
    assert run[env_at : env_at + 5] == ["env", "-i", "PATH=/usr/bin:/bin", "HOME=/sandbox-run", "LANG=C.UTF-8"]
    assert run[env_at + 5 :] == ["timeout", "-k", "2", "7", "python3", "-I", "-B", "program.py"]
    assert "-e" not in run and "--env" not in run and "--env-file" not in run
    for command in fake.commands:
        joined = " ".join(command[1]) + (command[2] or "")
        for name in SECRET_NAMES:
            assert name not in joined and f"value-of-{name}" not in joined


def test_the_program_goes_in_over_stdin_not_on_a_mount_or_the_command_line(fake):
    program = "print('a marker 8c1f')"
    run_program(program, command_runner=fake)

    assert fake.stdin_of("stage") == program
    assert all("8c1f" not in " ".join(command[1]) for command in fake.commands)
    assert fake.argv_of("stage")[fake.argv_of("stage").index("-u") + 1] == "root"
    assert all(fake.stdin_of(verb) is None for verb in ("create", "check", "run", "rm"))


def test_timeouts_and_output_caps_are_passed_to_the_runner(fake):
    limits = SandboxLimits(timeout_s=5, max_output_bytes=1234, create_timeout_s=99, management_timeout_s=11)
    run_program("print(1)", limits=limits, command_runner=fake)

    by_verb = {command[0]: command for command in fake.commands}
    assert by_verb["create"][3] == 99
    assert by_verb["run"][3] == 5 + sandbox.KILL_GRACE_S + sandbox.HOST_TIMEOUT_MARGIN_S
    assert by_verb["run"][4] == 1234
    assert by_verb["rm"][3] == 11


# ---- refusals before any sandbox exists ----


@pytest.mark.parametrize("runner", ["bash", "node", "sh", "", "Python", "python3"])
def test_a_runner_off_the_allow_list_is_refused_without_creating_a_sandbox(fake, runner):
    result = run_program("echo hi", runner=runner, command_runner=fake)

    assert (result.status, result.error_type) == ("error", "runner_not_allowed")
    assert result.sandbox_name is None
    assert fake.commands == []


def test_only_python_is_allowed():
    assert sandbox.ALLOWED_RUNNERS == ("python",)


def test_a_program_over_the_size_cap_is_refused_without_creating_a_sandbox(fake):
    result = run_program("x" * 101, limits=SandboxLimits(max_program_bytes=100), command_runner=fake)

    assert result.error_type == "program_too_large"
    assert fake.commands == []


# ---- failures fail closed and the sandbox is still removed ----


def test_create_failure_is_sandbox_unavailable_and_nothing_runs(fake):
    fake.script("create", CommandResult(exit_code=1, stdout="", stderr="error: daemon not running"))

    result = run_program("print(1)", command_runner=fake)

    assert (result.status, result.error_type) == ("error", "sandbox_unavailable")
    assert "daemon not running" in result.error_message
    assert "stage" not in fake.verbs() and "run" not in fake.verbs()
    assert "rm" in fake.verbs()


def test_missing_sbx_binary_is_sandbox_unavailable(fake):
    missing = CommandResult(exit_code=None, stdout="", stderr="", start_error="FileNotFoundError: sbx")
    fake.script("create", missing)
    fake.script("rm", missing, missing)
    fake.script("ls", missing, missing)

    result = run_program("print(1)", command_runner=fake)

    assert result.error_type == "sandbox_unavailable"
    assert "could not start sbx" in result.error_message
    assert result.sandbox_removed is False


@pytest.mark.parametrize(
    "check",
    [
        CommandResult(exit_code=0, stdout='{"allowed": true}', stderr=""),
        CommandResult(exit_code=1, stdout="not json", stderr=""),
        CommandResult(exit_code=1, stdout="", stderr="error: unknown sandbox"),
        CommandResult(exit_code=0, stdout='{"reason": "no verdict"}', stderr=""),
        CommandResult(exit_code=0, stdout='["allowed", false]', stderr=""),
        CommandResult(exit_code=None, stdout="", stderr="", timed_out=True),
    ],
)
def test_program_is_not_run_unless_sbx_confirms_the_network_is_denied(fake, check):
    fake.script("check", check)

    result = run_program("print(1)", command_runner=fake)

    assert (result.status, result.error_type) == ("error", "network_policy_not_applied")
    assert fake.verbs() == ["create", "check", "rm"]
    assert result.sandbox_removed is True


def test_the_network_check_asks_about_this_sandbox_and_the_configured_canary(fake):
    result = run_program("print(1)", limits=SandboxLimits(network_canary="pypi.org:443"), command_runner=fake)

    check = fake.argv_of("check")
    assert check[check.index("--sandbox") + 1] == result.sandbox_name
    assert check[-1] == "pypi.org:443"
    assert check[check.index("policy") + 1] == "check", "the only policy command the gate may run is the read-only check"


def test_stage_failure_is_sandbox_unavailable_and_the_program_is_not_run(fake):
    fake.script("stage", CommandResult(exit_code=1, stdout="", stderr="install: failed"))

    result = run_program("print(1)", command_runner=fake)

    assert result.error_type == "sandbox_unavailable"
    assert "stage" in result.error_message
    assert fake.verbs() == ["create", "check", "stage", "rm"]


def test_non_zero_exit_keeps_the_output_and_fails(fake):
    fake.script("run", program_output("partial\n", "Traceback: ZeroDivisionError\n", exit_code=1))

    result = run_program("print(1/0)", command_runner=fake)

    assert (result.status, result.error_type, result.exit_code) == ("error", "nonzero_exit", 1)
    assert result.stdout == "partial\n" and "ZeroDivisionError" in result.stderr
    assert result.timed_out is False and result.sandbox_removed is True


def test_in_sandbox_timeout_is_a_timeout(fake):
    fake.script("run", program_output("start\n", exit_code=124))

    result = run_program("while True: pass", command_runner=fake)

    assert (result.error_type, result.timed_out, result.exit_code) == ("timeout", True, 124)
    assert result.sandbox_removed is True


def test_host_side_timeout_is_a_timeout_and_the_sandbox_is_removed(fake):
    fake.script("run", CommandResult(exit_code=None, stdout="", stderr="", timed_out=True, duration_s=30.0))

    result = run_program("while True: pass", command_runner=fake)

    assert (result.error_type, result.timed_out, result.exit_code) == ("timeout", True, None)
    assert fake.verbs()[-1] == "rm" and result.sandbox_removed is True


def test_output_over_the_cap_fails_and_keeps_only_the_capped_part(fake):
    fake.script("run", CommandResult(exit_code=None, stdout="x" * 50, stderr="", stdout_truncated=True))

    result = run_program("print('x' * 10**9)", limits=SandboxLimits(max_output_bytes=50), command_runner=fake)

    assert (result.status, result.error_type) == ("error", "output_too_large")
    assert result.stdout == "x" * 50
    assert (result.stdout_truncated, result.stderr_truncated) == (True, False)
    assert result.sandbox_removed is True


def test_sbx_exit_125_on_the_run_is_sandbox_unavailable_not_a_program_error(fake):
    fake.script("run", program_output("", "error: sandbox not found", exit_code=125))

    assert run_program("print(1)", command_runner=fake).error_type == "sandbox_unavailable"


@pytest.mark.parametrize("failing_verb", ["create", "check", "stage", "run"])
def test_a_runner_that_raises_still_removes_the_sandbox(fake, failing_verb):
    fake.script(failing_verb, RuntimeError("runner broke"))

    result = run_program("print(1)", command_runner=fake)

    assert (result.status, result.error_type) == ("error", "sandbox_unavailable")
    assert "runner broke" in result.error_message
    assert fake.verbs()[-1] == "rm"
    assert result.sandbox_removed is True


def test_removal_is_retried_once(fake):
    # The first rm fails and ls still lists the sandbox, so rm is tried once more.
    fake.script("rm", CommandResult(exit_code=1, stdout="", stderr="busy"))

    result = run_program("print(1)", command_runner=fake)

    assert fake.verbs()[-3:] == ["rm", "ls", "rm"]
    assert result.sandbox_removed is True


def test_removal_that_keeps_failing_is_reported_not_hidden(fake):
    failed = CommandResult(exit_code=1, stdout="", stderr="busy")
    fake.script("rm", failed, failed)
    fake.script("run", program_output("1\n"))

    result = run_program("print(1)", command_runner=fake)

    assert result.sandbox_removed is False
    assert fake.verbs().count("rm") == 2
    assert result.ok, "the program's result stands; the leftover sandbox is reported separately"


def test_a_sandbox_that_was_never_created_counts_as_removed(fake):
    fake.script("create", CommandResult(exit_code=1, stdout="", stderr="no"))
    fake.script("rm", CommandResult(exit_code=1, stdout="", stderr="not found"))

    assert run_program("print(1)", command_runner=fake).sandbox_removed is True


def test_removal_only_ever_names_a_prefixed_sandbox(fake):
    assert sandbox._remove(fake, SandboxLimits(), "gov-lab-agent") is False
    assert sandbox._remove(fake, SandboxLimits(), "warrant-control-ai-agent") is False
    assert fake.commands == []


# ---- classify_run: the full table ----


@pytest.mark.parametrize(
    "run, expected",
    [
        (CommandResult(None, "", "", start_error="FileNotFoundError"), ("sandbox_unavailable", False)),
        (CommandResult(125, "", "sandbox not found"), ("sandbox_unavailable", False)),
        (CommandResult(None, "", "", timed_out=True), ("timeout", True)),
        (CommandResult(None, "x", "", stdout_truncated=True), ("output_too_large", False)),
        (CommandResult(None, "", "x", stderr_truncated=True), ("output_too_large", False)),
        (CommandResult(124, "", ""), ("timeout", True)),
        (CommandResult(137, "", "", duration_s=10.0), ("timeout", True)),
        (CommandResult(137, "", "", duration_s=12.5), ("timeout", True)),
        (CommandResult(137, "", "", duration_s=0.4), ("nonzero_exit", False)),
        (CommandResult(1, "", ""), ("nonzero_exit", False)),
        (CommandResult(127, "", ""), ("nonzero_exit", False)),
        (CommandResult(-9, "", ""), ("nonzero_exit", False)),
        (CommandResult(0, "ok", ""), (None, False)),
        (CommandResult(0, "ok", "a warning on stderr"), (None, False)),
    ],
)
def test_classify_run(run, expected):
    assert sandbox.classify_run(run, timeout_s=10) == expected


def test_error_types_are_split_into_program_and_infrastructure():
    assert set(sandbox.PROGRAM_ERRORS) == {"timeout", "nonzero_exit", "output_too_large"}
    assert not set(sandbox.PROGRAM_ERRORS) & set(sandbox.INFRASTRUCTURE_ERRORS)


# ---- the real subprocess runner, against local commands only (never sbx) ----


def test_host_environment_is_path_and_home_only(host_secrets):
    assert set(sandbox.host_environment()) <= {"PATH", "HOME"}


def test_subprocess_runner_does_not_pass_the_host_environment(host_secrets):
    script = "import os; print(sorted(os.environ))"
    result = sandbox.SubprocessRunner().run([sys.executable, "-I", "-c", script], None, 20, 10_000)

    assert result.exit_code == 0
    for name in SECRET_NAMES:
        assert name not in result.stdout


def test_subprocess_runner_separates_streams_exit_code_and_stdin():
    script = "import sys; sys.stdout.write(sys.stdin.read().upper()); sys.stderr.write('err'); sys.exit(3)"
    result = sandbox.SubprocessRunner().run([sys.executable, "-I", "-c", script], "hello", 20, 10_000)

    assert (result.exit_code, result.stdout, result.stderr) == (3, "HELLO", "err")
    assert not result.timed_out and not result.output_truncated


def test_subprocess_runner_kills_on_timeout():
    result = sandbox.SubprocessRunner().run(
        [sys.executable, "-I", "-c", "import time; print('start', flush=True); time.sleep(60)"], None, 1.0, 10_000
    )

    assert result.timed_out and result.exit_code is None
    assert result.stdout == "start\n"
    assert result.duration_s < 10


def test_subprocess_runner_caps_output_and_stops_the_producer():
    script = "import sys\nwhile True: sys.stdout.write('x' * 65536)"
    result = sandbox.SubprocessRunner().run([sys.executable, "-I", "-c", script], None, 30, 1000)

    assert result.stdout_truncated and not result.stderr_truncated
    assert len(result.stdout) == 1000
    assert result.exit_code is None and not result.timed_out
    assert result.duration_s < 10


def test_subprocess_runner_reports_a_missing_executable():
    result = sandbox.SubprocessRunner().run(["/nonexistent/inference-arbiter-no-such-binary"], None, 5, 1000)

    assert result.exit_code is None and "FileNotFoundError" in result.start_error
