"""Shared fakes: no test in this suite touches the network or a real database."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from inference_arbiter import codetest, docs_server, sandbox, toolgate
from inference_arbiter.clients import ChatCall, EmbedCall, EmbedResult
from inference_arbiter.config import HarnessConfig
from inference_arbiter.router import Harness
from inference_arbiter.schemas import CallResult

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CONFIG_PATH = REPO_ROOT / "config" / "models.yaml"

BASE_CONFIG = {
    "providers": {
        "ollama": {"kind": "ollama_native", "base_url": "http://ollama.invalid"},
        "openrouter": {
            "kind": "openrouter",
            "base_url": "http://openrouter.invalid",
            "api_key_env": "TEST_OPENROUTER_KEY",
        },
    },
    "tiers": {
        "triage": {
            "model": "triage-model",
            "provider": "ollama",
            "timeout_s": 5,
            "think": False,
            "num_ctx": 4096,
            "max_output_tokens": 200,
        },
        "local": {"model": "local-model", "provider": "ollama", "timeout_s": 30, "on_error": "mid"},
        "code": {"model": "code-model", "provider": "ollama", "timeout_s": 30, "on_error": "mid"},
        "local_heavy": {
            "model": "heavy-model",
            "provider": "ollama",
            "timeout_s": 60,
            "effort": "medium",
            "think": "effort",
            "on_error": "mid",
        },
        "mid": {
            "model": "mid-model",
            "provider": "ollama",
            "cloud": True,
            "timeout_s": 30,
            "on_error": "local",
        },
        "frontier": {
            "model": "frontier-model",
            "provider": "openrouter",
            "cloud": True,
            "timeout_s": 60,
            "effort": "medium",
            "on_error": "mid",
            "input_price_per_mtok": 8.0,
            "output_price_per_mtok": 40.0,
        },
    },
    "confidence_threshold": 0.80,
    "fallback_tier": "mid",
    "escalation": {"local": "mid", "code": "mid", "mid": "frontier"},
    "allow_cloud_fallback": True,
    "store_prompt_text": True,
    "triage_max_chars": 200,
}


def ok_result(text: str = "an answer") -> CallResult:
    return CallResult(
        text=text,
        status="ok",
        latency_ms=12.0,
        prompt_tokens=10,
        completion_tokens=5,
        cost_usd=0.0,
        cost_source="local_zero",
        ollama_load_duration_ms=3.0,
    )


def error_result(error_type: str = "http_5xx") -> CallResult:
    return CallResult(status="error", error_type=error_type, error_message="boom", latency_ms=7.0)


def triage_json(label: str, confidence: float, reason: str = "test reason") -> CallResult:
    return ok_result(json.dumps({"label": label, "confidence": confidence, "reason": reason}))


class FakeClient:
    """Scripted client: results are queued per model name and every call is recorded."""

    def __init__(self) -> None:
        self.calls: list[ChatCall] = []
        self._queues: dict[str, list[CallResult]] = {}
        self.embed_calls: list[EmbedCall] = []
        self._embed_queue: list[EmbedResult] = []

    def script(self, model: str, *results: CallResult) -> None:
        self._queues.setdefault(model, []).extend(results)

    def chat(self, call: ChatCall) -> CallResult:
        self.calls.append(call)
        queue = self._queues.get(call.model, [])
        if not queue:
            raise AssertionError(f"unexpected call to model '{call.model}'")
        return queue.pop(0)

    def models_called(self) -> list[str]:
        return [call.model for call in self.calls]

    def script_embed(self, *results: EmbedResult) -> None:
        self._embed_queue.extend(results)

    def embed(self, call: EmbedCall) -> EmbedResult:
        self.embed_calls.append(call)
        if not self._embed_queue:
            raise AssertionError(f"unexpected embed call to model '{call.model}'")
        return self._embed_queue.pop(0)


class FakeTelemetry:
    def __init__(self) -> None:
        self.recorded: list[tuple] = []

        self.sandbox_recorded: list[list] = []
        self.tool_calls_recorded: list[list] = []

    def record(self, run, calls, sandbox_runs=None, tool_calls=None) -> None:
        self.recorded.append((run, list(calls)))
        self.sandbox_recorded.append(list(sandbox_runs or []))
        self.tool_calls_recorded.append(list(tool_calls or []))


class FakeSandboxRunner:
    """Stands in for the `sbx` subprocess. Every command succeeds unless scripted otherwise.

    Commands are recognised by their sbx verb: create, check (policy check), stage (the
    exec that receives the program on stdin), run (the exec that runs it), rm and ls.
    `script("run", result, ...)` queues results for that verb; an exception instance is
    raised instead of returned.
    """

    DENIED = '{"allowed": false, "reason": "Denied by local rule"}'

    def __init__(self) -> None:
        self.commands: list[tuple[str, list[str], str | None, float, int]] = []
        self._queues: dict[str, list] = {}
        self.existing: set[str] = set()

    def script(self, verb: str, *results) -> None:
        self._queues.setdefault(verb, []).extend(results)

    def verbs(self) -> list[str]:
        return [command[0] for command in self.commands]

    def argv_of(self, verb: str, occurrence: int = 0) -> list[str]:
        return [command[1] for command in self.commands if command[0] == verb][occurrence]

    def stdin_of(self, verb: str, occurrence: int = 0) -> str | None:
        return [command[2] for command in self.commands if command[0] == verb][occurrence]

    @staticmethod
    def _verb(argv: list[str]) -> str:
        if "create" in argv:
            return "create"
        if "policy" in argv:
            return "check"
        if "rm" in argv:
            return "rm"
        if "ls" in argv:
            return "ls"
        return "run" if "env" in argv else "stage"

    def run(self, argv, stdin_text, timeout_s, max_output_bytes):
        verb = self._verb(argv)
        self.commands.append((verb, list(argv), stdin_text, timeout_s, max_output_bytes))
        queue = self._queues.get(verb, [])
        scripted = queue.pop(0) if queue else None
        if isinstance(scripted, Exception):
            raise scripted
        if verb == "create" and (scripted is None or scripted.exit_code == 0):
            self.existing.add(argv[argv.index("--name") + 1])
        if verb == "rm" and (scripted is None or scripted.exit_code == 0):
            self.existing.discard(argv[-1])
        if scripted is not None:
            return scripted
        if verb == "check":
            return sandbox.CommandResult(exit_code=1, stdout=self.DENIED, stderr="")
        if verb == "ls":
            return sandbox.CommandResult(exit_code=0, stdout="\n".join(sorted(self.existing)), stderr="")
        return sandbox.CommandResult(exit_code=0, stdout="", stderr="", duration_s=0.2)


def program_output(stdout: str = "", stderr: str = "", exit_code: int | None = 0, **fields) -> sandbox.CommandResult:
    return sandbox.CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr, duration_s=0.2, **fields)


def run_program_on_host(tmp_path, program: str) -> sandbox.SandboxResult:
    """Run one program the way the gate does (`python3 -I -B program.py`), without a sandbox.

    Only for programs built from sources the tests themselves hold.
    """
    program_path = tmp_path / "program.py"
    program_path.write_text(program, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(program_path)], capture_output=True, text=True, timeout=60, cwd=tmp_path
    )
    command = sandbox.CommandResult(
        exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr, duration_s=0.1
    )
    error_type, timed_out = sandbox.classify_run(command, timeout_s=60)
    return sandbox_result(error_type, completed.stdout, completed.returncode, completed.stderr, timed_out)


def run_on_host(tmp_path, solution: str, tests: str, function_name: str = "add") -> sandbox.SandboxResult:
    """Run a solution against a test module through the codetest driver, on the host interpreter."""
    return run_program_on_host(tmp_path, codetest.build_bundle(solution, tests, function_name))


def sandbox_result(
    error_type: str | None = None, stdout: str = "", exit_code: int | None = 0, stderr: str = "",
    timed_out: bool = False,
) -> sandbox.SandboxResult:
    return sandbox.SandboxResult(
        status="ok" if error_type is None else "error", error_type=error_type, error_message=None,
        runner="python", sandbox_name="inference-arbiter-test", exit_code=exit_code,
        stdout=stdout, stderr=stderr, stdout_truncated=False, stderr_truncated=False, timed_out=timed_out,
        duration_ms=1.0, total_ms=2.0, network_policy="deny_all", environment_names=("PATH",),
        sandbox_removed=True,
    )


class FolderToolSession:
    """A ToolSession that answers from a real folder with the server's own index and page
    code, in this process. No subprocess and no `mcp` package: only the transport is faked.

    `on_call` is called before every tool call, so a test can see what was true at that
    moment. `extra_tools` are offered on top of the server's two. `fail_on` maps a tool
    name to an exception or a ToolReply returned instead of the real answer.
    """

    environment_names = ("HOME", "PATH")

    def __init__(self, root: str, limits: docs_server.ServerLimits, start_timeout_s: float = 0.0) -> None:
        self.root = root
        self.limits = limits
        self.index = docs_server.build_index(docs_server.resolve_root(root), limits)
        self.sent: list[tuple[str, dict]] = []
        self.closed = False
        self.close_result: bool | None = True
        self.extra_tools: list[str] = []
        self.fail_on: dict[str, object] = {}
        self.on_call = None

    def list_tools(self, timeout_s: float) -> list[str]:
        return [docs_server.TOOL_LIST_DOCUMENTS, docs_server.TOOL_READ_DOCUMENT, *self.extra_tools]

    def call_tool(self, name: str, arguments: dict, timeout_s: float) -> toolgate.ToolReply:
        if self.closed:
            raise AssertionError("a tool was called after the session was closed")
        if self.on_call is not None:
            self.on_call(name, arguments)
        self.sent.append((name, arguments))
        if name in self.fail_on:
            failure = self.fail_on[name]
            if isinstance(failure, Exception):
                raise failure
            return failure
        if name == docs_server.TOOL_LIST_DOCUMENTS:
            return toolgate.ToolReply(docs_server.encode_result(docs_server.list_documents_result(self.index)))
        try:
            result = docs_server.read_document_result(self.index, self.limits, **arguments)
        except docs_server.DocsRequestError as request_error:
            return toolgate.ToolReply(str(request_error), is_error=True)
        return toolgate.ToolReply(docs_server.encode_result(result))

    def close(self) -> bool | None:
        self.closed = True
        return self.close_result


class ToolSessions:
    """The session opener a test harness uses: remembers every session it opened."""

    def __init__(self) -> None:
        self.opened: list[FolderToolSession] = []
        self.open_error: Exception | None = None
        self.prepare = None

    def __call__(self, root: str, limits: docs_server.ServerLimits, start_timeout_s: float) -> FolderToolSession:
        if self.open_error is not None:
            raise self.open_error
        session = FolderToolSession(root, limits, start_timeout_s)
        if self.prepare is not None:
            self.prepare(session)
        self.opened.append(session)
        return session

    @property
    def last(self) -> FolderToolSession:
        return self.opened[-1]


class HarnessKit:
    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.client = FakeClient()
        self.telemetry = FakeTelemetry()
        self.sandbox_runner = FakeSandboxRunner()
        self.tool_sessions = ToolSessions()
        self.harness = Harness(
            config,
            {"ollama": self.client, "openrouter": self.client},
            self.telemetry,
            sandbox_runner=self.sandbox_runner,
            tool_session_opener=self.tool_sessions,
        )

    @property
    def tool_call_records(self):
        return self.telemetry.tool_calls_recorded[-1]

    @property
    def sandbox_records(self):
        return self.telemetry.sandbox_recorded[-1]

    @property
    def run_record(self):
        return self.telemetry.recorded[-1][0]

    @property
    def call_records(self):
        return self.telemetry.recorded[-1][1]


CHAIN_VERIFIER_TIER = {
    "model": "verify-model",
    "provider": "ollama",
    "role": "verifier",
    "cloud": True,
    "timeout_s": 30,
}


def draft_json(answer: str = "a draft answer", key_claims: list[str] | None = None) -> CallResult:
    claims = key_claims if key_claims is not None else ["claim one"]
    return ok_result(json.dumps({"answer": answer, "key_claims": claims, "assumptions": []}))


def verdict_json(verdict: str, critique: str = "test critique", failed_claims: list[str] | None = None) -> CallResult:
    return ok_result(
        json.dumps({"verdict": verdict, "critique": critique, "failed_claims": failed_claims or []})
    )


def make_chain_config(chain_overrides: dict | None = None, **overrides) -> HarnessConfig:
    """BASE_CONFIG plus a chat verifier tier and a chain block (repair on frontier).

    Defaults to repair_style "repair" so the draft-and-critique repair path stays covered;
    the shipped default, "escalate", has its own tests.
    """
    raw_config = copy.deepcopy(BASE_CONFIG)
    raw_config["tiers"]["verify"] = copy.deepcopy(CHAIN_VERIFIER_TIER)
    raw_config["chain"] = {
        "verifier_tier": "verify",
        "repair_tier": "frontier",
        "repair_style": "repair",
    }
    raw_config["chain"].update(chain_overrides or {})
    raw_config.update(overrides)
    config = HarnessConfig.model_validate(raw_config)
    config.sha256 = "test-config-sha"
    return config


GROUND_CHAT_TIER = {
    "model": "ground-model",
    "provider": "ollama",
    "role": "verifier",
    "timeout_s": 30,
}
EMBED_TIER = {"model": "embed-model", "provider": "ollama", "role": "embedding", "timeout_s": 30}


def make_openbook_config(cache_dir, openbook_overrides: dict | None = None, **overrides) -> HarnessConfig:
    """BASE_CONFIG plus a chat grounding verifier and an openbook block.

    Drafter mid, escalation frontier, no embedding tier (lexical ranking) and a zero
    token budget so retrieval always runs. cache_dir must be a temporary directory.
    """
    raw_config = copy.deepcopy(BASE_CONFIG)
    raw_config["tiers"]["ground_chat"] = copy.deepcopy(GROUND_CHAT_TIER)
    raw_config["tiers"]["embed"] = copy.deepcopy(EMBED_TIER)
    raw_config["openbook"] = {
        "drafter_tier": "mid",
        "escalation_tier": "frontier",
        "grounding_verifier_tier": "ground_chat",
        "full_context_token_budget": 0,
        "cache_dir": str(cache_dir),
    }
    raw_config["openbook"].update(openbook_overrides or {})
    raw_config.update(overrides)
    config = HarnessConfig.model_validate(raw_config)
    config.sha256 = "test-config-sha"
    return config


def make_mcp_config(cache_dir, mcp_overrides: dict | None = None, **overrides) -> HarnessConfig:
    """The open-book test config plus an mcp_docs block planning on mid."""
    return make_openbook_config(cache_dir, mcp_docs={"plan_tier": "mid", **(mcp_overrides or {})}, **overrides)


def make_execute_config(execute_overrides: dict | None = None, **overrides) -> HarnessConfig:
    """BASE_CONFIG plus an execute block answering on mid."""
    raw_config = copy.deepcopy(BASE_CONFIG)
    raw_config["execute"] = {"tier": "mid"}
    raw_config["execute"].update(execute_overrides or {})
    raw_config.update(overrides)
    config = HarnessConfig.model_validate(raw_config)
    config.sha256 = "test-config-sha"
    return config


def execute_draft_json(program: str = "print(6 * 7)", claimed_answer: str | None = "42") -> CallResult:
    return ok_result(json.dumps({"working": "multiply", "program": program, "claimed_answer": claimed_answer}))


def make_config(**overrides) -> HarnessConfig:
    raw_config = copy.deepcopy(BASE_CONFIG)
    raw_config.update(overrides)
    config = HarnessConfig.model_validate(raw_config)
    config.sha256 = "test-config-sha"
    return config


@pytest.fixture
def base_config_dict() -> dict:
    return copy.deepcopy(BASE_CONFIG)


@pytest.fixture
def kit() -> HarnessKit:
    return HarnessKit(make_config())


@pytest.fixture
def chain_kit() -> HarnessKit:
    return HarnessKit(make_chain_config())
