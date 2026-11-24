"""The real stdio session against the real documents server, on a temporary folder.

No model and no sandbox: each test starts the server as a subprocess of this interpreter.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

pytest.importorskip("mcp")

from inference_arbiter import mcp_session, toolgate  # noqa: E402
from inference_arbiter.docs_server import ServerLimits  # noqa: E402
from inference_arbiter.toolgate import ORIGIN_CODE, ORIGIN_PLAN, ToolGate, ToolSessionError  # noqa: E402

MARKER = "PLANTED-OUTSIDE-SECRET"


@pytest.fixture
def docs(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("# Alpha\n\nfirst line\nsecond line\n", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "b.txt").write_text("beta\n", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text(MARKER, encoding="utf-8")
    (root / "link.md").symlink_to(outside)
    (root / ".env").write_text("KEY=" + MARKER, encoding="utf-8")
    return root


@pytest.fixture
def session(docs):
    opened = mcp_session.McpStdioSession(str(docs))
    yield opened
    opened.close()


def test_round_trip_through_the_gate(session):
    gate = ToolGate(session)
    assert gate.check_tools() is None
    listed = gate.call(ORIGIN_CODE, "list_documents", {})
    assert [entry.path for entry in listed.payload.documents] == ["a.md", "sub/b.txt"]
    assert listed.payload.skipped == {"link": 1, "name": 1}
    read = gate.call(ORIGIN_CODE, "read_document", {"path": "a.md", "start_line": 1, "max_lines": 400})
    assert read.payload.text == "# Alpha\n\nfirst line\nsecond line"
    assert read.payload.sha256 == listed.payload.documents[0].sha256
    assert [call.status for call in gate.calls] == ["ok", "ok"]


def test_the_server_offers_exactly_the_allow_list(session):
    assert sorted(session.list_tools(10)) == sorted(toolgate.ALLOWED_TOOLS)


@pytest.mark.parametrize("path", ["link.md", ".env", "../outside.md", "sub/../a.md"])
def test_the_server_itself_refuses_what_is_not_in_its_index(session, path):
    """Asked directly, without the gate in front: the server's own check holds."""
    reply = session.call_tool("read_document", {"path": path, "start_line": 1, "max_lines": 10}, 10)
    assert reply.is_error is True
    assert MARKER not in reply.text


def test_an_absolute_path_is_refused_by_the_server(session, docs):
    reply = session.call_tool(
        "read_document", {"path": str(docs / "a.md"), "start_line": 1, "max_lines": 10}, 10
    )
    assert reply.is_error is True


def test_a_tool_the_server_does_not_have_is_an_error_reply(session):
    assert session.call_tool("write_document", {"path": "a.md", "text": "x"}, 10).is_error is True


def test_a_planned_call_outside_the_folder_is_refused_before_the_server(session):
    gate = ToolGate(session)
    gate.call(ORIGIN_CODE, "list_documents", {})
    refused = gate.call(ORIGIN_PLAN, "read_document", {"path": "link.md", "start_line": 1, "max_lines": 5})
    assert (refused.call.status, refused.call.error_type) == ("refused", toolgate.REFUSED_UNKNOWN_DOCUMENT)
    assert gate.calls_executed == 1


def test_close_stops_the_server_and_keeps_its_stderr(docs):
    session = mcp_session.McpStdioSession(str(docs))
    session.list_tools(10)
    assert session.close() is True
    assert session.server_pid is not None and mcp_session.process_is_gone(session.server_pid)
    assert "documents=2" in session.stderr_text
    with pytest.raises(ToolSessionError) as closed:
        session.list_tools(10)
    assert closed.value.error_type == toolgate.ERROR_MCP_UNAVAILABLE
    assert session.close() is True


def test_the_recorded_environment_names_are_the_fixed_ones(session, monkeypatch):
    assert set(session.environment_names) <= {"HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"}


def test_a_child_process_sees_only_the_fixed_names(docs, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "PLANTED-KEY-VALUE")
    probe = "import os, sys; sys.stderr.write('names=' + ','.join(sorted(os.environ)) + chr(10))"
    try:
        mcp_session.McpStdioSession(str(docs), start_timeout_s=5, arguments=["-I", "-c", probe])
    except ToolSessionError as start_error:
        message = str(start_error)
    names_line = message[message.index("names=") + len("names="):]
    seen = set(names_line.split()[0].split(","))
    # macOS adds __CF_USER_TEXT_ENCODING to every process; nothing of ours is passed on.
    assert seen - {"__CF_USER_TEXT_ENCODING", "LC_CTYPE"} <= {"HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"}
    assert "OPENROUTER_API_KEY" not in seen


def test_a_server_that_never_answers_times_out_and_is_killed(docs):
    started_at = time.monotonic()
    marker = f"hung-{os.getpid()}-{time.monotonic_ns()}"
    script = f"import sys, time; sys.stderr.write('docs-server started pid=' + str(__import__('os').getpid()) + ' {marker}' + chr(10)); sys.stderr.flush(); time.sleep(120)"
    with pytest.raises(ToolSessionError) as hung:
        mcp_session.McpStdioSession(str(docs), start_timeout_s=1, arguments=["-I", "-c", script])
    assert hung.value.error_type == toolgate.ERROR_MCP_TIMEOUT
    assert time.monotonic() - started_at < 30
    pid = int(str(hung.value).split("pid=")[1].split()[0])
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not mcp_session.process_is_gone(pid):
        time.sleep(0.05)
    assert mcp_session.process_is_gone(pid)


def test_a_refused_root_is_a_typed_start_error():
    with pytest.raises(ToolSessionError) as refused:
        mcp_session.McpStdioSession("/", start_timeout_s=10)
    assert refused.value.error_type == toolgate.ERROR_MCP_UNAVAILABLE
    assert "refused" in str(refused.value)


def test_the_server_ends_on_its_own_after_its_lifetime(docs):
    session = mcp_session.McpStdioSession(str(docs), ServerLimits(max_lifetime_s=1.0))
    try:
        session.list_tools(10)
        time.sleep(1.6)
        with pytest.raises(ToolSessionError):
            session.list_tools(5)
    finally:
        session.close()


def test_the_command_line_is_isolated_and_carries_the_limits(docs):
    command = mcp_session.server_command(str(docs), ServerLimits(max_documents=3))
    assert command[:3] == ["-I", "-m", "inference_arbiter.docs_server"]
    assert command[command.index("--max-documents") + 1] == "3"
    assert sys.executable
