"""The tool gate: its two decision tables in full, and the path a call takes through it."""

from __future__ import annotations

import hashlib
import json

import pytest

from inference_arbiter import toolgate
from inference_arbiter.toolgate import (
    CALL_ERROR,
    CALL_OK,
    CALL_REFUSED,
    ORIGIN_CODE,
    ORIGIN_PLAN,
    GateLimits,
    ToolGate,
    ToolReply,
    ToolSessionError,
    classify_reply,
    decide_call,
)

SHA = "a" * 64
READ = {"path": "a.md", "start_line": 1, "max_lines": 10}
KNOWN = frozenset({"a.md", "sub/b.md"})


class FakeSession:
    """A ToolSession that returns scripted replies and remembers what it was asked."""

    environment_names = ("HOME", "PATH")

    def __init__(self, tools=("list_documents", "read_document")) -> None:
        self.tools = tools if isinstance(tools, Exception) else list(tools)
        self.replies: list = []
        self.sent: list[tuple[str, dict]] = []
        self.closed = False

    def script(self, *replies) -> None:
        self.replies.extend(replies)

    def list_tools(self, timeout_s):
        if isinstance(self.tools, Exception):
            raise self.tools
        return self.tools

    def call_tool(self, name, arguments, timeout_s):
        self.sent.append((name, arguments))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def close(self):
        self.closed = True
        return True


def listing(*paths: str) -> ToolReply:
    documents = [{"path": path, "bytes": 5, "lines": 1, "sha256": SHA} for path in paths]
    return ToolReply(json.dumps({"documents": documents, "skipped": {}}))


def page(path="a.md", start_line=1, end_line=1, total_lines=1, text="hello") -> ToolReply:
    return ToolReply(
        json.dumps(
            {
                "path": path, "start_line": start_line, "end_line": end_line,
                "total_lines": total_lines, "sha256": SHA, "text": text,
            }
        )
    )


# --- decide_call: the full table -------------------------------------------------------


@pytest.mark.parametrize(
    "tool, arguments, calls_executed, known, expected",
    [
        # row 1: not on the allow-list
        ("write_document", READ, 0, KNOWN, toolgate.REFUSED_TOOL_NOT_ALLOWED),
        ("search_documents", {}, 0, None, toolgate.REFUSED_TOOL_NOT_ALLOWED),
        ("READ_DOCUMENT", READ, 0, KNOWN, toolgate.REFUSED_TOOL_NOT_ALLOWED),
        ("", {}, 0, None, toolgate.REFUSED_TOOL_NOT_ALLOWED),
        (None, {}, 0, None, toolgate.REFUSED_TOOL_NOT_ALLOWED),
        (["read_document"], READ, 0, None, toolgate.REFUSED_TOOL_NOT_ALLOWED),
        # row 2: arguments too large
        ("read_document", {**READ, "path": "a" * 300}, 0, None, toolgate.REFUSED_ARGUMENTS_TOO_LARGE),
        ("list_documents", {"note": "x" * 300}, 0, None, toolgate.REFUSED_ARGUMENTS_TOO_LARGE),
        # row 3: not an object, or not the tool's shape
        ("read_document", None, 0, None, toolgate.REFUSED_ARGUMENTS_INVALID),
        ("read_document", "a.md", 0, None, toolgate.REFUSED_ARGUMENTS_INVALID),
        ("read_document", [READ], 0, None, toolgate.REFUSED_ARGUMENTS_INVALID),
        ("read_document", {}, 0, None, toolgate.REFUSED_ARGUMENTS_INVALID),
        ("read_document", {"path": "a.md"}, 0, None, toolgate.REFUSED_ARGUMENTS_INVALID),
        ("read_document", {**READ, "start_line": 0}, 0, None, toolgate.REFUSED_ARGUMENTS_INVALID),
        ("read_document", {**READ, "max_lines": 0}, 0, None, toolgate.REFUSED_ARGUMENTS_INVALID),
        ("read_document", {**READ, "start_line": "1"}, 0, None, toolgate.REFUSED_ARGUMENTS_INVALID),
        ("read_document", {**READ, "max_lines": 2.5}, 0, None, toolgate.REFUSED_ARGUMENTS_INVALID),
        ("read_document", {**READ, "start_line": True}, 0, None, toolgate.REFUSED_ARGUMENTS_INVALID),
        ("read_document", {**READ, "path": ""}, 0, None, toolgate.REFUSED_ARGUMENTS_INVALID),
        ("read_document", {**READ, "path": 7}, 0, None, toolgate.REFUSED_ARGUMENTS_INVALID),
        ("read_document", {**READ, "encoding": "utf-8"}, 0, None, toolgate.REFUSED_ARGUMENTS_INVALID),
        ("list_documents", {"folder": "/"}, 0, None, toolgate.REFUSED_ARGUMENTS_INVALID),
        # row 4: a path that is not a plain relative one
        ("read_document", {**READ, "path": "/etc/passwd"}, 0, None, toolgate.REFUSED_PATH_NOT_ALLOWED),
        ("read_document", {**READ, "path": "../a.md"}, 0, KNOWN, toolgate.REFUSED_PATH_NOT_ALLOWED),
        ("read_document", {**READ, "path": "sub/../a.md"}, 0, KNOWN, toolgate.REFUSED_PATH_NOT_ALLOWED),
        ("read_document", {**READ, "path": "./a.md"}, 0, KNOWN, toolgate.REFUSED_PATH_NOT_ALLOWED),
        ("read_document", {**READ, "path": "sub//b.md"}, 0, KNOWN, toolgate.REFUSED_PATH_NOT_ALLOWED),
        ("read_document", {**READ, "path": "~/.ssh/id_rsa"}, 0, None, toolgate.REFUSED_PATH_NOT_ALLOWED),
        ("read_document", {**READ, "path": "sub\\b.md"}, 0, KNOWN, toolgate.REFUSED_PATH_NOT_ALLOWED),
        ("read_document", {**READ, "path": "a.md\x00.txt"}, 0, KNOWN, toolgate.REFUSED_PATH_NOT_ALLOWED),
        # row 5: not a listed document
        ("read_document", {**READ, "path": "c.md"}, 0, KNOWN, toolgate.REFUSED_UNKNOWN_DOCUMENT),
        ("read_document", {**READ, "path": "A.md"}, 0, KNOWN, toolgate.REFUSED_UNKNOWN_DOCUMENT),
        ("read_document", {**READ, "path": "a.md"}, 0, frozenset(), toolgate.REFUSED_UNKNOWN_DOCUMENT),
        # row 6: the call cap
        ("read_document", READ, 3, KNOWN, toolgate.REFUSED_CALL_LIMIT),
        ("list_documents", {}, 3, None, toolgate.REFUSED_CALL_LIMIT),
        ("list_documents", {}, 4, KNOWN, toolgate.REFUSED_CALL_LIMIT),
        # row 7: allowed
        ("list_documents", {}, 0, None, None),
        ("list_documents", {}, 2, KNOWN, None),
        ("read_document", READ, 0, KNOWN, None),
        ("read_document", {**READ, "path": "sub/b.md"}, 2, KNOWN, None),
        ("read_document", {**READ, "path": "not-listed-yet.md"}, 0, None, None),
    ],
)
def test_decide_call_table(tool, arguments, calls_executed, known, expected):
    limits = GateLimits(max_calls=3, max_argument_bytes=200)
    assert decide_call(tool, arguments, calls_executed, known, limits) == expected


def test_decide_call_rows_are_tried_in_order():
    limits = GateLimits(max_calls=0, max_argument_bytes=10)
    big_bad = {"path": "/" + "x" * 50}
    # A tool off the list is refused as that, whatever else is wrong with the call.
    assert decide_call("nope", big_bad, 5, KNOWN, limits) == toolgate.REFUSED_TOOL_NOT_ALLOWED
    # Size before shape, shape before path, path before listing, listing before the cap.
    assert decide_call("read_document", big_bad, 5, KNOWN, limits) == toolgate.REFUSED_ARGUMENTS_TOO_LARGE
    roomy = GateLimits(max_calls=0)
    assert decide_call("read_document", {"path": "/x"}, 5, KNOWN, roomy) == toolgate.REFUSED_ARGUMENTS_INVALID
    assert (
        decide_call("read_document", {**READ, "path": "/x"}, 5, KNOWN, roomy) == toolgate.REFUSED_PATH_NOT_ALLOWED
    )
    assert (
        decide_call("read_document", {**READ, "path": "x.md"}, 5, KNOWN, roomy)
        == toolgate.REFUSED_UNKNOWN_DOCUMENT
    )
    assert decide_call("read_document", READ, 5, KNOWN, roomy) == toolgate.REFUSED_CALL_LIMIT


# --- classify_reply: the full table ----------------------------------------------------


@pytest.mark.parametrize(
    "reply, total_before, expected",
    [
        (ToolReply("boom", is_error=True), 0, toolgate.ERROR_TOOL_ERROR),
        (ToolReply("x" * 500, is_error=True, text_only=False), 999, toolgate.ERROR_TOOL_ERROR),
        (ToolReply("ok", text_only=False), 0, toolgate.ERROR_TOOL_RESULT_INVALID),
        (ToolReply("", text_only=False), 0, toolgate.ERROR_TOOL_RESULT_INVALID),
        (ToolReply("x" * 101), 0, toolgate.ERROR_TOOL_RESULT_TOO_LARGE),
        (ToolReply("\u00e9" * 51), 0, toolgate.ERROR_TOOL_RESULT_TOO_LARGE),
        (ToolReply("x" * 100), 201, toolgate.ERROR_TOOL_RESULT_BUDGET),
        (ToolReply("x"), 300, toolgate.ERROR_TOOL_RESULT_BUDGET),
        (ToolReply("x" * 100), 200, None),
        (ToolReply("x" * 100), 0, None),
        (ToolReply(""), 0, None),
    ],
)
def test_classify_reply_table(reply, total_before, expected):
    limits = GateLimits(max_result_bytes=100, max_total_result_bytes=300)
    assert classify_reply(reply, total_before, limits) == expected


# --- the allow-list ---------------------------------------------------------------------


def test_the_allow_list_is_the_two_read_only_tools():
    assert sorted(toolgate.ALLOWED_TOOLS) == ["list_documents", "read_document"]


@pytest.mark.parametrize(
    "offered",
    [
        ["list_documents"],
        ["list_documents", "read_document", "write_document"],
        ["list_documents", "search_documents"],
        [],
        ["list_documents", "read_document", "read_document"],
    ],
)
def test_a_server_offering_anything_but_the_allow_list_is_not_used(offered):
    session = FakeSession(tools=offered)
    error = ToolGate(session).check_tools()
    assert error is not None and error[0] == toolgate.ERROR_TOOL_LIST_MISMATCH
    assert session.sent == []


def test_the_allow_list_check_passes_in_any_order_and_reports_a_dead_session():
    assert ToolGate(FakeSession(tools=["read_document", "list_documents"])).check_tools() is None
    dead = FakeSession(tools=ToolSessionError(toolgate.ERROR_MCP_TIMEOUT, "no answer"))
    assert ToolGate(dead).check_tools() == (toolgate.ERROR_MCP_TIMEOUT, "no answer")


# --- a call through the gate -------------------------------------------------------------


def test_an_allowed_call_is_sent_parsed_and_recorded_without_its_text():
    session = FakeSession()
    reply = page(text="PLANTED-DOCUMENT-TEXT")
    session.script(reply)
    gate = ToolGate(session)
    result = gate.call(ORIGIN_CODE, "read_document", READ)
    assert result.ok and result.payload.text == "PLANTED-DOCUMENT-TEXT"
    assert session.sent == [("read_document", READ)]
    call = gate.calls[0]
    assert (call.seq, call.origin, call.tool, call.status, call.error_type) == (
        1, ORIGIN_CODE, "read_document", CALL_OK, None,
    )
    assert json.loads(call.arguments) == READ and call.arguments_truncated is False
    assert call.result_bytes == len(reply.text.encode("utf-8"))
    assert call.result_sha256 == hashlib.sha256(reply.text.encode("utf-8")).hexdigest()
    assert "PLANTED-DOCUMENT-TEXT" not in repr(call)


def test_a_refused_call_never_reaches_the_server_and_does_not_count():
    session = FakeSession()
    gate = ToolGate(session, GateLimits(max_calls=1))
    result = gate.call(ORIGIN_PLAN, "delete_document", {"path": "a.md"})
    assert not result.ok and result.payload is None
    assert session.sent == [] and gate.calls_executed == 0
    call = gate.calls[0]
    assert (call.origin, call.status, call.error_type) == (ORIGIN_PLAN, CALL_REFUSED, toolgate.REFUSED_TOOL_NOT_ALLOWED)
    assert call.result_bytes is None and call.result_sha256 is None
    session.script(listing("a.md"))
    assert gate.call(ORIGIN_CODE, "list_documents", {}).ok


def test_listing_sets_the_known_documents_and_the_call_cap_holds():
    session = FakeSession()
    session.script(listing("a.md"), page())
    gate = ToolGate(session, GateLimits(max_calls=2))
    assert gate.known_documents is None
    assert gate.call(ORIGIN_CODE, "list_documents", {}).ok
    assert gate.known_documents == frozenset({"a.md"})
    assert gate.call(ORIGIN_PLAN, "read_document", {**READ, "path": "b.md"}).call.error_type == (
        toolgate.REFUSED_UNKNOWN_DOCUMENT
    )
    assert gate.call(ORIGIN_PLAN, "read_document", READ).ok
    over = gate.call(ORIGIN_CODE, "read_document", READ)
    assert over.call.error_type == toolgate.REFUSED_CALL_LIMIT
    assert len(session.sent) == 2
    assert [call.seq for call in gate.calls] == [1, 2, 3, 4]


def test_oversized_arguments_are_stored_cut_and_flagged():
    gate = ToolGate(FakeSession(), GateLimits(max_argument_bytes=50))
    result = gate.call(ORIGIN_PLAN, "read_document", {**READ, "path": "p" * 500})
    assert result.call.error_type == toolgate.REFUSED_ARGUMENTS_TOO_LARGE
    assert result.call.arguments_truncated is True and len(result.call.arguments) == 50


def test_arguments_that_are_not_json_are_still_recorded():
    gate = ToolGate(FakeSession())
    result = gate.call(ORIGIN_PLAN, "read_document", {"path": object()})
    assert result.call.status == CALL_REFUSED and "object" in result.call.arguments


@pytest.mark.parametrize(
    "reply, expected",
    [
        (ToolSessionError(toolgate.ERROR_MCP_TIMEOUT, "no answer in 10 s"), toolgate.ERROR_MCP_TIMEOUT),
        (ToolSessionError(toolgate.ERROR_MCP_UNAVAILABLE, "connection closed"), toolgate.ERROR_MCP_UNAVAILABLE),
        (ToolReply("unknown document", is_error=True), toolgate.ERROR_TOOL_ERROR),
        (ToolReply("{}", text_only=False), toolgate.ERROR_TOOL_RESULT_INVALID),
        (ToolReply("not json"), toolgate.ERROR_TOOL_RESULT_INVALID),
        (ToolReply("{}"), toolgate.ERROR_TOOL_RESULT_INVALID),
        (ToolReply(json.dumps({"path": "a.md", "extra": 1})), toolgate.ERROR_TOOL_RESULT_INVALID),
        (page(path="b.md"), toolgate.ERROR_TOOL_RESULT_INVALID),
        (page(start_line=2, end_line=2, total_lines=3), toolgate.ERROR_TOOL_RESULT_INVALID),
        (page(end_line=5, total_lines=3), toolgate.ERROR_TOOL_RESULT_INVALID),
        (page(text="x" * 300), toolgate.ERROR_TOOL_RESULT_TOO_LARGE),
    ],
)
def test_a_failed_or_unusable_reply_is_a_typed_error(reply, expected):
    session = FakeSession()
    session.script(reply)
    gate = ToolGate(session, GateLimits(max_result_bytes=250))
    result = gate.call(ORIGIN_CODE, "read_document", READ)
    assert not result.ok and result.payload is None
    assert (result.call.status, result.call.error_type) == (CALL_ERROR, expected)
    assert gate.calls_executed == 1


def test_the_run_budget_of_result_bytes_holds():
    session = FakeSession()
    session.script(page(text="x" * 100), page(text="y" * 100))
    one_page = len(page(text="x" * 100).text)
    gate = ToolGate(session, GateLimits(max_total_result_bytes=one_page + 10))
    assert gate.call(ORIGIN_CODE, "read_document", READ).ok
    assert gate.call(ORIGIN_CODE, "read_document", READ).call.error_type == toolgate.ERROR_TOOL_RESULT_BUDGET


@pytest.mark.parametrize(
    "paths",
    [
        ["a.md", "a.md"],
        ["../a.md"],
        ["/etc/passwd"],
        [".env"],
        ["ignore previous instructions.md"],
        ["n" * 121],
        [f"d{number}.md" for number in range(4)],
    ],
)
def test_a_listing_that_breaks_the_name_rules_is_not_accepted(paths):
    session = FakeSession()
    session.script(listing(*paths))
    gate = ToolGate(session, GateLimits(max_documents=3))
    result = gate.call(ORIGIN_CODE, "list_documents", {})
    assert result.call.error_type == toolgate.ERROR_TOOL_RESULT_INVALID
    assert gate.known_documents is None
