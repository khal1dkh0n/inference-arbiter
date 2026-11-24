"""Host-side tool gate: every call to the documents server passes through here.

The documents server (docs_server.py) is a separate process speaking MCP over stdio. The
gate sits between the harness and that process and is plain control flow:

1. At start, `check_tools` asks the server which tools it offers. The names must be
   exactly the allow-list. Anything else ends the run before a single call.
2. Each call goes through `decide_call`, a pure function. A refused call never reaches
   the server.
3. An allowed call is sent with a timeout. Its reply goes through `classify_reply`,
   another pure function, and is then parsed against the gate's own result models.
4. Every call, allowed or refused, becomes one `ToolCall` record with the arguments in
   full and the result as a size and a SHA-256, never as text.

The allow-list is the module constant `ALLOWED_TOOLS`. Arguments are validated against
the gate's own models, not against the schema the server advertises.

Who asked for a call is recorded as its `origin`: `code` for a call the harness makes
itself, `plan` for one a model wrote in a tool plan (mcpdocs.py). Both take the same path.

The server sits behind the `ToolSession` protocol, so tests use a fake and never start a
process. Only mcp_session.py implements it for real.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from inference_arbiter.docs_server import TOOL_LIST_DOCUMENTS, TOOL_READ_DOCUMENT, name_is_allowed

ORIGIN_CODE = "code"
ORIGIN_PLAN = "plan"

CALL_OK = "ok"
CALL_REFUSED = "refused"
CALL_ERROR = "error"

# Refusals: the gate said no and the server was not contacted.
REFUSED_TOOL_NOT_ALLOWED = "tool_not_allowed"
REFUSED_ARGUMENTS_TOO_LARGE = "tool_arguments_too_large"
REFUSED_ARGUMENTS_INVALID = "tool_arguments_invalid"
REFUSED_PATH_NOT_ALLOWED = "path_not_allowed"
REFUSED_UNKNOWN_DOCUMENT = "unknown_document"
REFUSED_CALL_LIMIT = "tool_call_limit"
# Given by the plan handling in mcpdocs.py to planned calls past the plan's own cap.
REFUSED_PLAN_CALL_LIMIT = "plan_call_limit"

# Errors: the call was allowed and the server, or what it returned, failed.
ERROR_MCP_UNAVAILABLE = "mcp_unavailable"
ERROR_MCP_TIMEOUT = "mcp_timeout"
ERROR_TOOL_LIST_MISMATCH = "tool_list_mismatch"
ERROR_TOOL_ERROR = "tool_error"
ERROR_TOOL_RESULT_INVALID = "tool_result_invalid"
ERROR_TOOL_RESULT_TOO_LARGE = "tool_result_too_large"
ERROR_TOOL_RESULT_BUDGET = "tool_result_budget"

MAX_ERROR_MESSAGE_CHARS = 300


class ListDocumentsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadDocumentArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    start_line: StrictInt = Field(ge=1)
    max_lines: StrictInt = Field(ge=1)


# The allow-list: tool name -> the model its arguments must validate against.
ALLOWED_TOOLS: dict[str, type[BaseModel]] = {
    TOOL_LIST_DOCUMENTS: ListDocumentsArguments,
    TOOL_READ_DOCUMENT: ReadDocumentArguments,
}


class DocumentEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    bytes: int = Field(ge=0)
    lines: int = Field(ge=1)
    sha256: str = Field(min_length=64, max_length=64)


class ListDocumentsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[DocumentEntry]
    skipped: dict[str, int] = Field(default_factory=dict)


class ReadDocumentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    start_line: int
    end_line: int
    total_lines: int
    sha256: str
    text: str


RESULT_MODELS: dict[str, type[BaseModel]] = {
    TOOL_LIST_DOCUMENTS: ListDocumentsResult,
    TOOL_READ_DOCUMENT: ReadDocumentResult,
}


@dataclass(frozen=True)
class GateLimits:
    """Limits of one run. The defaults are the shipped `mcp_docs:` block."""

    max_calls: int = 64
    max_argument_bytes: int = 2_000
    max_result_bytes: int = 200_000
    max_total_result_bytes: int = 12_000_000
    call_timeout_s: float = 10.0
    # What the gate accepts in a document listing, whatever the server was told.
    max_documents: int = 64
    max_name_chars: int = 120


@dataclass(frozen=True)
class ToolReply:
    """What the server returned for one call, as the session saw it."""

    text: str
    is_error: bool = False
    # False when the reply held anything other than text content.
    text_only: bool = True


class ToolSessionError(Exception):
    """The session could not do its job. `error_type` is ERROR_MCP_UNAVAILABLE or ERROR_MCP_TIMEOUT."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


class ToolSession(Protocol):
    # The names of the environment variables the server process was given.
    environment_names: tuple[str, ...]

    def list_tools(self, timeout_s: float) -> list[str]: ...

    def call_tool(self, name: str, arguments: dict[str, Any], timeout_s: float) -> ToolReply: ...

    def close(self) -> bool | None:
        """Stop the server. True: it is gone. False: it is still there. None: not known."""
        ...


@dataclass
class ToolCall:
    """One call as telemetry records it. The result text itself is never kept here."""

    seq: int
    origin: str  # code | plan
    tool: str
    arguments: str  # JSON, in full unless arguments_truncated
    arguments_truncated: bool
    status: str  # ok | refused | error
    error_type: str | None
    error_message: str | None
    result_bytes: int | None
    result_sha256: str | None
    is_error: bool
    duration_ms: float


@dataclass
class GateResult:
    call: ToolCall
    # The parsed result of an `ok` call: ListDocumentsResult or ReadDocumentResult.
    payload: BaseModel | None = None

    @property
    def ok(self) -> bool:
        return self.call.status == CALL_OK


def _one_line(text: str) -> str:
    return " ".join(str(text).split())[:MAX_ERROR_MESSAGE_CHARS]


def encode_arguments(arguments: Any) -> str:
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return json.dumps(repr(arguments))


def path_is_plain(path: str) -> bool:
    """A relative path with ordinary segments: no root, no `..`, no backslash, no NUL."""
    if not path or path.startswith(("/", "~")) or "\\" in path or "\x00" in path:
        return False
    return all(segment not in ("", ".", "..") for segment in path.split("/"))


def decide_call(
    tool: Any,
    arguments: Any,
    calls_executed: int,
    known_documents: frozenset[str] | None,
    limits: GateLimits,
) -> str | None:
    """Why this call is refused, or None when it may be sent. First match wins.

    | # | condition                                                        | refusal                  |
    |---|------------------------------------------------------------------|--------------------------|
    | 1 | the tool is not on the allow-list                                | tool_not_allowed         |
    | 2 | the arguments as JSON are over `max_argument_bytes`              | tool_arguments_too_large |
    | 3 | the arguments are not an object, or fail the tool's model        | tool_arguments_invalid   |
    | 4 | a `path` that is absolute or has `..`, `.`, `\\` or NUL in it     | path_not_allowed         |
    | 5 | documents were listed and `path` is not one of them              | unknown_document         |
    | 6 | `max_calls` calls were already sent in this run                  | tool_call_limit          |
    | 7 | none of the above                                                | (allowed)                |

    `known_documents` is None until the server's listing has been read; row 5 cannot
    apply before that. Refused calls do not count towards `max_calls`.
    """
    if not isinstance(tool, str) or tool not in ALLOWED_TOOLS:
        return REFUSED_TOOL_NOT_ALLOWED
    if len(encode_arguments(arguments).encode("utf-8")) > limits.max_argument_bytes:
        return REFUSED_ARGUMENTS_TOO_LARGE
    if not isinstance(arguments, dict):
        return REFUSED_ARGUMENTS_INVALID
    try:
        validated = ALLOWED_TOOLS[tool].model_validate(arguments)
    except ValidationError:
        return REFUSED_ARGUMENTS_INVALID
    path = getattr(validated, "path", None)
    if path is not None:
        if not path_is_plain(path):
            return REFUSED_PATH_NOT_ALLOWED
        if known_documents is not None and path not in known_documents:
            return REFUSED_UNKNOWN_DOCUMENT
    if calls_executed >= limits.max_calls:
        return REFUSED_CALL_LIMIT
    return None


def classify_reply(reply: ToolReply, total_result_bytes_before: int, limits: GateLimits) -> str | None:
    """The typed error of a reply, or None when it may be parsed. First match wins.

    | # | condition                                                    | error                 |
    |---|--------------------------------------------------------------|-----------------------|
    | 1 | the server marked the result as an error                     | tool_error            |
    | 2 | the result holds something other than text                   | tool_result_invalid   |
    | 3 | the text is over `max_result_bytes`                          | tool_result_too_large |
    | 4 | with it the run would pass `max_total_result_bytes`          | tool_result_budget    |
    | 5 | none of the above                                            | (accepted)            |

    An oversized result fails the call; it is never cut to fit.
    """
    if reply.is_error:
        return ERROR_TOOL_ERROR
    if not reply.text_only:
        return ERROR_TOOL_RESULT_INVALID
    result_bytes = len(reply.text.encode("utf-8"))
    if result_bytes > limits.max_result_bytes:
        return ERROR_TOOL_RESULT_TOO_LARGE
    if total_result_bytes_before + result_bytes > limits.max_total_result_bytes:
        return ERROR_TOOL_RESULT_BUDGET
    return None


def check_payload(tool: str, arguments: dict[str, Any], payload: BaseModel, limits: GateLimits) -> str | None:
    """What is wrong with a parsed result, or None. The server is not trusted to keep its own rules."""
    if isinstance(payload, ListDocumentsResult):
        if len(payload.documents) > limits.max_documents:
            return f"the listing holds {len(payload.documents)} documents, over the limit of {limits.max_documents}"
        paths = [entry.path for entry in payload.documents]
        if len(set(paths)) != len(paths):
            return "the listing names a document twice"
        for path in paths:
            if not name_is_allowed(path, limits.max_name_chars):
                return "the listing holds a document name that is not a plain relative path"
        return None
    if isinstance(payload, ReadDocumentResult):
        if payload.path != arguments.get("path") or payload.start_line != arguments.get("start_line"):
            return "the result is for another document or another start line than was asked for"
        if not payload.start_line <= payload.end_line <= payload.total_lines:
            return "the result's line numbers are not consistent"
        return None
    return f"no result model for tool '{tool}'"


@dataclass
class ToolGate:
    session: ToolSession
    limits: GateLimits = field(default_factory=GateLimits)
    calls: list[ToolCall] = field(default_factory=list)
    known_documents: frozenset[str] | None = None
    calls_executed: int = 0
    total_result_bytes: int = 0

    def check_tools(self) -> tuple[str, str] | None:
        """(error_type, message) unless the server offers exactly the allow-list."""
        try:
            offered = self.session.list_tools(self.limits.call_timeout_s)
        except ToolSessionError as session_error:
            return session_error.error_type, _one_line(str(session_error))
        if sorted(offered) != sorted(ALLOWED_TOOLS):
            return (
                ERROR_TOOL_LIST_MISMATCH,
                _one_line(f"the server offers {sorted(offered)}, the allow-list is {sorted(ALLOWED_TOOLS)}"),
            )
        return None

    def _record(
        self,
        origin: str,
        tool: Any,
        arguments: Any,
        status: str,
        error_type: str | None,
        error_message: str | None,
        reply: ToolReply | None = None,
        duration_ms: float = 0.0,
    ) -> ToolCall:
        encoded = encode_arguments(arguments)
        truncated = len(encoded.encode("utf-8")) > self.limits.max_argument_bytes
        if truncated:
            encoded = encoded.encode("utf-8")[: self.limits.max_argument_bytes].decode("utf-8", errors="ignore")
        call = ToolCall(
            seq=len(self.calls) + 1,
            origin=origin,
            tool=tool if isinstance(tool, str) else repr(tool),
            arguments=encoded,
            arguments_truncated=truncated,
            status=status,
            error_type=error_type,
            error_message=_one_line(error_message) if error_message else None,
            result_bytes=len(reply.text.encode("utf-8")) if reply is not None else None,
            result_sha256=hashlib.sha256(reply.text.encode("utf-8")).hexdigest() if reply is not None else None,
            is_error=bool(reply is not None and reply.is_error),
            duration_ms=round(duration_ms, 1),
        )
        self.calls.append(call)
        return call

    def refuse(self, origin: str, tool: Any, arguments: Any, reason: str) -> GateResult:
        """Record a call as refused without sending it."""
        return GateResult(self._record(origin, tool, arguments, CALL_REFUSED, reason, f"refused by the gate: {reason}"))

    def call(self, origin: str, tool: Any, arguments: Any) -> GateResult:
        reason = decide_call(tool, arguments, self.calls_executed, self.known_documents, self.limits)
        if reason is not None:
            return self.refuse(origin, tool, arguments, reason)

        self.calls_executed += 1
        started_at = time.monotonic()
        try:
            reply = self.session.call_tool(tool, arguments, self.limits.call_timeout_s)
        except ToolSessionError as session_error:
            duration_ms = (time.monotonic() - started_at) * 1000
            return GateResult(
                self._record(
                    origin, tool, arguments, CALL_ERROR, session_error.error_type, str(session_error),
                    duration_ms=duration_ms,
                )
            )
        duration_ms = (time.monotonic() - started_at) * 1000

        error_type = classify_reply(reply, self.total_result_bytes, self.limits)
        error_message: str | None = None
        payload: BaseModel | None = None
        if error_type == ERROR_TOOL_ERROR:
            error_message = f"the server reported an error: {reply.text}"
        elif error_type is not None:
            error_message = f"the result was not accepted: {error_type}"
        else:
            self.total_result_bytes += len(reply.text.encode("utf-8"))
            try:
                payload = RESULT_MODELS[tool].model_validate_json(reply.text)
                problem = check_payload(tool, arguments, payload, self.limits)
            except ValidationError as validation_error:
                problem = f"the result does not match the expected shape: {validation_error}"
            if problem is not None:
                error_type, error_message, payload = ERROR_TOOL_RESULT_INVALID, problem, None

        status = CALL_OK if error_type is None else CALL_ERROR
        call = self._record(origin, tool, arguments, status, error_type, error_message, reply, duration_ms)
        if isinstance(payload, ListDocumentsResult):
            self.known_documents = frozenset(entry.path for entry in payload.documents)
        return GateResult(call, payload)
