"""The real `ToolSession`: the documents server as a stdio subprocess, one per run.

This is the only module that imports the `mcp` package. Everything else reaches the
server through toolgate.ToolSession, so the rest of the harness runs without the package
installed and mcp mode then ends with a typed `mcp_unavailable` error.

The `mcp` client is asynchronous and the harness is not, so the session keeps one event
loop on a helper thread (an anyio blocking portal) for as long as the server is alive.

What the server process gets:
- the command line built in `server_command`: this interpreter in isolated mode (`-I`),
  the server module, the root and the limits;
- the environment the `mcp` client builds for a stdio server: HOME, LOGNAME, PATH, SHELL,
  TERM and USER, when set. Nothing else from this process is passed on, so no API key
  loaded from `.env` reaches it. The names are kept on `environment_names`;
- its stderr goes to a temporary file, read back up to STDERR_CAP_BYTES when the session
  closes. It never reaches the harness's own stderr.

No sampling, elicitation or roots handler is registered: the client answers any such
request from the server with an error.

Stopping: `close` leaves the client context, which closes the server's stdin, waits and
then kills its process tree. `close` then checks that the process is gone.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Any

import anyio
from anyio.from_thread import start_blocking_portal
from mcp import Client, StdioServerParameters, stdio_client
from mcp.client.stdio import get_default_environment

from inference_arbiter.docs_server import STARTED_LINE_PREFIX, ServerLimits
from inference_arbiter.toolgate import (
    ERROR_MCP_TIMEOUT,
    ERROR_MCP_UNAVAILABLE,
    ToolReply,
    ToolSessionError,
)

SERVER_MODULE = "inference_arbiter.docs_server"
STDERR_CAP_BYTES = 20_000
# How long `close` waits for the process to disappear after the client has stopped it.
STOP_CHECK_S = 3.0
# The plain initialize handshake. The default mode first probes for a newer protocol and
# only then falls back, which makes a server that never answers take several timeouts to fail.
CLIENT_MODE = "legacy"
STARTED_LINE = re.compile(re.escape(STARTED_LINE_PREFIX) + r"(\d+)")


def server_command(root: str, limits: ServerLimits) -> list[str]:
    """The server's argument list, after the interpreter. -I: isolated mode, so neither the
    working folder nor a PYTHON* variable can put another module in its way."""
    return [
        "-I", "-m", SERVER_MODULE,
        "--root", root,
        "--max-documents", str(limits.max_documents),
        "--max-document-bytes", str(limits.max_document_bytes),
        "--max-total-bytes", str(limits.max_total_bytes),
        "--max-name-chars", str(limits.max_name_chars),
        "--max-lines-per-read", str(limits.max_lines_per_read),
        "--max-read-bytes", str(limits.max_read_bytes),
        "--max-lifetime-s", f"{limits.max_lifetime_s:g}",
    ]


def process_is_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def _describe(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup) and error.exceptions:
        return _describe(error.exceptions[0])
    return f"{type(error).__name__}: {error}"


def _is_timeout(error: BaseException) -> bool:
    if isinstance(error, BaseExceptionGroup):
        return any(_is_timeout(inner) for inner in error.exceptions)
    return isinstance(error, TimeoutError) or "timed out" in str(error).lower()


class McpStdioSession:
    """One server process and the client connected to it. Not reusable after `close`."""

    def __init__(
        self,
        root: str,
        server_limits: ServerLimits | None = None,
        start_timeout_s: float = 20.0,
        command: str | None = None,
        arguments: list[str] | None = None,
    ) -> None:
        limits = server_limits or ServerLimits()
        self.environment_names: tuple[str, ...] = tuple(sorted(get_default_environment()))
        self.server_pid: int | None = None
        self.stderr_text = ""
        self._closed = False
        self._errlog = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
        parameters = StdioServerParameters(
            command=command or sys.executable,
            args=arguments if arguments is not None else server_command(root, limits),
        )
        self._portal_context = start_blocking_portal()
        self._portal = self._portal_context.__enter__()
        self._client_context = None
        self._client: Client | None = None
        try:
            client_context = self._portal.wrap_async_context_manager(
                self._connected(parameters, start_timeout_s)
            )
            self._client = client_context.__enter__()
            self._client_context = client_context
        except BaseException as start_error:
            self._finish()
            error_type = ERROR_MCP_TIMEOUT if _is_timeout(start_error) else ERROR_MCP_UNAVAILABLE
            detail = self.stderr_text.strip().splitlines()[-1] if self.stderr_text.strip() else ""
            raise ToolSessionError(
                error_type, f"the documents server did not start: {_describe(start_error)} {detail}".strip()
            ) from start_error

    @asynccontextmanager
    async def _connected(self, parameters: StdioServerParameters, start_timeout_s: float):
        # The read timeout covers the handshake; every later call passes its own.
        async with Client(
            stdio_client(parameters, errlog=self._errlog),
            read_timeout_seconds=start_timeout_s,
            mode=CLIENT_MODE,
        ) as client:
            yield client

    async def _list_tools(self, timeout_s: float) -> list[str]:
        with anyio.fail_after(timeout_s):
            listing = await self._client.list_tools()
        if listing.next_cursor is not None:
            raise ValueError("the tool listing is paged; a server with that many tools is not ours")
        return [tool.name for tool in listing.tools]

    async def _call_tool(self, name: str, arguments: dict[str, Any], timeout_s: float) -> ToolReply:
        with anyio.fail_after(timeout_s):
            result = await self._client.call_tool(name, arguments, read_timeout_seconds=timeout_s)
        texts = [block.text for block in result.content if getattr(block, "type", None) == "text"]
        return ToolReply(
            text="\n".join(texts),
            is_error=bool(result.is_error),
            text_only=len(texts) == len(result.content) and len(texts) > 0,
        )

    def _run(self, function, *arguments):
        if self._closed or self._client is None:
            raise ToolSessionError(ERROR_MCP_UNAVAILABLE, "the session is closed")
        try:
            return self._portal.call(function, *arguments)
        except BaseException as call_error:
            error_type = ERROR_MCP_TIMEOUT if _is_timeout(call_error) else ERROR_MCP_UNAVAILABLE
            raise ToolSessionError(error_type, _describe(call_error)) from call_error

    def list_tools(self, timeout_s: float) -> list[str]:
        return self._run(self._list_tools, timeout_s)

    def call_tool(self, name: str, arguments: dict[str, Any], timeout_s: float) -> ToolReply:
        return self._run(self._call_tool, name, arguments, timeout_s)

    def _finish(self) -> None:
        """Leave the client context and the event loop, then read the server's stderr."""
        self._closed = True
        if self._client_context is not None:
            try:
                self._client_context.__exit__(None, None, None)
            except BaseException:
                pass
            self._client_context = None
        try:
            self._portal_context.__exit__(None, None, None)
        except BaseException:
            pass
        try:
            self._errlog.seek(0)
            self.stderr_text = self._errlog.read(STDERR_CAP_BYTES)
            self._errlog.close()
        except (OSError, ValueError):
            pass
        started = STARTED_LINE.search(self.stderr_text)
        if started is not None:
            self.server_pid = int(started.group(1))

    def close(self) -> bool | None:
        if not self._closed:
            self._finish()
        if self.server_pid is None:
            return None
        deadline = time.monotonic() + STOP_CHECK_S
        while time.monotonic() < deadline:
            if process_is_gone(self.server_pid):
                return True
            time.sleep(0.02)
        return process_is_gone(self.server_pid)


def open_session(root: str, server_limits: ServerLimits, start_timeout_s: float) -> McpStdioSession:
    return McpStdioSession(root, server_limits, start_timeout_s)
