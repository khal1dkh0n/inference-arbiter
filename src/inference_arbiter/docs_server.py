"""Read-only MCP server for one folder of local text documents (mcp mode).

Run as a stdio subprocess by mcp_session.py, one process per run:

    python -I -m inference_arbiter.docs_server --root DIR

It offers two tools and nothing else:

- `list_documents()`: the documents it serves, as path, size in bytes, line count and
  SHA-256 of the text.
- `read_document(path, start_line, max_lines)`: lines of one listed document.

How the folder is confined. This is the server's own code, not the operating system:
the process runs as the calling user and could read any file that user can.

1. The root is resolved once at start. `/`, the home folder and any folder above the
   home folder are refused as a root.
2. The folder is walked once, at start, without following symbolic links. A file is
   indexed only when it is a regular file, is not a symbolic link, has an allowed suffix,
   has a plain name (see NAME_SEGMENT), resolves to a path under the root, is inside the
   size limits and is UTF-8 text. Names starting with a dot are never indexed.
3. The text of every indexed file is read once, at start, and kept in memory. After
   that the server never opens a file again, so a file swapped for a link later changes
   nothing.
4. `read_document` takes a path only as an exact key of that index. `..`, an absolute
   path, a link or any other spelling is simply not a key.

The server has no tool that writes, and it opens files read-only. It ends when its stdin
closes, and on its own after `--max-lifetime-s` whatever the caller does.

The functions that build the index and cut pages are plain Python and import nothing
from `mcp`; only `main` does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

TOOL_LIST_DOCUMENTS = "list_documents"
TOOL_READ_DOCUMENT = "read_document"

ALLOWED_SUFFIXES = (".md", ".txt")
# One path segment: starts with a letter or digit, so never a dot file and never "..".
NAME_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MAX_PATH_DEPTH = 4

SKIP_LINK = "link"
SKIP_NOT_A_FILE = "not_a_file"
SKIP_SUFFIX = "suffix"
SKIP_NAME = "name"
SKIP_OUTSIDE_ROOT = "outside_root"
SKIP_TOO_LARGE = "too_large"
SKIP_NOT_TEXT = "not_text"
SKIP_EMPTY = "empty"
SKIP_LONG_LINE = "long_line"
SKIP_UNREADABLE = "unreadable"
SKIP_OVER_DOCUMENT_LIMIT = "over_document_limit"
SKIP_OVER_TOTAL_LIMIT = "over_total_limit"

STARTED_LINE_PREFIX = "docs-server started pid="
EXIT_BAD_ROOT = 2
EXIT_LIFETIME_OVER = 3


class DocsRootError(Exception):
    """The folder cannot be served. The message says why."""


class DocsRequestError(Exception):
    """A tool call asked for something the index does not hold."""


@dataclass(frozen=True)
class ServerLimits:
    """Limits of one server process. The defaults are the shipped `mcp_docs:` block."""

    max_documents: int = 64
    max_document_bytes: int = 2_000_000
    max_total_bytes: int = 20_000_000
    max_name_chars: int = 120
    max_lines_per_read: int = 400
    # Text bytes of one read result. A document with a longer single line is not indexed.
    max_read_bytes: int = 64_000
    max_lifetime_s: float = 300.0


@dataclass(frozen=True)
class Document:
    path: str  # relative to the root, POSIX separators
    text: str
    lines: tuple[str, ...]
    size_bytes: int  # of the normalised text, UTF-8
    sha256: str  # of the normalised text, UTF-8


@dataclass
class DocumentIndex:
    documents: dict[str, Document] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def resolve_root(root: str | Path) -> Path:
    """The real path of the folder to serve. Refuses `/`, the home folder and its parents."""
    try:
        resolved = Path(root).expanduser().resolve(strict=True)
    except OSError as resolve_error:
        raise DocsRootError(f"cannot serve '{root}': {resolve_error}") from resolve_error
    if not resolved.is_dir():
        raise DocsRootError(f"cannot serve '{root}': it is not a folder")
    home = Path.home().resolve()
    if resolved == home or resolved in home.parents:
        raise DocsRootError(f"cannot serve '{root}': the home folder and folders above it are refused")
    if resolved == Path(resolved.anchor):
        raise DocsRootError(f"cannot serve '{root}': the filesystem root is refused")
    return resolved


def name_is_allowed(relative_path: str, max_name_chars: int) -> bool:
    if not relative_path or len(relative_path) > max_name_chars:
        return False
    segments = relative_path.split("/")
    if len(segments) > MAX_PATH_DEPTH:
        return False
    return all(NAME_SEGMENT.match(segment) for segment in segments)


def normalise_text(raw_bytes: bytes) -> str | None:
    """The text as open-book mode loads a file (sources.py), or None when it is not text."""
    if b"\x00" in raw_bytes:
        return None
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text.replace("\r\n", "\n").strip()


def _read_regular_file(path: Path, max_bytes: int) -> tuple[bytes | None, str | None]:
    """(bytes, None) or (None, skip reason). Never follows a link in the last component."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return None, SKIP_UNREADABLE
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            return None, SKIP_NOT_A_FILE
        if status.st_size > max_bytes:
            return None, SKIP_TOO_LARGE
        chunks: list[bytes] = []
        received = 0
        while True:
            chunk = os.read(descriptor, 65_536)
            if not chunk:
                break
            received += len(chunk)
            if received > max_bytes:
                return None, SKIP_TOO_LARGE
            chunks.append(chunk)
        return b"".join(chunks), None
    except OSError:
        return None, SKIP_UNREADABLE
    finally:
        os.close(descriptor)


def build_index(root: Path, limits: ServerLimits) -> DocumentIndex:
    """Walk `root` once and load every document that passes the rules. `root` is resolved."""
    index = DocumentIndex()
    candidates: list[tuple[str, Path]] = []
    for folder, folder_names, file_names in os.walk(root, followlinks=False):
        folder_path = Path(folder)
        kept_folders = []
        for folder_name in sorted(folder_names):
            if (folder_path / folder_name).is_symlink():
                index.skip(SKIP_LINK)
            elif NAME_SEGMENT.match(folder_name):
                kept_folders.append(folder_name)
            else:
                index.skip(SKIP_NAME)
        folder_names[:] = kept_folders
        for file_name in sorted(file_names):
            file_path = folder_path / file_name
            candidates.append((file_path.relative_to(root).as_posix(), file_path))

    total_bytes = 0
    for relative_path, file_path in sorted(candidates):
        if file_path.is_symlink():
            index.skip(SKIP_LINK)
            continue
        if not name_is_allowed(relative_path, limits.max_name_chars):
            index.skip(SKIP_NAME)
            continue
        if file_path.suffix.lower() not in ALLOWED_SUFFIXES:
            index.skip(SKIP_SUFFIX)
            continue
        try:
            real_path = Path(os.path.realpath(file_path))
        except OSError:
            index.skip(SKIP_UNREADABLE)
            continue
        if real_path != file_path or root not in real_path.parents:
            index.skip(SKIP_OUTSIDE_ROOT)
            continue
        if len(index.documents) >= limits.max_documents:
            index.skip(SKIP_OVER_DOCUMENT_LIMIT)
            continue
        raw_bytes, skip_reason = _read_regular_file(file_path, limits.max_document_bytes)
        if raw_bytes is None:
            index.skip(skip_reason or SKIP_UNREADABLE)
            continue
        text = normalise_text(raw_bytes)
        if text is None:
            index.skip(SKIP_NOT_TEXT)
            continue
        if not text:
            index.skip(SKIP_EMPTY)
            continue
        lines = tuple(text.split("\n"))
        if any(len(line.encode("utf-8")) > limits.max_read_bytes for line in lines):
            index.skip(SKIP_LONG_LINE)
            continue
        encoded = text.encode("utf-8")
        if total_bytes + len(encoded) > limits.max_total_bytes:
            index.skip(SKIP_OVER_TOTAL_LIMIT)
            continue
        total_bytes += len(encoded)
        index.documents[relative_path] = Document(
            path=relative_path,
            text=text,
            lines=lines,
            size_bytes=len(encoded),
            sha256=hashlib.sha256(encoded).hexdigest(),
        )
    return index


def list_documents_result(index: DocumentIndex) -> dict:
    return {
        "documents": [
            {
                "path": document.path,
                "bytes": document.size_bytes,
                "lines": len(document.lines),
                "sha256": document.sha256,
            }
            for document in index.documents.values()
        ],
        "skipped": dict(sorted(index.skipped.items())),
    }


def read_document_result(
    index: DocumentIndex, limits: ServerLimits, path: str, start_line: int, max_lines: int
) -> dict:
    """Lines `start_line` onward of one document: at most `max_lines`, at most `max_read_bytes`.

    At least one line is always returned; no indexed document has a line over the byte cap.
    Joining the `text` of consecutive pages with a newline gives the document's text.
    """
    if not isinstance(path, str) or path not in index.documents:
        raise DocsRequestError("unknown document: use a path exactly as list_documents gives it")
    document = index.documents[path]
    total_lines = len(document.lines)
    if isinstance(start_line, bool) or not isinstance(start_line, int) or not 1 <= start_line <= total_lines:
        raise DocsRequestError(f"start_line must be between 1 and {total_lines}")
    if isinstance(max_lines, bool) or not isinstance(max_lines, int) or max_lines < 1:
        raise DocsRequestError("max_lines must be 1 or more")

    wanted = min(max_lines, limits.max_lines_per_read)
    taken: list[str] = []
    taken_bytes = 0
    for line in document.lines[start_line - 1 : start_line - 1 + wanted]:
        line_bytes = len(line.encode("utf-8")) + 1
        if taken and taken_bytes + line_bytes > limits.max_read_bytes:
            break
        taken.append(line)
        taken_bytes += line_bytes
    return {
        "path": document.path,
        "start_line": start_line,
        "end_line": start_line + len(taken) - 1,
        "total_lines": total_lines,
        "sha256": document.sha256,
        "text": "\n".join(taken),
    }


def encode_result(result: dict) -> str:
    # ensure_ascii=False: an escaped non-ASCII character would be six bytes on the wire.
    return json.dumps(result, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    defaults = ServerLimits()
    parser = argparse.ArgumentParser(prog="inference_arbiter.docs_server", description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="the folder to serve")
    parser.add_argument("--max-documents", type=int, default=defaults.max_documents)
    parser.add_argument("--max-document-bytes", type=int, default=defaults.max_document_bytes)
    parser.add_argument("--max-total-bytes", type=int, default=defaults.max_total_bytes)
    parser.add_argument("--max-name-chars", type=int, default=defaults.max_name_chars)
    parser.add_argument("--max-lines-per-read", type=int, default=defaults.max_lines_per_read)
    parser.add_argument("--max-read-bytes", type=int, default=defaults.max_read_bytes)
    parser.add_argument("--max-lifetime-s", type=float, default=defaults.max_lifetime_s)
    return parser


def limits_from_arguments(arguments: argparse.Namespace) -> ServerLimits:
    return ServerLimits(
        max_documents=arguments.max_documents,
        max_document_bytes=arguments.max_document_bytes,
        max_total_bytes=arguments.max_total_bytes,
        max_name_chars=arguments.max_name_chars,
        max_lines_per_read=arguments.max_lines_per_read,
        max_read_bytes=arguments.max_read_bytes,
        max_lifetime_s=arguments.max_lifetime_s,
    )


def _end_after(lifetime_s: float) -> None:
    """End the process after `lifetime_s`, whatever the event loop is doing."""
    timer = threading.Timer(lifetime_s, lambda: os._exit(EXIT_LIFETIME_OVER))
    timer.daemon = True
    timer.start()


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    limits = limits_from_arguments(arguments)
    try:
        root = resolve_root(arguments.root)
    except DocsRootError as root_error:
        print(f"docs-server: {root_error}", file=sys.stderr, flush=True)
        return EXIT_BAD_ROOT
    _end_after(limits.max_lifetime_s)
    index = build_index(root, limits)

    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    read_only = ToolAnnotations(
        read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
    )
    server = MCPServer(name="inference-arbiter-docs", log_level="ERROR")

    @server.tool(
        name=TOOL_LIST_DOCUMENTS,
        description="List the documents this server holds: path, bytes, lines and sha256 of each.",
        annotations=read_only,
        structured_output=False,
    )
    def list_documents() -> str:
        return encode_result(list_documents_result(index))

    @server.tool(
        name=TOOL_READ_DOCUMENT,
        description="Read lines of one listed document, from start_line, at most max_lines.",
        annotations=read_only,
        structured_output=False,
    )
    def read_document(path: str, start_line: int, max_lines: int) -> str:
        try:
            return encode_result(read_document_result(index, limits, path, start_line, max_lines))
        except DocsRequestError as request_error:
            raise ToolError(str(request_error)) from request_error

    print(f"{STARTED_LINE_PREFIX}{os.getpid()} documents={len(index.documents)}", file=sys.stderr, flush=True)
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
