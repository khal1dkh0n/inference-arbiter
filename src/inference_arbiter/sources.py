"""Open-book sources: load files and URLs, turn HTML into text, split into passages.

Everything here is deterministic and uses no model. Loading is fail-closed: a source
that cannot be read, is not text, is too large or is empty raises SourceLoadError, and
the run stops before any model call. A fetched page is never trusted as instructions;
it is only ever handed to a model as quoted data.

HTML is converted with the standard library parser: script, style, navigation and other
page chrome are dropped, headings become Markdown headings and code stays in fenced
blocks, so one chunker serves text, Markdown and HTML alike.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import httpx

URL_SCHEMES = ("http://", "https://")
HTML_SUFFIXES = (".html", ".htm", ".xhtml")
HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")
TEXT_CONTENT_TYPE_PREFIX = "text/"
PAGE_CACHE_SUBDIR = "pages"
FETCH_CHUNK_BYTES = 65_536
MAX_REDIRECTS = 5
CHARS_PER_TOKEN_ESTIMATE = 4

KIND_TEXT = "text"
KIND_HTML = "html"

# Elements whose whole content is dropped: code that is not prose, and page chrome.
HTML_SKIPPED_ELEMENTS = frozenset(
    {
        "script", "style", "noscript", "template", "svg", "head", "nav", "footer",
        "aside", "form", "iframe", "button", "select", "dialog",
    }
)
HTML_HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
# Elements that end the current paragraph when they open or close.
HTML_BLOCK_ELEMENTS = frozenset(
    {
        "p", "div", "section", "article", "main", "header", "ul", "ol", "dl", "dt", "dd",
        "table", "thead", "tbody", "blockquote", "figure", "figcaption", "hr", "details",
        "summary",
    }
)
CODE_FENCE = "```"
# Marks a forced line break (<br>, end of a table row) inside a block being collected.
HTML_LINE_BREAK = "\x1e"

HEADING_LINE = re.compile(r"^(#{1,6})\s+(\S.*)$")
HORIZONTAL_RULE = re.compile(r"^(-{3,}|\*{3,}|_{3,})$")


class SourceLoadError(Exception):
    """A source could not be loaded. The message names the source and the reason."""


@dataclass
class SourceDocument:
    index: int  # position in the request's sources list, from 0
    source: str  # the path or URL exactly as the caller gave it
    kind: str  # text | html
    text: str
    from_cache: bool = False


@dataclass
class Passage:
    passage_id: str  # "S<source number>-P<passage number>", both from 1
    source_index: int
    source: str
    position: int  # order within its source, from 0
    heading: str  # enclosing headings, outermost first, joined with " > "; may be empty
    text: str


def is_url(source: str) -> bool:
    return source.lower().startswith(URL_SCHEMES)


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN_ESTIMATE


# --- HTML to text ------------------------------------------------------------------


class _HtmlTextExtractor(HTMLParser):
    """Collects blocks of text. Whitespace is collapsed everywhere except inside <pre>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._inline: list[str] = []
        self._skip_depth = 0
        self._pre_depth = 0
        self._pre_text: list[str] = []
        self._heading_level: int | None = None
        self._block_prefix = ""

    def _flush(self) -> None:
        lines = []
        for raw_line in "".join(self._inline).split(HTML_LINE_BREAK):
            line = " ".join(raw_line.split()).removesuffix(" |")
            if line:
                lines.append(line)
        self._inline = []
        if not lines:
            self._block_prefix = ""
            return
        block = "\n".join(lines)
        if self._heading_level is not None:
            block = "#" * self._heading_level + " " + " ".join(lines)
        elif self._block_prefix:
            block = self._block_prefix + block
        self._block_prefix = ""
        self.blocks.append(block)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in HTML_SKIPPED_ELEMENTS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "pre":
            if self._pre_depth == 0:
                self._flush()
                self._pre_text = []
            self._pre_depth += 1
            return
        if self._pre_depth:
            return
        if tag in HTML_HEADING_LEVELS:
            self._flush()
            self._heading_level = HTML_HEADING_LEVELS[tag]
        elif tag == "li":
            self._flush()
            self._block_prefix = "- "
        elif tag in HTML_BLOCK_ELEMENTS:
            self._flush()
        elif tag == "br":
            self._inline.append(HTML_LINE_BREAK)
        elif tag == "code":
            self._inline.append("`")

    def handle_endtag(self, tag: str) -> None:
        if tag in HTML_SKIPPED_ELEMENTS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "pre":
            if self._pre_depth:
                self._pre_depth -= 1
                if self._pre_depth == 0:
                    code_text = "".join(self._pre_text).strip("\n")
                    if code_text.strip():
                        self.blocks.append(f"{CODE_FENCE}\n{code_text}\n{CODE_FENCE}")
            return
        if self._pre_depth:
            return
        if tag in HTML_HEADING_LEVELS:
            self._flush()
            self._heading_level = None
        elif tag == "li" or tag in HTML_BLOCK_ELEMENTS:
            self._flush()
        elif tag in ("td", "th"):
            self._inline.append(" | ")
        elif tag == "tr":
            self._inline.append(HTML_LINE_BREAK)
        elif tag == "code":
            self._inline.append("`")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._pre_depth:
            self._pre_text.append(data)
            return
        self._inline.append(data.replace(HTML_LINE_BREAK, " "))

    def close(self) -> None:
        super().close()
        self._flush()


def html_to_text(html: str) -> str:
    extractor = _HtmlTextExtractor()
    extractor.feed(html)
    extractor.close()
    return "\n\n".join(extractor.blocks)


def _looks_like_html(text: str) -> bool:
    head = text.lstrip()[:200].lower()
    return head.startswith("<!doctype html") or head.startswith("<html")


# --- Loading -----------------------------------------------------------------------


def _load_file(source: str, max_source_bytes: int) -> tuple[str, str]:
    path = Path(source).expanduser()
    if not path.is_file():
        raise SourceLoadError(f"cannot load source '{source}': no such file")
    try:
        size_bytes = path.stat().st_size
        if size_bytes > max_source_bytes:
            raise SourceLoadError(
                f"cannot load source '{source}': {size_bytes} bytes is over the "
                f"{max_source_bytes} byte limit"
            )
        raw_bytes = path.read_bytes()
    except OSError as read_error:
        raise SourceLoadError(f"cannot load source '{source}': {read_error}") from read_error

    if b"\x00" in raw_bytes:
        raise SourceLoadError(f"cannot load source '{source}': it is not a text file")
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as decode_error:
        raise SourceLoadError(
            f"cannot load source '{source}': it is not UTF-8 text ({decode_error.reason})"
        ) from decode_error

    is_html = path.suffix.lower() in HTML_SUFFIXES or _looks_like_html(text)
    return (KIND_HTML if is_html else KIND_TEXT), text


def _page_cache_path(cache_dir: Path, url: str) -> Path:
    return cache_dir / PAGE_CACHE_SUBDIR / (hashlib.sha256(url.encode("utf-8")).hexdigest() + ".json")


def _read_page_cache(cache_path: Path, url: str, max_age_s: float) -> tuple[str, str] | None:
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached["url"] != url or time.time() - float(cached["fetched_at_unix"]) > max_age_s:
            return None
        kind, body = cached["kind"], cached["body"]
        if kind not in (KIND_TEXT, KIND_HTML) or not isinstance(body, str):
            return None
        return kind, body
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_page_cache(cache_path: Path, url: str, kind: str, body: str) -> None:
    # The cache is an optimisation: failing to write it must not fail the run.
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"url": url, "kind": kind, "fetched_at_unix": time.time(), "body": body}),
            encoding="utf-8",
        )
    except OSError:
        pass


def _fetch_url(
    url: str, timeout_s: float, max_source_bytes: int, transport: httpx.BaseTransport | None
) -> tuple[str, str]:
    try:
        with httpx.Client(
            timeout=timeout_s,
            transport=transport,
            follow_redirects=True,
            max_redirects=MAX_REDIRECTS,
        ) as http_client:
            with http_client.stream("GET", url) as response:
                if response.status_code >= 400:
                    raise SourceLoadError(f"cannot load source '{url}': HTTP {response.status_code}")
                content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if content_type in HTML_CONTENT_TYPES:
                    kind = KIND_HTML
                elif content_type.startswith(TEXT_CONTENT_TYPE_PREFIX):
                    kind = KIND_TEXT
                else:
                    raise SourceLoadError(
                        f"cannot load source '{url}': content type '{content_type or 'unknown'}' "
                        "is not text"
                    )
                received = bytearray()
                for chunk in response.iter_bytes(FETCH_CHUNK_BYTES):
                    received.extend(chunk)
                    if len(received) > max_source_bytes:
                        raise SourceLoadError(
                            f"cannot load source '{url}': the response is over the "
                            f"{max_source_bytes} byte limit"
                        )
                encoding = response.charset_encoding or "utf-8"
    except httpx.TimeoutException as timeout_error:
        raise SourceLoadError(
            f"cannot load source '{url}': timed out after {timeout_s} s"
        ) from timeout_error
    except httpx.HTTPError as http_error:
        raise SourceLoadError(f"cannot load source '{url}': {http_error!r}") from http_error

    try:
        body = bytes(received).decode(encoding, errors="replace")
    except LookupError:
        body = bytes(received).decode("utf-8", errors="replace")
    return kind, body


def load_source(
    index: int,
    source: str,
    *,
    cache_dir: str | Path,
    fetch_timeout_s: float,
    max_source_bytes: int,
    page_cache_max_age_s: float,
    transport: httpx.BaseTransport | None = None,
) -> SourceDocument:
    from_cache = False
    if is_url(source):
        cache_path = _page_cache_path(Path(cache_dir), source)
        cached = _read_page_cache(cache_path, source, page_cache_max_age_s)
        if cached is not None:
            kind, raw_text = cached
            from_cache = True
        else:
            kind, raw_text = _fetch_url(source, fetch_timeout_s, max_source_bytes, transport)
            _write_page_cache(cache_path, source, kind, raw_text)
    elif "://" in source:
        raise SourceLoadError(f"cannot load source '{source}': only http and https URLs are fetched")
    else:
        kind, raw_text = _load_file(source, max_source_bytes)

    text = html_to_text(raw_text) if kind == KIND_HTML else raw_text.replace("\r\n", "\n")
    text = text.strip()
    if not text:
        raise SourceLoadError(f"cannot load source '{source}': it contains no text")
    return SourceDocument(index=index, source=source, kind=kind, text=text, from_cache=from_cache)


def load_sources(
    sources: list[str],
    *,
    cache_dir: str | Path,
    fetch_timeout_s: float,
    max_source_bytes: int,
    page_cache_max_age_s: float,
    transport: httpx.BaseTransport | None = None,
) -> list[SourceDocument]:
    """Load every source in order. The first one that fails stops the run."""
    return [
        load_source(
            index,
            source,
            cache_dir=cache_dir,
            fetch_timeout_s=fetch_timeout_s,
            max_source_bytes=max_source_bytes,
            page_cache_max_age_s=page_cache_max_age_s,
            transport=transport,
        )
        for index, source in enumerate(sources)
    ]


# --- Chunking ----------------------------------------------------------------------


def _split_blocks(text: str) -> list[str]:
    """Blank-line separated blocks. A fenced code block is one block; a heading line is its own."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False

    def end_block() -> None:
        if current:
            blocks.append("\n".join(current))
            current.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(CODE_FENCE):
            if not in_fence:
                end_block()
                current.append(line)
                in_fence = True
            else:
                current.append(line)
                in_fence = False
                end_block()
            continue
        if in_fence:
            current.append(line)
            continue
        if not stripped or HORIZONTAL_RULE.match(stripped):
            end_block()
            continue
        if HEADING_LINE.match(stripped):
            end_block()
            blocks.append(stripped)
            continue
        current.append(line.rstrip())
    end_block()
    return blocks


def _split_long_block(block: str, max_chars: int) -> list[str]:
    """Cut an oversized block at line ends, then at spaces, never mid-word when avoidable."""
    pieces: list[str] = []
    current = ""
    for line in block.split("\n"):
        was_cut = False
        while len(line) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            cut_at = line.rfind(" ", 0, max_chars)
            if cut_at <= 0:
                cut_at = max_chars
            pieces.append(line[:cut_at])
            line = line[cut_at:].lstrip()
            was_cut = True
        if was_cut and not line:
            continue
        if current and len(current) + 1 + len(line) > max_chars:
            pieces.append(current)
            current = line
        elif current:
            current += "\n" + line
        else:
            current = line
    if current:
        pieces.append(current)
    return pieces


def chunk_document(document: SourceDocument, max_passage_chars: int) -> list[Passage]:
    """Split one document into passages on headings and paragraphs.

    A passage never crosses a heading. Paragraphs under one heading are packed together
    up to max_passage_chars. A heading with no text of its own is not a passage: it is
    carried in the `heading` path of the passages below it. Ids depend only on the
    source's position in the request and the passage's position in the document, so
    the same sources always give the same ids.
    """
    passage_texts: list[tuple[str, str]] = []  # (heading path, text)
    heading_stack: list[tuple[int, str]] = []
    pending_heading_line: str | None = None
    current = ""

    def heading_path() -> str:
        return " > ".join(title for _, title in heading_stack)

    def emit() -> None:
        nonlocal current
        if current:
            passage_texts.append((heading_path(), current))
            current = ""

    for block in _split_blocks(document.text):
        heading_match = HEADING_LINE.match(block)
        if heading_match is not None:
            emit()
            level = len(heading_match.group(1))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_match.group(2).strip()))
            pending_heading_line = block
            continue

        for piece in _split_long_block(block, max_passage_chars):
            if pending_heading_line is not None:
                # The section's first passage opens with its own heading line.
                current = pending_heading_line + "\n\n" + piece
                pending_heading_line = None
            elif current and len(current) + 2 + len(piece) > max_passage_chars:
                emit()
                current = piece
            elif current:
                current += "\n\n" + piece
            else:
                current = piece
    emit()

    return [
        Passage(
            passage_id=f"S{document.index + 1}-P{position + 1}",
            source_index=document.index,
            source=document.source,
            position=position,
            heading=heading,
            text=text,
        )
        for position, (heading, text) in enumerate(passage_texts)
    ]


def chunk_documents(documents: list[SourceDocument], max_passage_chars: int) -> list[Passage]:
    passages: list[Passage] = []
    for document in documents:
        passages.extend(chunk_document(document, max_passage_chars))
    return passages
