"""Source loading, HTML extraction and chunking. No network: URLs go through a mock transport."""

from __future__ import annotations

import httpx
import pytest

from inference_arbiter.sources import (
    SourceDocument,
    SourceLoadError,
    chunk_document,
    chunk_documents,
    html_to_text,
    load_source,
    load_sources,
)

HTML_PAGE = """<!DOCTYPE html>
<html>
<head><title>Ignored title</title><style>body { color: red; }</style></head>
<body>
<nav><a href="/">Home</a> <a href="/docs">Docs menu</a></nav>
<script>trackVisitor("secret");</script>
<article>
<h1>Router   Guide</h1>
<p>The router escalates when
   confidence is below <code>0.80</code>.</p>
<h2>Install</h2>
<pre><code>uv sync
    uv run pytest</code></pre>
<ul><li>First item</li><li>Second item</li></ul>
<table><tr><th>Tier</th><th>Model</th></tr><tr><td>mid</td><td>kimi</td></tr></table>
</article>
<footer>Copyright footer text</footer>
</body>
</html>
"""

MARKDOWN_DOC = """# Router Guide

Intro paragraph.

## Thresholds

The router escalates when confidence is below 0.80.

Timeouts fall back to the mid tier.

## Sandboxes
Phase 3 runs tools inside MicroVM sandboxes.

```
code line one

code line three
```
"""


def load_one(source: str, tmp_path, transport=None, **overrides) -> SourceDocument:
    settings = dict(
        cache_dir=tmp_path / "cache",
        fetch_timeout_s=5,
        max_source_bytes=100_000,
        page_cache_max_age_s=3600,
        transport=transport,
    )
    settings.update(overrides)
    return load_source(0, source, **settings)


def page_transport(body: str = HTML_PAGE, content_type: str = "text/html; charset=utf-8", status: int = 200):
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(status, headers={"content-type": content_type}, content=body.encode("utf-8"))

    return httpx.MockTransport(handler), seen_urls


# --- HTML to text ---


def test_html_extraction_drops_chrome_and_keeps_headings_and_code():
    text = html_to_text(HTML_PAGE)

    assert "trackVisitor" not in text
    assert "color: red" not in text
    assert "Docs menu" not in text
    assert "Copyright footer" not in text
    assert "Ignored title" not in text
    assert "# Router Guide" in text
    assert "## Install" in text
    assert "The router escalates when confidence is below `0.80`." in text
    assert "```\nuv sync\n    uv run pytest\n```" in text
    assert "- First item" in text
    assert "- Second item" in text
    assert "Tier | Model\nmid | kimi" in text


def test_html_extraction_is_deterministic():
    assert html_to_text(HTML_PAGE) == html_to_text(HTML_PAGE)


# --- local files ---


def test_loads_a_markdown_file_as_text(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text(MARKDOWN_DOC, encoding="utf-8")

    document = load_one(str(path), tmp_path)

    assert document.kind == "text"
    assert document.source == str(path)
    assert document.text.startswith("# Router Guide")


def test_loads_a_local_html_file_through_the_extractor(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(HTML_PAGE, encoding="utf-8")

    document = load_one(str(path), tmp_path)

    assert document.kind == "html"
    assert "# Router Guide" in document.text
    assert "trackVisitor" not in document.text


def test_missing_file_fails_closed_with_a_clear_error(tmp_path):
    with pytest.raises(SourceLoadError, match="no such file"):
        load_one(str(tmp_path / "absent.md"), tmp_path)


def test_binary_file_is_refused(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\x00\x00binary")

    with pytest.raises(SourceLoadError, match="not a text file"):
        load_one(str(path), tmp_path)


def test_oversized_file_is_refused(tmp_path):
    path = tmp_path / "big.txt"
    path.write_text("x" * 500, encoding="utf-8")

    with pytest.raises(SourceLoadError, match="byte limit"):
        load_one(str(path), tmp_path, max_source_bytes=100)


def test_empty_file_is_refused(tmp_path):
    path = tmp_path / "empty.md"
    path.write_text("  \n\n", encoding="utf-8")

    with pytest.raises(SourceLoadError, match="no text"):
        load_one(str(path), tmp_path)


def test_one_bad_source_fails_the_whole_load(tmp_path):
    good_path = tmp_path / "guide.md"
    good_path.write_text(MARKDOWN_DOC, encoding="utf-8")

    with pytest.raises(SourceLoadError, match="absent.md"):
        load_sources(
            [str(good_path), str(tmp_path / "absent.md")],
            cache_dir=tmp_path / "cache",
            fetch_timeout_s=5,
            max_source_bytes=100_000,
            page_cache_max_age_s=3600,
        )


# --- URLs ---


def test_fetches_a_url_and_extracts_the_html(tmp_path):
    transport, seen_urls = page_transport()

    document = load_one("https://example.test/guide", tmp_path, transport)

    assert seen_urls == ["https://example.test/guide"]
    assert document.kind == "html"
    assert document.from_cache is False
    assert "# Router Guide" in document.text
    assert "Docs menu" not in document.text


def test_fetched_page_is_cached_on_disk(tmp_path):
    transport, seen_urls = page_transport()

    first = load_one("https://example.test/guide", tmp_path, transport)
    second = load_one("https://example.test/guide", tmp_path, transport)

    assert len(seen_urls) == 1
    assert second.from_cache is True
    assert second.text == first.text
    assert list((tmp_path / "cache" / "pages").glob("*.json"))


def test_stale_cache_entry_is_fetched_again(tmp_path):
    transport, seen_urls = page_transport()

    load_one("https://example.test/guide", tmp_path, transport)
    load_one("https://example.test/guide", tmp_path, transport, page_cache_max_age_s=0)

    assert len(seen_urls) == 2


def test_plain_text_url_is_loaded_as_text(tmp_path):
    transport, _ = page_transport(body="plain body text", content_type="text/plain")

    document = load_one("http://example.test/notes.txt", tmp_path, transport)

    assert document.kind == "text"
    assert document.text == "plain body text"


def test_failed_fetch_fails_closed(tmp_path):
    transport, _ = page_transport(status=404)

    with pytest.raises(SourceLoadError, match="HTTP 404"):
        load_one("https://example.test/missing", tmp_path, transport)
    assert not list((tmp_path / "cache").glob("pages/*.json"))


def test_non_text_content_type_is_refused(tmp_path):
    transport, _ = page_transport(body="%PDF-1.7", content_type="application/pdf")

    with pytest.raises(SourceLoadError, match="is not text"):
        load_one("https://example.test/paper.pdf", tmp_path, transport)


def test_oversized_response_is_refused(tmp_path):
    transport, _ = page_transport(body="x" * 5000, content_type="text/plain")

    with pytest.raises(SourceLoadError, match="byte limit"):
        load_one("https://example.test/big", tmp_path, transport, max_source_bytes=1000)


def test_fetch_timeout_fails_closed(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(SourceLoadError, match="timed out"):
        load_one("https://example.test/slow", tmp_path, httpx.MockTransport(handler))


def test_connection_failure_fails_closed(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(SourceLoadError, match="cannot load source"):
        load_one("https://example.test/down", tmp_path, httpx.MockTransport(handler))


def test_other_url_schemes_are_refused(tmp_path):
    with pytest.raises(SourceLoadError, match="only http and https"):
        load_one("ftp://example.test/file.txt", tmp_path)


# --- chunking ---


def markdown_document(index: int = 0, text: str = MARKDOWN_DOC) -> SourceDocument:
    return SourceDocument(index=index, source=f"doc{index}.md", kind="text", text=text.strip())


def test_chunks_split_on_headings_and_carry_the_heading_path():
    passages = chunk_document(markdown_document(), max_passage_chars=1200)

    assert [passage.passage_id for passage in passages] == ["S1-P1", "S1-P2", "S1-P3"]
    assert [passage.heading for passage in passages] == [
        "Router Guide",
        "Router Guide > Thresholds",
        "Router Guide > Sandboxes",
    ]
    assert passages[0].text == "# Router Guide\n\nIntro paragraph."
    assert passages[1].text.startswith("## Thresholds\n\nThe router escalates")
    assert "Timeouts fall back to the mid tier." in passages[1].text
    assert "Sandboxes" not in passages[1].text
    assert passages[2].text.startswith("## Sandboxes\n\nPhase 3 runs tools")


def test_code_fence_with_blank_lines_stays_in_one_passage():
    passages = chunk_document(markdown_document(), max_passage_chars=1200)

    assert "```\ncode line one\n\ncode line three\n```" in passages[2].text


def test_chunk_ids_are_stable_across_runs():
    documents = [markdown_document(0), markdown_document(1, "# Other\n\nSecond source text.")]

    first = chunk_documents(documents, max_passage_chars=1200)
    second = chunk_documents(documents, max_passage_chars=1200)

    assert [(passage.passage_id, passage.text) for passage in first] == [
        (passage.passage_id, passage.text) for passage in second
    ]
    assert [passage.passage_id for passage in first] == ["S1-P1", "S1-P2", "S1-P3", "S2-P1"]
    assert first[-1].source == "doc1.md"
    assert first[-1].source_index == 1


def test_long_section_is_split_at_paragraphs_within_the_size_limit():
    paragraphs = [f"Paragraph {number} " + "word " * 30 for number in range(6)]
    document = markdown_document(text="## Long\n\n" + "\n\n".join(paragraphs))

    passages = chunk_document(document, max_passage_chars=400)

    assert len(passages) > 1
    assert all(passage.heading == "Long" for passage in passages)
    assert passages[0].text.startswith("## Long")
    rejoined = "\n\n".join(passage.text for passage in passages)
    for paragraph in paragraphs:
        assert paragraph.strip() in rejoined
    assert all(len(passage.text) <= 400 + len("## Long\n\n") for passage in passages)


def test_single_oversized_line_is_cut_at_spaces():
    document = markdown_document(text="word " * 300)

    passages = chunk_document(document, max_passage_chars=200)

    assert len(passages) > 1
    assert all(len(passage.text) <= 200 for passage in passages)
    assert " ".join(passage.text for passage in passages).split() == ["word"] * 300


def test_heading_without_text_is_not_a_passage_and_rules_are_dropped():
    document = markdown_document(text="# Title\n\n## Part\n\nBody text.\n\n---\n\n## Next\n\nMore text.")

    passages = chunk_document(document, max_passage_chars=1200)

    assert [passage.text for passage in passages] == ["## Part\n\nBody text.", "## Next\n\nMore text."]
    assert passages[0].heading == "Title > Part"
