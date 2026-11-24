"""Fetch the open-book eval corpus and check it against the manifest.

    uv run python scripts/fetch_corpus.py            fetch missing files, then verify every hash
    uv run python scripts/fetch_corpus.py --check    verify only, no network
    uv run python scripts/fetch_corpus.py --refetch  fetch every remote file again
    uv run python scripts/fetch_corpus.py --pin      write the hashes found on disk into the manifest

The manifest (eval/openbook/corpus.yaml) lists every source document with a SHA-256 of its
bytes. Documents with a url are third-party pages: they are fetched into data/corpus/
(git-ignored) and are never committed. Documents without a url are files already in this
repository. Questions and evidence quotes are written against one exact corpus, so a file
that is missing or whose hash differs makes the eval refuse to run. --pin accepts new
content; after pinning, every evidence quote in eval/openbook/questions.yaml has to be
checked again (tests/test_openbook_eval.py does that for the files present).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = Path("eval/openbook/corpus.yaml")
# A generic client name: no project, user or machine detail leaves this machine.
USER_AGENT = "corpus-fetch/1.0 (python-httpx)"
FETCH_TIMEOUT_S = 30.0

PROBLEM_MISSING = "missing"
PROBLEM_HASH_MISMATCH = "hash_mismatch"
PROBLEM_NOT_PINNED = "not_pinned"

MANIFEST_HEADER = """\
# Open-book eval corpus manifest. Maintained by scripts/fetch_corpus.py (--pin rewrites it).
# Documents with a url are third-party pages, fetched into data/corpus/ (git-ignored) and
# never committed. Documents without a url are files in this repository.
# sha256 is over the file bytes. The eval refuses to run when a file is missing or differs.
"""


class CorpusError(Exception):
    """The corpus on disk is not the corpus the manifest describes."""


def sha256_of_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    parsed = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(parsed, dict) or not isinstance(parsed.get("documents"), list):
        raise CorpusError(f"{manifest_path} has no 'documents' list")
    seen_ids: set[str] = set()
    for document in parsed["documents"]:
        if "id" not in document or "path" not in document:
            raise CorpusError(f"{manifest_path}: every document needs an id and a path")
        if document["id"] in seen_ids:
            raise CorpusError(f"{manifest_path}: duplicate document id '{document['id']}'")
        seen_ids.add(document["id"])
    return parsed


def document_path(document: dict[str, Any], root: str | Path) -> Path:
    return Path(root) / document["path"]


def find_problems(manifest: dict[str, Any], root: str | Path) -> list[dict[str, str]]:
    """One entry per document that is missing, unpinned or differs from its pinned hash."""
    problems: list[dict[str, str]] = []
    for document in manifest["documents"]:
        path = document_path(document, root)
        if not path.is_file():
            problems.append({"id": document["id"], "path": document["path"], "problem": PROBLEM_MISSING})
            continue
        pinned_hash = document.get("sha256")
        if not pinned_hash:
            problems.append({"id": document["id"], "path": document["path"], "problem": PROBLEM_NOT_PINNED})
            continue
        if sha256_of_bytes(path.read_bytes()) != pinned_hash:
            problems.append(
                {"id": document["id"], "path": document["path"], "problem": PROBLEM_HASH_MISMATCH}
            )
    return problems


def describe_problems(problems: list[dict[str, str]], manifest_path: str | Path) -> str:
    explanations = {
        PROBLEM_MISSING: "file is missing",
        PROBLEM_HASH_MISMATCH: "content differs from the hash in the manifest",
        PROBLEM_NOT_PINNED: "manifest has no sha256 for it",
    }
    lines = [f"open-book corpus does not match {manifest_path}:"]
    for problem in problems:
        lines.append(f"  {problem['id']} ({problem['path']}): {explanations[problem['problem']]}")
    lines.append(
        "Run `uv run python scripts/fetch_corpus.py` to fetch missing files. A file that differs "
        "has changed since the questions were written: results would not be comparable, so the "
        "eval does not run. To accept the new content run fetch_corpus.py --pin and re-check the "
        "evidence quotes in eval/openbook/questions.yaml."
    )
    return "\n".join(lines)


def verify_corpus(manifest_path: str | Path, root: str | Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """Return {document id: {path, sha256, text, url}} or raise CorpusError naming every problem."""
    manifest = load_manifest(manifest_path)
    problems = find_problems(manifest, root)
    if problems:
        raise CorpusError(describe_problems(problems, manifest_path))
    corpus: dict[str, dict[str, Any]] = {}
    for document in manifest["documents"]:
        path = document_path(document, root)
        corpus[document["id"]] = {
            "path": str(path),
            "sha256": document["sha256"],
            "url": document.get("url"),
            "text": path.read_text(encoding="utf-8"),
        }
    return corpus


def corpus_fingerprint(corpus: dict[str, dict[str, Any]]) -> str:
    """One hash over every document id and content hash, to tie a result file to a corpus."""
    joined = "\n".join(f"{document_id}:{corpus[document_id]['sha256']}" for document_id in sorted(corpus))
    return sha256_of_bytes(joined.encode("utf-8"))


def fetch_document(client: httpx.Client, url: str) -> bytes:
    response = client.get(url)
    response.raise_for_status()
    return response.content


def fetch_documents(
    manifest: dict[str, Any],
    root: str | Path,
    client: httpx.Client,
    refetch: bool = False,
    pause_s: float = 0.0,
) -> list[str]:
    """Download remote documents that are absent (or all of them with refetch). Returns log lines.

    pause_s is slept after each download, so a manifest of several hundred pages does not
    arrive at the server as one burst.
    """
    log_lines: list[str] = []
    for document in manifest["documents"]:
        url = document.get("url")
        if not url:
            continue
        path = document_path(document, root)
        if path.is_file() and not refetch:
            log_lines.append(f"kept     {document['id']}: {document['path']} already present")
            continue
        try:
            content = fetch_document(client, url)
        except httpx.HTTPError as fetch_error:
            log_lines.append(f"FAILED   {document['id']}: {url}: {fetch_error!r}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        log_lines.append(f"fetched  {document['id']}: {len(content)} bytes from {url}")
        if pause_s > 0:
            time.sleep(pause_s)
    return log_lines


def pin_manifest(manifest_path: str | Path, root: str | Path) -> list[str]:
    """Write the hash of every file present into the manifest. Returns the ids that changed."""
    manifest = load_manifest(manifest_path)
    changed: list[str] = []
    for document in manifest["documents"]:
        path = document_path(document, root)
        if not path.is_file():
            continue
        found_hash = sha256_of_bytes(path.read_bytes())
        if document.get("sha256") != found_hash:
            document["sha256"] = found_hash
            document["pinned_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            changed.append(document["id"])
    if changed:
        body = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True, width=120)
        Path(manifest_path).write_text(MANIFEST_HEADER + body, encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify the open-book eval corpus")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--root", default=str(REPO_ROOT), help="directory that document paths are relative to")
    parser.add_argument("--check", action="store_true", help="verify only, no network")
    parser.add_argument("--refetch", action="store_true", help="download remote files even when present")
    parser.add_argument("--pin", action="store_true", help="record the hashes found on disk in the manifest")
    parser.add_argument("--pause-s", type=float, default=0.0, help="seconds to wait after each download")
    arguments = parser.parse_args(argv)

    try:
        manifest = load_manifest(arguments.manifest)
    except (OSError, CorpusError) as manifest_error:
        print(f"error: {manifest_error}", file=sys.stderr)
        return 2

    if not arguments.check:
        with httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=FETCH_TIMEOUT_S, follow_redirects=True
        ) as client:
            for log_line in fetch_documents(
                manifest, arguments.root, client, refetch=arguments.refetch, pause_s=arguments.pause_s
            ):
                print(log_line)

    if arguments.pin:
        changed = pin_manifest(arguments.manifest, arguments.root)
        print(f"pinned {len(changed)} document(s): {', '.join(changed) or 'none changed'}")
        manifest = load_manifest(arguments.manifest)

    problems = find_problems(manifest, arguments.root)
    if problems:
        print(describe_problems(problems, arguments.manifest), file=sys.stderr)
        return 1
    print(f"corpus ok: {len(manifest['documents'])} documents match {arguments.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
