"""Rank passages for a question: local embeddings first, a lexical score as the fallback.

No vector database: the corpus of one request is a few hundred passages at most, so
cosine similarity over a Python list is enough. Vectors are cached on disk by a hash of
the model name and the exact text embedded, so a passage is embedded once however many
questions are asked about it.

Three rankings, chosen in config:
- `hybrid` (default): reciprocal rank fusion of the embedding ranking and the lexical
  ranking. Measured 2026-11-21 on the two spec documents (30 passages, 15 questions,
  17 needed passages, top 6): embeddings alone found 15, BM25 alone 17, the fusion 16.
  Embeddings ranked a Markdown table and an exact-number lookup poorly. The questions
  were written from the documents' own wording, which favours BM25; BM25 misses
  paraphrases by construction, which that sample did not test. The fusion needs no
  score calibration. Re-measure on the open-book eval set before trusting the default.
- `embedding`: cosine similarity only.
- `lexical`: BM25 only, no embedding request.

Which method actually ranked the passages is always recorded: `hybrid`, `embedding`,
`lexical` (asked for, or embeddings were off or failed; the note says why) or `all`
(the corpus fit the token budget and nothing was dropped).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from inference_arbiter.clients import EmbedResult
from inference_arbiter.schemas import CallResult
from inference_arbiter.sources import Passage, estimate_tokens

METHOD_ALL = "all"
METHOD_EMBEDDING = "embedding"
METHOD_LEXICAL = "lexical"
METHOD_HYBRID = "hybrid"

# Reciprocal rank fusion constant: the usual value, large enough that no single
# ranking's first place dominates.
RRF_K = 60

EMBEDDING_CACHE_SUBDIR = "embeddings"
# Texts per embedding request. Small enough that one request stays well inside the
# embedding tier's timeout on a local model.
EMBED_BATCH_SIZE = 64
NOTE_NO_EMBEDDING_TIER = "no embedding tier configured"
NOTE_LEXICAL_CONFIGURED = "lexical ranking configured"

# Standard BM25 constants.
BM25_K1 = 1.5
BM25_B = 0.75

WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by do does for from how in is it of on or that the this to "
    "was what when where which who why will with".split()
)

# texts in -> vectors out (one per text), with the call as telemetry sees it
Embedder = Callable[[list[str]], EmbedResult]


@dataclass
class RetrievalResult:
    passages: list[Passage]  # the selected passages, best first (document order for `all`)
    method: str
    scores: dict[str, float] = field(default_factory=dict)  # passage_id -> score, all passages
    embed_call: CallResult | None = None  # None when no embedding request was made
    note: str | None = None  # why the lexical fallback was used


def tokenize(text: str) -> list[str]:
    return [word for word in WORD.findall(text.lower()) if word not in STOPWORDS]


def lexical_scores(question: str, passages: list[Passage]) -> list[float]:
    """BM25 of each passage (heading included) against the question's words."""
    documents = [tokenize(f"{passage.heading}\n{passage.text}") for passage in passages]
    if not documents:
        return []
    average_length = sum(len(document) for document in documents) / len(documents) or 1.0
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document))

    query_terms = set(tokenize(question))
    scores: list[float] = []
    for document in documents:
        term_counts = Counter(document)
        score = 0.0
        for term in query_terms:
            term_count = term_counts.get(term, 0)
            if term_count == 0:
                continue
            containing = document_frequency[term]
            inverse_frequency = math.log(1 + (len(documents) - containing + 0.5) / (containing + 0.5))
            length_factor = 1 - BM25_B + BM25_B * len(document) / average_length
            score += inverse_frequency * term_count * (BM25_K1 + 1) / (term_count + BM25_K1 * length_factor)
        scores.append(score)
    return scores


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors have different lengths")
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _embedding_cache_path(cache_dir: Path, model: str, text: str) -> Path:
    content_hash = hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()
    return cache_dir / EMBEDDING_CACHE_SUBDIR / f"{content_hash}.json"


def _read_cached_vector(cache_path: Path) -> list[float] | None:
    try:
        vector = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(vector, list) or not vector:
        return None
    if not all(isinstance(value, (int, float)) for value in vector):
        return None
    return vector


def _write_cached_vector(cache_path: Path, vector: list[float]) -> None:
    # The cache is an optimisation: failing to write it must not fail the run.
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(vector), encoding="utf-8")
    except OSError:
        pass


def _merge_embed_calls(calls: list[CallResult]) -> CallResult:
    """Several batch requests reported as one embed call: latency and tokens are summed."""
    if len(calls) == 1:
        return calls[0]
    token_counts = [call.prompt_tokens for call in calls if call.prompt_tokens is not None]
    return calls[-1].model_copy(
        update={
            "latency_ms": round(sum(call.latency_ms for call in calls), 1),
            "prompt_tokens": sum(token_counts) if token_counts else None,
        }
    )


def embed_texts(
    texts: list[str], model: str, embedder: Embedder, cache_dir: str | Path
) -> tuple[list[list[float]] | None, CallResult | None]:
    """Vectors for the texts, from the cache where possible.

    Texts not in the cache are embedded EMBED_BATCH_SIZE at a time, so a corpus of several
    thousand passages does not have to fit one request. Each finished batch is cached, so
    a failure part-way keeps the work done. The batches are reported as one embed call.

    Returns (vectors or None on failure, the embed call or None when all were cached).
    """
    cache_root = Path(cache_dir)
    vectors: list[list[float] | None] = []
    missing_positions: list[int] = []
    for position, text in enumerate(texts):
        cached_vector = _read_cached_vector(_embedding_cache_path(cache_root, model, text))
        vectors.append(cached_vector)
        if cached_vector is None:
            missing_positions.append(position)

    batch_calls: list[CallResult] = []
    for batch_start in range(0, len(missing_positions), EMBED_BATCH_SIZE):
        batch_positions = missing_positions[batch_start : batch_start + EMBED_BATCH_SIZE]
        embed_result = embedder([texts[position] for position in batch_positions])
        batch_calls.append(embed_result.call)
        if not embed_result.call.ok or len(embed_result.vectors) != len(batch_positions):
            return None, _merge_embed_calls(batch_calls)
        for position, vector in zip(batch_positions, embed_result.vectors):
            vectors[position] = vector
            _write_cached_vector(_embedding_cache_path(cache_root, model, texts[position]), vector)

    embed_call = _merge_embed_calls(batch_calls) if batch_calls else None
    return [vector for vector in vectors if vector is not None], embed_call


def _scores_by_id(passages: list[Passage], scores: list[float]) -> dict[str, float]:
    return {passage.passage_id: round(score, 6) for passage, score in zip(passages, scores)}


def reciprocal_rank_fusion(score_lists: list[list[float]]) -> list[float]:
    """Fuse rankings: each passage scores the sum of 1 / (RRF_K + its rank) over the lists."""
    if not score_lists:
        return []
    fused = [0.0] * len(score_lists[0])
    for scores in score_lists:
        order = sorted(range(len(scores)), key=lambda position: scores[position], reverse=True)
        for rank, position in enumerate(order, start=1):
            fused[position] += 1.0 / (RRF_K + rank)
    return fused


def _best_first(
    passages: list[Passage],
    scores: list[float],
    top_k: int,
    retrieval_token_budget: int | None = None,
) -> list[Passage]:
    """The best-ranked passages: the first `top_k`, or as many as fit the token budget.

    With a budget, `top_k` is ignored: passages are taken in rank order until the next
    one would pass the budget. The best passage is always taken.
    """
    # sorted() is stable, so equal scores keep document order.
    ranked = [passage for passage, _ in sorted(zip(passages, scores), key=lambda pair: pair[1], reverse=True)]
    if retrieval_token_budget is None:
        return ranked[:top_k]
    selected: list[Passage] = []
    tokens_used = 0
    for passage in ranked:
        passage_tokens = estimate_tokens(passage.text)
        if selected and tokens_used + passage_tokens > retrieval_token_budget:
            break
        selected.append(passage)
        tokens_used += passage_tokens
    return selected


def select_passages(
    question: str,
    passages: list[Passage],
    *,
    top_k: int,
    full_context_token_budget: int,
    embedder: Embedder | None = None,
    embedding_model: str | None = None,
    query_prefix: str = "",
    document_prefix: str = "",
    cache_dir: str | Path = "data/cache",
    ranking: str = METHOD_HYBRID,
    retrieval_token_budget: int | None = None,
) -> RetrievalResult:
    corpus_tokens = sum(estimate_tokens(passage.text) for passage in passages)
    if corpus_tokens <= full_context_token_budget:
        return RetrievalResult(passages=list(passages), method=METHOD_ALL)

    embed_call: CallResult | None = None
    note: str | None = NOTE_NO_EMBEDDING_TIER
    if ranking == METHOD_LEXICAL:
        note = NOTE_LEXICAL_CONFIGURED
    elif embedder is not None and embedding_model is not None:
        # The passage text alone is embedded: the heading path repeats the document title
        # in every passage and measurably blurred the ranking.
        texts = [query_prefix + question] + [document_prefix + passage.text for passage in passages]
        vectors, embed_call = embed_texts(texts, embedding_model, embedder, cache_dir)
        if vectors is not None:
            try:
                scores = [cosine_similarity(vectors[0], vector) for vector in vectors[1:]]
            except ValueError as shape_error:
                note = f"embedding vectors unusable: {shape_error}"
            else:
                method = METHOD_EMBEDDING
                if ranking == METHOD_HYBRID:
                    method = METHOD_HYBRID
                    scores = reciprocal_rank_fusion([scores, lexical_scores(question, passages)])
                return RetrievalResult(
                    passages=_best_first(passages, scores, top_k, retrieval_token_budget),
                    method=method,
                    scores=_scores_by_id(passages, scores),
                    embed_call=embed_call,
                )
        else:
            failed_type = embed_call.error_type if embed_call is not None else "unknown"
            note = f"embedding call failed ({failed_type})"

    scores = lexical_scores(question, passages)
    return RetrievalResult(
        passages=_best_first(passages, scores, top_k, retrieval_token_budget),
        method=METHOD_LEXICAL,
        scores=_scores_by_id(passages, scores),
        embed_call=embed_call,
        note=note,
    )
