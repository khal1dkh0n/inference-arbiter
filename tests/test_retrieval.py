"""Passage ranking: embeddings, the lexical fallback, fusion, caching. No network."""

from __future__ import annotations

import json

import httpx
import pytest

from inference_arbiter.clients import EmbedCall, EmbedResult, OllamaNativeClient
from inference_arbiter.retrieval import (
    cosine_similarity,
    lexical_scores,
    reciprocal_rank_fusion,
    select_passages,
)
from inference_arbiter.schemas import CallResult
from inference_arbiter.sources import Passage

from conftest import error_result


def make_passage(number: int, text: str, heading: str = "") -> Passage:
    return Passage(
        passage_id=f"S1-P{number}",
        source_index=0,
        source="doc.md",
        position=number - 1,
        heading=heading,
        text=text,
    )


PASSAGES = [
    make_passage(1, "Sandboxes isolate tool execution inside MicroVMs."),
    make_passage(2, "The router escalates when confidence is below the threshold of 0.80."),
    make_passage(3, "Telemetry rows are written to SQLite after every run."),
]

QUESTION = "What confidence threshold makes the router escalate?"


class FakeEmbedder:
    """Returns the scripted vector for each text, keyed by a substring of it."""

    def __init__(self, vectors_by_fragment: dict[str, list[float]]) -> None:
        self.vectors_by_fragment = vectors_by_fragment
        self.batches: list[list[str]] = []

    def __call__(self, texts: list[str]) -> EmbedResult:
        self.batches.append(list(texts))
        vectors = []
        for text in texts:
            matches = [vector for fragment, vector in self.vectors_by_fragment.items() if fragment in text]
            assert len(matches) == 1, text
            vectors.append(matches[0])
        return EmbedResult(CallResult(status="ok", latency_ms=5.0, prompt_tokens=20), vectors)


def embedder_preferring_telemetry() -> FakeEmbedder:
    # The embedding ranking is deliberately different from the lexical one: 3, 1, 2.
    return FakeEmbedder(
        {
            "confidence threshold makes": [1.0, 0.0],
            "Telemetry rows": [0.9, 0.1],
            "Sandboxes isolate": [0.5, 0.5],
            "router escalates": [0.0, 1.0],
        }
    )


def select(tmp_path, **overrides):
    settings = dict(
        top_k=2,
        full_context_token_budget=0,
        cache_dir=tmp_path / "cache",
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    )
    settings.update(overrides)
    return select_passages(QUESTION, PASSAGES, **settings)


# --- lexical ---


def test_lexical_scores_rank_the_passage_sharing_the_question_words_first():
    scores = lexical_scores(QUESTION, PASSAGES)

    assert scores[1] > scores[0]
    assert scores[1] > scores[2]
    assert scores[0] == 0.0


def test_lexical_ranking_is_used_when_no_embedder_is_configured(tmp_path):
    result = select(tmp_path)

    assert result.method == "lexical"
    assert result.note == "no embedding tier configured"
    assert result.embed_call is None
    assert result.passages[0].passage_id == "S1-P2"
    assert len(result.passages) == 2


def test_lexical_ties_keep_document_order(tmp_path):
    result = select_passages(
        "zebra", PASSAGES, top_k=3, full_context_token_budget=0, cache_dir=tmp_path / "cache"
    )

    assert [passage.passage_id for passage in result.passages] == ["S1-P1", "S1-P2", "S1-P3"]


def test_lexical_ranking_configured_makes_no_embedding_request(tmp_path):
    embedder = embedder_preferring_telemetry()

    result = select(tmp_path, embedder=embedder, embedding_model="embed-model", ranking="lexical")

    assert embedder.batches == []
    assert result.method == "lexical"
    assert result.note == "lexical ranking configured"


# --- embeddings ---


def test_embedding_ranking_orders_by_cosine_similarity(tmp_path):
    embedder = embedder_preferring_telemetry()

    result = select(tmp_path, embedder=embedder, embedding_model="embed-model", ranking="embedding", top_k=3)

    assert result.method == "embedding"
    assert result.note is None
    assert [passage.passage_id for passage in result.passages] == ["S1-P3", "S1-P1", "S1-P2"]
    assert result.embed_call is not None and result.embed_call.ok
    assert result.scores["S1-P3"] > result.scores["S1-P1"] > result.scores["S1-P2"]


def test_question_and_passages_are_embedded_in_one_request_with_task_prefixes(tmp_path):
    embedder = embedder_preferring_telemetry()

    select(tmp_path, embedder=embedder, embedding_model="embed-model", ranking="embedding")

    assert len(embedder.batches) == 1
    batch = embedder.batches[0]
    assert batch[0] == "search_query: " + QUESTION
    assert batch[1:] == ["search_document: " + passage.text for passage in PASSAGES]


def test_embeddings_are_cached_on_disk_by_content(tmp_path):
    first_embedder = embedder_preferring_telemetry()
    select(tmp_path, embedder=first_embedder, embedding_model="embed-model", ranking="embedding")

    second_embedder = embedder_preferring_telemetry()
    result = select(tmp_path, embedder=second_embedder, embedding_model="embed-model", ranking="embedding")

    assert second_embedder.batches == []
    assert result.method == "embedding"
    assert result.embed_call is None
    assert result.passages[0].passage_id == "S1-P3"
    assert len(list((tmp_path / "cache" / "embeddings").glob("*.json"))) == 4


def test_only_uncached_texts_are_sent_and_another_model_does_not_share_the_cache(tmp_path):
    select(tmp_path, embedder=embedder_preferring_telemetry(), embedding_model="embed-model", ranking="embedding")

    new_question_embedder = FakeEmbedder({"different question": [1.0, 0.0]})
    select_passages(
        "A different question?",
        PASSAGES,
        top_k=2,
        full_context_token_budget=0,
        embedder=new_question_embedder,
        embedding_model="embed-model",
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        cache_dir=tmp_path / "cache",
    )
    assert new_question_embedder.batches == [["search_query: A different question?"]]

    other_model_embedder = embedder_preferring_telemetry()
    select(tmp_path, embedder=other_model_embedder, embedding_model="other-model", ranking="embedding")
    assert len(other_model_embedder.batches[0]) == 4


def test_failed_embedding_call_falls_back_to_lexical_and_says_why(tmp_path):
    def failing_embedder(texts: list[str]) -> EmbedResult:
        return EmbedResult(error_result("connection"))

    result = select(tmp_path, embedder=failing_embedder, embedding_model="embed-model")

    assert result.method == "lexical"
    assert result.note == "embedding call failed (connection)"
    assert result.embed_call is not None and not result.embed_call.ok
    assert result.passages[0].passage_id == "S1-P2"
    assert not (tmp_path / "cache" / "embeddings").exists()


def test_mismatched_vector_sizes_fall_back_to_lexical(tmp_path):
    embedder = FakeEmbedder(
        {
            "confidence threshold makes": [1.0, 0.0],
            "Telemetry rows": [0.9, 0.1, 0.3],
            "Sandboxes isolate": [0.5, 0.5],
            "router escalates": [0.0, 1.0],
        }
    )

    result = select(tmp_path, embedder=embedder, embedding_model="embed-model")

    assert result.method == "lexical"
    assert "unusable" in result.note


# --- hybrid ---


def test_hybrid_fuses_the_embedding_and_lexical_rankings(tmp_path):
    result = select(
        tmp_path, embedder=embedder_preferring_telemetry(), embedding_model="embed-model", top_k=3
    )

    # Embedding ranks 3, 1, 2. Lexical ranks 2 first and ties 1 and 3 at zero (1 before 3).
    # Passages 2 and 3 each hold a first and a last place and tie; passage 1 is second in
    # both, which scores just under them. Embeddings alone would have put passage 2 last.
    assert result.method == "hybrid"
    assert [passage.passage_id for passage in result.passages] == ["S1-P2", "S1-P3", "S1-P1"]
    assert result.scores["S1-P2"] == result.scores["S1-P3"] > result.scores["S1-P1"]


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([[0.9, 0.5, 0.1], [0.8, 0.1, 0.5]])

    assert fused[0] > fused[1]
    assert fused[0] > fused[2]
    assert fused[1] == pytest.approx(fused[2])


# --- skipping retrieval ---


def test_small_corpus_within_the_token_budget_uses_every_passage(tmp_path):
    embedder = embedder_preferring_telemetry()

    result = select(tmp_path, embedder=embedder, embedding_model="embed-model", full_context_token_budget=500)

    assert result.method == "all"
    assert [passage.passage_id for passage in result.passages] == ["S1-P1", "S1-P2", "S1-P3"]
    assert embedder.batches == []


def test_cosine_similarity_handles_zero_vectors():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


# --- the Ollama embed endpoint ---


def test_ollama_embed_posts_to_the_native_endpoint_and_returns_vectors():
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "embeddings": [[0.1, 0.2], [0.3, 0.4]],
                "prompt_eval_count": 12,
                "load_duration": 2_000_000,
                "total_duration": 5_000_000,
            },
        )

    client = OllamaNativeClient("http://ollama.invalid", transport=httpx.MockTransport(handler))
    result = client.embed(
        EmbedCall(model="embed-model", inputs=["a", "b"], timeout_s=5, keep_alive="10m", num_ctx=2048)
    )

    assert seen == [
        (
            "http://ollama.invalid/api/embed",
            {"model": "embed-model", "input": ["a", "b"], "keep_alive": "10m", "options": {"num_ctx": 2048}},
        )
    ]
    assert result.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert result.call.ok
    assert result.call.prompt_tokens == 12
    assert result.call.cost_source == "local_zero"
    assert result.call.ollama_load_duration_ms == 2.0


@pytest.mark.parametrize(
    "body",
    [
        {"embeddings": [[0.1, 0.2]]},
        {"embeddings": [[0.1], []]},
        {"embeddings": [[0.1], ["x"]]},
        {"vectors": []},
    ],
)
def test_ollama_embed_rejects_a_response_without_one_vector_per_input(body):
    client = OllamaNativeClient(
        "http://ollama.invalid", transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    )

    result = client.embed(EmbedCall(model="embed-model", inputs=["a", "b"], timeout_s=5))

    assert not result.call.ok
    assert result.call.error_type == "invalid_response"
    assert result.vectors == []


def test_ollama_embed_reports_http_errors_and_timeouts():
    failing = OllamaNativeClient(
        "http://ollama.invalid",
        transport=httpx.MockTransport(lambda request: httpx.Response(500, text="model not found")),
    )
    assert failing.embed(EmbedCall(model="m", inputs=["a"], timeout_s=5)).call.error_type == "http_5xx"

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    timing_out = OllamaNativeClient("http://ollama.invalid", transport=httpx.MockTransport(slow))
    assert timing_out.embed(EmbedCall(model="m", inputs=["a"], timeout_s=5)).call.error_type == "timeout"


def test_token_budget_takes_ranked_passages_until_the_budget_is_filled():
    # estimate_tokens is chars / 4: each 400-character passage is about 100 tokens.
    filler = " filler" * 55
    passages = [
        make_passage(1, "sandbox microvm isolation" + filler),
        make_passage(2, "confidence threshold escalate router" + filler),
        make_passage(3, "confidence threshold" + filler),
        make_passage(4, "telemetry sqlite rows" + filler),
    ]
    result = select_passages(
        "confidence threshold escalate router",
        passages,
        top_k=1,
        full_context_token_budget=0,
        ranking="lexical",
        retrieval_token_budget=250,
    )
    # top_k is ignored: two passages fit 250 tokens, the third would pass it.
    assert [passage.passage_id for passage in result.passages] == ["S1-P2", "S1-P3"]


def test_token_budget_smaller_than_one_passage_still_returns_the_best_passage():
    result = select_passages(
        QUESTION, PASSAGES, top_k=3, full_context_token_budget=0, ranking="lexical", retrieval_token_budget=1
    )
    assert [passage.passage_id for passage in result.passages] == ["S1-P2"]


def test_embedding_is_batched_and_reported_as_one_call(tmp_path, monkeypatch):
    import inference_arbiter.retrieval as retrieval_module

    monkeypatch.setattr(retrieval_module, "EMBED_BATCH_SIZE", 2)
    batch_sizes: list[int] = []

    def embedder(texts: list[str]) -> EmbedResult:
        batch_sizes.append(len(texts))
        call = CallResult(status="ok", latency_ms=10.0, prompt_tokens=len(texts))
        return EmbedResult(call=call, vectors=[[1.0, float(len(text))] for text in texts])

    vectors, embed_call = retrieval_module.embed_texts(
        ["a", "bb", "ccc", "dddd", "eeeee"], "embed-model", embedder, tmp_path
    )

    assert batch_sizes == [2, 2, 1]
    assert vectors == [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]
    assert embed_call.latency_ms == 30.0
    assert embed_call.prompt_tokens == 5


def test_a_failed_batch_keeps_the_vectors_of_the_batches_before_it(tmp_path, monkeypatch):
    import inference_arbiter.retrieval as retrieval_module

    monkeypatch.setattr(retrieval_module, "EMBED_BATCH_SIZE", 2)
    requests: list[list[str]] = []

    def embedder(texts: list[str]) -> EmbedResult:
        requests.append(texts)
        if len(requests) == 2:
            return EmbedResult(call=error_result("timeout"))
        call = CallResult(status="ok", latency_ms=5.0)
        return EmbedResult(call=call, vectors=[[1.0, 2.0] for _ in texts])

    vectors, embed_call = retrieval_module.embed_texts(["a", "b", "c"], "embed-model", embedder, tmp_path)
    assert vectors is None
    assert embed_call.error_type == "timeout"

    requests.clear()
    retrieval_module.embed_texts(["a", "b", "c"], "embed-model", embedder, tmp_path)
    # "a" and "b" were cached by the first batch; only "c" is requested again.
    assert requests == [["c"]]
