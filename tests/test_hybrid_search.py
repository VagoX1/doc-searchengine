import pytest

import hybrid_search
from hybrid_search import HybridIndex, reciprocal_rank_fusion


def passage(source, chunk_id, content, score):
    return {
        "source": source,
        "page_number": None,
        "chunk_id": chunk_id,
        "content": content,
        "score": score,
    }


def test_rrf_combines_rankings_and_keeps_component_ranks():
    passage_a = passage("a.txt", 1, "First passage.", 0.91)
    passage_b_tfidf = passage("b.txt", 1, "Second passage.", 0.62)
    passage_b_semantic = passage("b.txt", 1, "Second passage.", 0.93)
    passage_c = passage("c.txt", 1, "Third passage.", 0.71)

    results = reciprocal_rank_fusion(
        {
            "tfidf": [passage_a, passage_b_tfidf],
            "semantic": [passage_b_semantic, passage_c],
        },
        top_k=3,
        rrf_k=60,
    )

    assert [result["source"] for result in results] == [
        "b.txt",
        "a.txt",
        "c.txt",
    ]
    assert results[0]["tfidf_rank"] == 2
    assert results[0]["semantic_rank"] == 1
    assert results[0]["tfidf_score"] == 0.62
    assert results[0]["semantic_score"] == 0.93
    assert results[0]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_returns_each_passage_once_and_applies_top_k():
    repeated_passage = passage("a.txt", 1, "Repeated passage.", 0.9)

    results = reciprocal_rank_fusion(
        {
            "tfidf": [repeated_passage, repeated_passage],
            "semantic": [repeated_passage],
        },
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["tfidf_rank"] == 1
    assert results[0]["semantic_rank"] == 1


@pytest.mark.parametrize(
    ("top_k", "rrf_k"),
    [
        (0, 60),
        (True, 60),
        (3, 0),
        (3, False),
    ],
)
def test_rrf_rejects_invalid_parameters(top_k, rrf_k):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({}, top_k=top_k, rrf_k=rrf_k)


def test_rrf_accepts_unlimited_results():
    results = reciprocal_rank_fusion(
        {"tfidf": [passage("a.txt", 1, "A.", 0.8)]},
        top_k=None,
    )

    assert len(results) == 1


def test_build_hybrid_index_builds_both_indexes_from_same_chunks(monkeypatch):
    chunks = [passage("a.txt", 1, "A passage.", 0.8)]
    vectorizer = object()
    tfidf_vectors = object()
    semantic_vectors = object()
    model = object()
    calls = []

    def fake_build_index(documents):
        calls.append(("tfidf", documents))
        return vectorizer, tfidf_vectors

    def fake_build_semantic_index(documents, model=None):
        calls.append(("semantic", documents, model))
        return semantic_vectors

    monkeypatch.setattr(hybrid_search, "build_index", fake_build_index)
    monkeypatch.setattr(
        hybrid_search,
        "build_semantic_index",
        fake_build_semantic_index,
    )

    index = hybrid_search.build_hybrid_index(chunks, model=model)

    assert calls == [
        ("tfidf", chunks),
        ("semantic", chunks, model),
    ]
    assert index.tfidf_vectorizer is vectorizer
    assert index.tfidf_vectors is tfidf_vectors
    assert index.semantic_vectors is semantic_vectors
    assert index.model is model


def test_build_hybrid_index_reuses_cached_semantic_vectors(monkeypatch):
    import numpy as np

    chunks = [passage("a.txt", 1, "A passage.", 0.8)]
    vectorizer = object()
    tfidf_vectors = object()
    cached_semantic_vectors = np.asarray([[1.0, 0.0]])

    monkeypatch.setattr(
        hybrid_search,
        "build_index",
        lambda documents: (vectorizer, tfidf_vectors),
    )
    monkeypatch.setattr(
        hybrid_search,
        "build_semantic_index",
        lambda *args, **kwargs: pytest.fail(
            "Cached semantic vectors should avoid re-embedding documents"
        ),
    )

    index = hybrid_search.build_hybrid_index(
        chunks,
        semantic_vectors=cached_semantic_vectors,
    )

    assert index.tfidf_vectorizer is vectorizer
    assert index.tfidf_vectors is tfidf_vectors
    assert np.array_equal(index.semantic_vectors, cached_semantic_vectors)


def test_search_hybrid_queries_both_retrievers_then_fuses(monkeypatch):
    chunks = [
        passage("a.txt", 1, "First passage.", 0.0),
        passage("b.txt", 1, "Second passage.", 0.0),
        passage("c.txt", 1, "Third passage.", 0.0),
    ]
    index = HybridIndex(
        tfidf_vectorizer=object(),
        tfidf_vectors=object(),
        semantic_vectors=object(),
        model=object(),
    )
    tfidf_candidates = [
        {**chunks[0], "score": 0.9},
        {**chunks[1], "score": 0.7},
    ]
    semantic_candidates = [
        {**chunks[1], "score": 0.95},
        {**chunks[2], "score": 0.6},
    ]
    requested_model = object()
    calls = {}

    def fake_search_index(documents, vectorizer, vectors, query, top_k):
        calls["tfidf"] = (documents, query, top_k)
        return tfidf_candidates

    def fake_search_semantic(documents, vectors, query, top_k, model):
        calls["semantic"] = (documents, query, top_k, model)
        return semantic_candidates

    monkeypatch.setattr(hybrid_search, "search_index", fake_search_index)
    monkeypatch.setattr(hybrid_search, "search_semantic", fake_search_semantic)

    results = hybrid_search.search_hybrid(
        chunks,
        index,
        "shared query",
        top_k=2,
        candidate_k=5,
        rrf_k=60,
        model=requested_model,
    )

    assert calls["tfidf"] == (chunks, "shared query", 5)
    assert calls["semantic"] == (
        chunks,
        "shared query",
        5,
        requested_model,
    )
    assert [result["source"] for result in results] == ["b.txt", "a.txt"]
    assert results[0]["tfidf_rank"] == 2
    assert results[0]["semantic_rank"] == 1


def test_search_hybrid_rejects_candidate_pool_smaller_than_output():
    chunks = [passage("a.txt", 1, "A passage.", 0.8)]
    index = HybridIndex(None, None, None)

    with pytest.raises(ValueError, match="candidate_k must be at least top_k"):
        hybrid_search.search_hybrid(
            chunks,
            index,
            "query",
            top_k=3,
            candidate_k=2,
        )
