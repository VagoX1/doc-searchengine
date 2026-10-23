import pytest

from hybrid_search import reciprocal_rank_fusion


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
