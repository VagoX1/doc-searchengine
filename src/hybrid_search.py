"""Build hybrid indexes and fuse TF-IDF and semantic rankings with RRF."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from semantic_search import build_semantic_index, search_semantic
from tf_idf import build_index, search_index
from validation import validate_documents, validate_query


def _validate_positive_int(value, name, allow_none=False):
    """Validate a positive integer parameter, optionally accepting None."""
    if allow_none and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _passage_key(result):
    """Identify a chunk consistently across retriever result dictionaries."""
    return (
        result["source"],
        result.get("page_number"),
        result.get("chunk_id"),
        result["content"],
    )


def reciprocal_rank_fusion(rankings, top_k=3, rrf_k=60):
    """Fuse named ranked lists using reciprocal rank fusion.

    Args:
        rankings: Mapping from retriever names to ranked result lists.
        top_k: Maximum number of fused results, or None for all candidates.
        rrf_k: Rank constant in ``1 / (rrf_k + rank)``.

    Returns:
        Deduplicated passages sorted by their summed RRF scores. Each result has
        ``score``/``rrf_score`` plus a rank and component score per method. The
        returned score is an RRF score, not a cosine similarity.
    """
    if not isinstance(rankings, dict):
        raise ValueError(
            "rankings must be a dictionary mapping method names to result lists"
        )

    _validate_positive_int(top_k, "top_k", allow_none=True)
    _validate_positive_int(rrf_k, "rrf_k")

    fused = {}

    # Accumulate one reciprocal-rank contribution per method and passage.
    for method, results in rankings.items():
        if not isinstance(method, str) or not method.strip():
            raise ValueError("Each ranking method needs a nonempty name")
        if not isinstance(results, list):
            raise ValueError(f"Results for {method} must be a list")

        for rank, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                raise ValueError("Each ranked result must be a dictionary")
            if not isinstance(result.get("source"), str) or not result["source"].strip():
                raise ValueError("Each result needs a nonempty source")
            if not isinstance(result.get("content"), str):
                raise ValueError("Each result needs content")

            key = _passage_key(result)
            if key not in fused:
                # Preserve passage metadata separately from method-specific scores.
                document = {
                    field: value
                    for field, value in result.items()
                    if field not in {"score", "rrf_score"}
                }
                fused[key] = {
                    "document": document,
                    "rrf_score": 0.0,
                    "ranks": {},
                    "component_scores": {},
                }

            entry = fused[key]

            # A duplicate within one method counts only at its first rank.
            if method in entry["ranks"]:
                continue

            entry["rrf_score"] += 1.0 / (rrf_k + rank)
            entry["ranks"][method] = rank
            entry["component_scores"][method] = result.get("score")

    ranked = []
    for entry in fused.values():
        score = entry["rrf_score"]
        result = {
            **entry["document"],
            "score": score,
            "rrf_score": score,
        }

        # Retain component rank/score details so evaluation reports are inspectable.
        for method in rankings:
            result[f"{method}_rank"] = entry["ranks"].get(method)
            result[f"{method}_score"] = entry["component_scores"].get(method)

        ranked.append(result)

    # Sorting is stable, so equal scores retain their first-seen order.
    ranked.sort(key=lambda result: result["rrf_score"], reverse=True)
    return ranked if top_k is None else ranked[:top_k]


@dataclass
class HybridIndex:
    """Hold TF-IDF and semantic indexes built over the same ordered chunks."""

    tfidf_vectorizer: Any
    tfidf_vectors: Any
    semantic_vectors: Any
    model: Any = None


def build_hybrid_index(chunks, model=None, semantic_vectors=None):
    """Build both retriever indexes once for reuse across queries."""
    validate_documents(chunks)

    # Fit lexical search over these chunks.
    tfidf_vectorizer, tfidf_vectors = build_index(chunks)

    # Reuse cached semantic vectors, or compute them for a new document.
    if semantic_vectors is None:
        semantic_vectors = build_semantic_index(chunks, model=model)
    else:
        semantic_vectors = np.asarray(semantic_vectors)
        if semantic_vectors.ndim != 2 or semantic_vectors.shape[0] != len(chunks):
            raise ValueError("Document records must match the semantic vectors")
        if not np.isfinite(semantic_vectors).all():
            raise ValueError("Embeddings must contain finite values")

    return HybridIndex(
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_vectors=tfidf_vectors,
        semantic_vectors=semantic_vectors,
        model=model,
    )


def search_hybrid(
    chunks,
    hybrid_index,
    query,
    top_k=3,
    candidate_k=20,
    rrf_k=60,
    model=None,
):
    """Retrieve candidates from both methods and return their fused ranking.

    Args:
        chunks: Ordered document chunks used to build ``hybrid_index``.
        hybrid_index: Indexes returned by :func:`build_hybrid_index`.
        query: User query shared by the TF-IDF and semantic retrievers.
        top_k: Number of fused results to return.
        candidate_k: Candidate limit requested from each retriever.
        rrf_k: Reciprocal-rank fusion constant.
        model: Optional semantic model override.
    """
    validate_documents(chunks)
    validate_query(query, top_k)
    _validate_positive_int(candidate_k, "candidate_k")
    _validate_positive_int(rrf_k, "rrf_k")

    if top_k is not None and candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k")
    if not isinstance(hybrid_index, HybridIndex):
        raise ValueError("hybrid_index must be built with build_hybrid_index")

    if not chunks or not query.strip():
        return []

    # Reuse the index's model unless the caller explicitly supplies another one.
    query_model = model if model is not None else hybrid_index.model

    # Each retriever produces an independent candidate ranking for the same query.
    tfidf_candidates = search_index(
        chunks,
        hybrid_index.tfidf_vectorizer,
        hybrid_index.tfidf_vectors,
        query,
        top_k=candidate_k,
    )
    semantic_candidates = search_semantic(
        chunks,
        hybrid_index.semantic_vectors,
        query,
        top_k=candidate_k,
        model=query_model,
    )

    # Fuse by rank rather than combining incomparable raw score scales.
    return reciprocal_rank_fusion(
        {
            "tfidf": tfidf_candidates,
            "semantic": semantic_candidates,
        },
        top_k=top_k,
        rrf_k=rrf_k,
    )
