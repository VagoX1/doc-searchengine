from dataclasses import dataclass

from semantic_search import build_semantic_index, search_semantic
from tf_idf import build_index, search_index
from validation import validate_documents, validate_query

"""Rank fusion for hybrid retrieval"""

def _validate_positive_int(value, name, allow_none=False):
    if allow_none and value is None:
        return
    if isinstance(value, bool) or not isinstance(value,int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")

def _passage_key(result):
    """Identify the same chunk across different ranked result lists."""
    return (
        result["source"],
        result.get("page_number"),
        result.get("chunk_id"),
        result["content"],
    )

def reciprocal_rank_fusion(rankings, top_k=3, rrf_k=60):
    """Fuse named ranked lists using reciprocal rank fusion.

    `rankings` maps method names such as "tfidf" and "semantic" to result lists.
    Ranks are one-based. Returned `score` is the fused RRF score, not a similarity.
    """
    if not isinstance(rankings, dict):
        raise ValueError("rankings must be a dictionary mapping method names to result lists")

    _validate_positive_int(top_k, "top_k", allow_none=True)
    _validate_positive_int(rrf_k, "rrf_k")

    fused = {}

    for method, results in rankings.items():
        if not isinstance(method, str) or not method.strip():
            raise ValueError("Each ranking method needs a nonempty name")
        if not isinstance(results, list):
            raise ValueError(f"Results for {method} must be a list")
        for rank, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                raise ValueError("Each ranked result must be a dictionary")
            if not isinstance(result.get("source"), str):
                raise ValueError("Each result needs a source")
            if not isinstance(result.get("content"), str):
                raise ValueError("Each result needs content")
            
            key = _passage_key(result)
            if key not in fused:
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

             # Count a passage at most once per ranking, using its first rank.

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

        for method in rankings:
            result[f"{method}_rank"] = entry["ranks"].get(method)
            result[f"{method}_score"] = entry["component_scores"].get(method)

        ranked.append(result)

    # Python's stable sort preserves first-seen order for equal RRF scores.
    ranked.sort(key=lambda result: result["rrf_score"], reverse=True)
    return ranked if top_k is None else ranked[:top_k]

@dataclass
class HybridIndex:
    """Both retrieval indexes built over the same ordered chunks."""

    tfidf_vectorizer: object
    tfidf_vectors: object
    semantic_vectors: object
    model: object = None

def build_hybrid_index(chunks, model=None):
    """Build TF-IDF and semantic indexes once for reuse across queries."""
    validate_documents(chunks)

    tfidf_vectorizer, tfidf_vectors = build_index(chunks)
    semantic_vectors = build_semantic_index(chunks, model=model)

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
    """Retrieve candidates with both methods, then fuse their rankings."""
    validate_documents(chunks)
    validate_query(query, top_k)
    _validate_positive_int(candidate_k, "candidate_k")
    _validate_positive_int(rrf_k, "rrf_k")

    if top_k is not NOne and candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k")
    if not isinstance(hybrid_index, HybridIndex):
        raise ValueError("hybrid_index must be built with build_hybrid_index")
    
    if not chunks or not query.strip():
        return []

    query_model = model if model is not None else hybrid_index.model

    semantic_candidates = search_semantic(
        chunks,
        hybrid_index.semantic_vectors,
        query,
        top_k=candidate_k,
        model=query_model,
    )

    return reciprocal_rank_fusion(
        {
            "tfidf": tfidf_candidates,
            "semantic": semantic_candidates,
        },
        top_k=top_k,
        rrf_k=rrf_k,
    )