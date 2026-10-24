"""Semantic retrieval. Importing this module does not load or download a model."""
from functools import lru_cache
import numpy as np
from validation import validate_documents, validate_query

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def get_model(model_name=MODEL_NAME, local_files_only=False):
    """Load and cache a sentence-transformer model on first use."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, local_files_only=local_files_only)


def build_semantic_index(documents, model=None):
    """Encode document contents into normalized vectors in document order."""
    validate_documents(documents)
    if not documents:
        return np.empty((0, 0), dtype=float)
    model = model if model is not None else get_model()
    return np.asarray(model.encode(
        [document["content"] for document in documents],
        normalize_embeddings=True,
        show_progress_bar=False,
    ))


def search_semantic(documents, document_vectors, query, top_k=3, model=None):
    """Rank documents by cosine similarity between their vectors and the query."""
    validate_documents(documents)
    validate_query(query, top_k)
    if not documents or not query.strip():
        return []
    vectors = np.asarray(document_vectors)
    if vectors.ndim != 2 or vectors.shape[0] != len(documents):
        raise ValueError("Document records must match the indexed vectors")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings must contain finite values")
    model = model if model is not None else get_model()
    query_vector = np.asarray(model.encode(query, normalize_embeddings=True, show_progress_bar=False))
    if query_vector.ndim != 1 or vectors.shape[1] != query_vector.shape[0]:
        raise ValueError("Query and document embeddings must use the same model and dimensions")
    scores = vectors @ query_vector
    results = [{**document, "score": float(score)} for document, score in zip(documents, scores)]
    results.sort(key=lambda result: result["score"], reverse=True)
    return results if top_k is None else results[:top_k]
