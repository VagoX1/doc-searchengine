"""Build and query reusable TF-IDF indexes for document chunks."""

from validation import validate_documents, validate_query

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_index(documents):
    """Fit a TF-IDF vectorizer and transform documents in their existing order."""
    validate_documents(documents)
    if not documents:
        return None, None
    texts = [document["content"] for document in documents]
    vectorizer = TfidfVectorizer()
    analyze = vectorizer.build_analyzer()
    if not any(analyze(text) for text in texts):
        return None, None
    return vectorizer, vectorizer.fit_transform(texts)


def search_index(documents, vectorizer, document_vectors, query, top_k=3):
    """Rank indexed documents by TF-IDF cosine similarity to a query."""
    validate_documents(documents)
    validate_query(query, top_k)
    if not documents or not query.strip() or vectorizer is None:
        return []
    if document_vectors is None or document_vectors.shape[0] != len(documents):
        raise ValueError("Document records must match the indexed vectors")
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, document_vectors)[0]
    results = [
        {**document, "score": float(score)}
        for document, score in zip(documents, scores)
        if score > 0
    ]
    results.sort(key=lambda result: result["score"], reverse=True)
    return results if top_k is None else results[:top_k]


def search_tfidf(documents, query):
    """Build a temporary index and return every positive-scoring TF-IDF match.

    For repeated searches over the same documents, call :func:`build_index` once
    and reuse :func:`search_index` instead.
    """
    validate_documents(documents)
    validate_query(query, None)
    if not documents or not query.strip():
        return []
    vectorizer, vectors = build_index(documents)
    return search_index(documents, vectorizer, vectors, query, top_k=None)
