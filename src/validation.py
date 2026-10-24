"""Small input checks shared by both retrieval methods."""


def validate_documents(documents):
    """Ensure every document record has a nonempty source and string content."""
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Each document must be a dictionary")
        if not isinstance(document.get("source"), str) or not document["source"].strip():
            raise ValueError("Each document needs a nonempty source string")
        if not isinstance(document.get("content"), str):
            raise ValueError("Each document needs string content")


def validate_query(query, top_k):
    """Check the query type and optional positive result limit."""
    if not isinstance(query, str):
        raise ValueError("Query must be a string")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k must be a positive integer or None")


def validate_chunk_settings(chunk_size, overlap):
    """Require a positive chunk size and overlap smaller than that size."""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (chunk_size, overlap)):
        raise ValueError("Chunk size and overlap must be integers")
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
