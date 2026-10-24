"""Command-line entry point for local document search."""
import argparse
import logging

from chunk_text import chunk_documents
from file_loader import load_documents
from semantic_search import build_semantic_index, search_semantic
from tf_idf import build_index, search_index
from validation import validate_query, validate_chunk_settings


def run_search(directory="documents", query="borrow books", chunk_size=100, overlap=20, top_k=3, method="semantic"):
    """Load, chunk, index, and search a document folder.

    Args:
        directory: Folder containing supported document files.
        query: User's search text.
        chunk_size: Maximum words per chunk.
        overlap: Words shared between adjacent chunks.
        top_k: Maximum number of passages to display.
        method: Either ``"semantic"`` or ``"tfidf"``.

    Returns:
        The ranked passage records, or an empty list when there are no results.
    """
    validate_query(query, top_k)
    validate_chunk_settings(chunk_size, overlap)
    if method not in {"semantic", "tfidf"}:
        raise ValueError("Method must be semantic or tfidf")
    if not query.strip():
        print("Enter a nonempty query.")
        return []
    documents = load_documents(directory)
    chunks = chunk_documents(documents, chunk_size, overlap)

    if not chunks:
        print("No documents to search.")
        return []

    if method == "semantic":
        # Semantic search embeds the query and compares it with document vectors.
        document_vectors = build_semantic_index(chunks)
        results = search_semantic(chunks, document_vectors, query, top_k=top_k)
    else:
        # TF-IDF search uses a fitted vectorizer and sparse document matrix.
        vectorizer, vectors = build_index(chunks)
        results = search_index(chunks, vectorizer, vectors, query, top_k=top_k)

    if not results:
        print("No matching passages found.")

    for result in results:
        content = result["content"]
        preview = content[:200] + "..." if len(content) > 200 else content

        location = result["source"]

        if result.get("page_number") is not None:
            location += f" - PDF page {result['page_number']}"

        print(
            f"{location} - chunk {result['chunk_id']} "
            f"(score: {result['score']:.3f})"
        )
        print(preview)
        print()

    return results


if __name__ == "__main__":
    # Configure logging and expose the search options as command-line arguments.
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Search local TXT, PDF, and DOCX documents")
    parser.add_argument("query", nargs="?", default="borrow books")
    parser.add_argument("--directory", default="documents")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--overlap", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--method", choices=["semantic", "tfidf"], default="semantic")
    args = parser.parse_args()
    try:
        run_search(args.directory, args.query, args.chunk_size, args.overlap, args.top_k, args.method)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Search failed: {exc}\n")
