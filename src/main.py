import argparse
import logging

from chunk_text import chunk_documents
from file_loader import load_documents
from semantic_search import build_semantic_index, search_semantic
from tf_idf import build_index, search_index
from validation import validate_query, validate_chunk_settings


def run_search(directory="documents", query="borrow books", chunk_size=100, overlap=20, top_k=3, method="semantic"):
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
        document_vectors = build_semantic_index(chunks)
        results = search_semantic(chunks, document_vectors, query, top_k=top_k)
    else:
        vectorizer, vectors = build_index(chunks)
        results = search_index(chunks, vectorizer, vectors, query, top_k=top_k)

    if not results:
        print("No matching passages found.")

    for result in results:
        content = result["content"]
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"{result['source']} - chunk {result['chunk_id']} (score: {result['score']:.3f})")
        print(preview)
        print()

    return results


if __name__ == "__main__":
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
