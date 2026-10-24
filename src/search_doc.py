"""Simple keyword-overlap baseline retained for comparison and learning."""
import re

def search_documents(documents, query):
    """Return documents ranked by the number of unique query words they contain."""
    if not documents or not query.strip():
        return []

    query_words = set(re.findall(r"\b\w+\b", query.lower()))
    results = []
    for document in documents:
        document_words = set(re.findall(r"\b\w+\b", document["content"].lower()))

        matching_words = query_words.intersection(document_words)
        score = len(matching_words)

        if score > 0:
            results.append({
                "source":document["source"],
                "score": score
            })
    results = sorted(
        results,
        key=lambda result:result["score"],
        reverse=True
    )

    return results
