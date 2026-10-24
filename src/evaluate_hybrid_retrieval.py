"""Compare semantic-only and hybrid evidence retrieval on fixed fixtures."""

import argparse
import json
from pathlib import Path

from chunk_text import chunk_documents
from evaluate_answer_retrieval import assess_retrieval
from file_loader import load_documents
from hybrid_search import build_hybrid_index, search_hybrid
from semantic_search import MODEL_NAME, get_model, search_semantic


# Resolve the evaluation data relative to the project, not the launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

# Use the same chunk settings and final result count for both methods.
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
TOP_K = 3
CANDIDATE_K = 20
RRF_K = 60


def validate_settings(top_k, candidate_k, rrf_k):
    """Validate evaluation settings before loading documents or a model."""
    for value, name in (
        (top_k, "top_k"),
        (candidate_k, "candidate_k"),
        (rrf_k, "rrf_k"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    if candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k")


def serialize_results(results):
    """Select JSON-friendly passage fields and hybrid rank diagnostics."""
    serialized = []

    for result in results:
        item = {
            "source": result["source"],
            "page_number": result.get("page_number"),
            "chunk_id": result.get("chunk_id"),
            "content": result["content"],
            "score": float(result["score"]),
        }

        # Hybrid results carry RRF plus each retriever's rank and original score.
        for field in (
            "rrf_score",
            "tfidf_rank",
            "tfidf_score",
            "semantic_rank",
            "semantic_score",
        ):
            if field in result:
                value = result[field]
                if value is not None and field.endswith("_score"):
                    value = float(value)
                item[field] = value

        serialized.append(item)

    return serialized


def summarize_method(details, method):
    """Compute evidence recall for one method, excluding unanswerable questions."""
    applicable = [
        row
        for row in details
        if row[method]["retrieval"]["retrieval_applicable"]
    ]

    required_count = sum(len(row["required_evidence"]) for row in applicable)
    missing_count = sum(
        len(row[method]["retrieval"]["missing_evidence"])
        for row in applicable
    )

    return {
        "answerable_and_ambiguous_cases": len(applicable),
        "all_required_evidence_recall_at_k": (
            sum(row[method]["retrieval"]["retrieval_ok"] for row in applicable)
            / len(applicable)
            if applicable else None
        ),
        "evidence_item_recall_at_k": (
            (required_count - missing_count) / required_count
            if required_count else None
        ),
        "unanswerable_cases": sum(
            row["category"] == "unanswerable"
            for row in details
        ),
    }


def calculate_delta(hybrid_metrics, semantic_metrics, metric):
    """Return hybrid minus semantic for a metric, or None if unavailable."""
    hybrid_value = hybrid_metrics[metric]
    semantic_value = semantic_metrics[metric]

    if hybrid_value is None or semantic_value is None:
        return None

    return hybrid_value - semantic_value


def format_percentage(value):
    """Format a metric for console output without failing on unavailable values."""
    return f"{value:.1%}" if value is not None else "n/a"


def evaluate_hybrid_retrieval(
    model_name=MODEL_NAME,
    offline=False,
    top_k=TOP_K,
    candidate_k=CANDIDATE_K,
    rrf_k=RRF_K,
):
    """Run both retrievers and save aggregate and per-question comparisons."""
    validate_settings(top_k, candidate_k, rrf_k)

    questions = json.loads(
        (EVALUATION_DIR / "answer_questions.json").read_text(encoding="utf-8")
    )
    if not questions:
        raise ValueError("Hybrid evaluation needs questions")

    documents = load_documents(EVALUATION_DIR / "answer_documents")
    chunks = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )
    if not chunks:
        raise ValueError("Answer evaluation corpus contains no searchable text")

    # Build one shared semantic model and both indexes before processing questions.
    model = get_model(model_name, local_files_only=offline)
    hybrid_index = build_hybrid_index(chunks, model=model)

    details = []

    for question in questions:
        query = question["question"]

        # Semantic-only is the baseline, using the same chunks and embedding model.
        semantic_results = search_semantic(
            chunks,
            hybrid_index.semantic_vectors,
            query,
            top_k=top_k,
            model=model,
        )

        # Hybrid retrieval uses TF-IDF and semantic candidate lists, then fuses ranks.
        hybrid_results = search_hybrid(
            chunks,
            hybrid_index,
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            model=model,
        )

        # Score both outputs against the same expected evidence for this question.
        details.append({
            **question,
            "semantic": {
                "retrieval": assess_retrieval(question, semantic_results),
                "results": serialize_results(semantic_results),
            },
            "hybrid": {
                "retrieval": assess_retrieval(question, hybrid_results),
                "results": serialize_results(hybrid_results),
            },
        })

    semantic_metrics = summarize_method(details, "semantic")
    hybrid_metrics = summarize_method(details, "hybrid")

    # A positive delta means hybrid found more required evidence on this fixture set.
    report = {
        "model": model_name,
        "chunk_size": CHUNK_SIZE,
        "overlap": CHUNK_OVERLAP,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "rrf_k": rrf_k,
        "metrics": {
            "semantic": semantic_metrics,
            "hybrid": hybrid_metrics,
            "delta_hybrid_minus_semantic": {
                "all_required_evidence_recall_at_k": calculate_delta(
                    hybrid_metrics,
                    semantic_metrics,
                    "all_required_evidence_recall_at_k",
                ),
                "evidence_item_recall_at_k": calculate_delta(
                    hybrid_metrics,
                    semantic_metrics,
                    "evidence_item_recall_at_k",
                ),
            },
        },
        "details": details,
    }

    output_path = EVALUATION_DIR / "hybrid_retrieval_results.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Print a compact summary while retaining detailed results in the JSON report.
    for method in ("semantic", "hybrid"):
        metrics = report["metrics"][method]
        print(
            f"{method}: all-required@{top_k}="
            f"{format_percentage(metrics['all_required_evidence_recall_at_k'])}, "
            f"evidence-item recall@{top_k}="
            f"{format_percentage(metrics['evidence_item_recall_at_k'])}"
        )

    delta = report["metrics"]["delta_hybrid_minus_semantic"]
    delta_value = delta["all_required_evidence_recall_at_k"]
    delta_text = f"{delta_value:+.1%}" if delta_value is not None else "n/a"
    print(f"Hybrid minus semantic all-required recall: {delta_text}")
    print(f"Per-question results: {output_path}")
    return report


if __name__ == "__main__":
    # Expose model and retrieval settings for repeatable local/CI evaluations.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    args = parser.parse_args()

    evaluate_hybrid_retrieval(
        model_name=args.model,
        offline=args.offline,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
    )
