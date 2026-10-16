"""Evaluate whether top-k semantic retrieval finds answer-required evidence."""
import argparse
import json
from pathlib import Path

from chunk_text import chunk_documents
from file_loader import load_documents
from semantic_search import (
    MODEL_NAME,
    build_semantic_index,
    get_model,
    search_semantic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
TOP_K = 3


def normalize(text):
    return " ".join(text.split()).lower()


def evidence_was_retrieved(expected, retrieved):
    expected_text = normalize(expected["text"])

    return any(
        result.get("source") == expected["source"]
        and expected_text in normalize(result.get("content", ""))
        for result in retrieved
    )


def assess_retrieval(question, retrieved):
    category = question["category"]
    if category == "unanswerable":
        return {
            "retrieval_applicable": False,
            "retrieval_ok": None,
            "missing_evidence": [],
            "failure_layer": "not_applicable",
        }
    if category not in {"supported", "ambiguous"}:
        raise ValueError(f"Unknown question category: {category}")

    missing = [
        evidence
        for evidence in question["required_evidence"]
        if not evidence_was_retrieved(evidence, retrieved)
    ]

    return {
        "retrieval_applicable": True,
        "retrieval_ok": not missing,
        "missing_evidence": missing,
        "failure_layer": "retrieval" if missing else None,
    }


def evaluate_answer_retrieval(
    model_name=MODEL_NAME,
    offline=False,
    top_k=TOP_K,
):
    """Run retrieval-only scoring and save a per-question JSON report."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    questions = json.loads(
        (EVALUATION_DIR / "answer_questions.json").read_text(encoding="utf-8")
    )
    if not questions:
        raise ValueError("Answer evaluation needs questions")

    documents = load_documents(EVALUATION_DIR / "answer_documents")
    chunks = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )
    if not chunks:
        raise ValueError("Answer evaluation corpus contains no searchable text")

    model = get_model(model_name, local_files_only=offline)
    vectors = build_semantic_index(chunks, model=model)

    rows = []
    for question in questions:
        retrieved = search_semantic(
            chunks,
            vectors,
            question["question"],
            top_k=top_k,
            model=model,
        )
        retrieval_check = assess_retrieval(question, retrieved)

        rows.append({
            **question,
            "retrieval": retrieval_check,
            "retrieved": [
                {
                    "source": result["source"],
                    "page_number": result.get("page_number"),
                    "chunk_id": result["chunk_id"],
                    "score": float(result["score"]),
                    "content": result["content"],
                }
                for result in retrieved
            ],
        })

    applicable = [
        row for row in rows
        if row["retrieval"]["retrieval_applicable"]
    ]
    required_count = sum(len(row["required_evidence"]) for row in applicable)
    missing_count = sum(
        len(row["retrieval"]["missing_evidence"])
        for row in applicable
    )

    metrics = {
        "answerable_and_ambiguous_cases": len(applicable),
        "all_required_evidence_recall_at_k": (
            sum(row["retrieval"]["retrieval_ok"] for row in applicable)
            / len(applicable)
            if applicable else None
        ),
        "evidence_item_recall_at_k": (
            (required_count - missing_count) / required_count
            if required_count else None
        ),
        "unanswerable_cases": sum(
            row["category"] == "unanswerable"
            for row in rows
        ),
    }

    report = {
        "model": model_name,
        "chunk_size": CHUNK_SIZE,
        "overlap": CHUNK_OVERLAP,
        "top_k": top_k,
        "metrics": metrics,
        "details": rows,
    }

    output_path = EVALUATION_DIR / "answer_retrieval_results.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    all_required_recall = metrics["all_required_evidence_recall_at_k"]
    evidence_item_recall = metrics["evidence_item_recall_at_k"]
    print(
        f"All-required-evidence@{top_k}: "
        f"{all_required_recall:.1%}" if all_required_recall is not None
        else f"All-required-evidence@{top_k}: n/a"
    )
    print(
        f"Evidence-item recall@{top_k}: "
        f"{evidence_item_recall:.1%}" if evidence_item_recall is not None
        else f"Evidence-item recall@{top_k}: n/a"
    )
    print(f"Per-question results: {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    evaluate_answer_retrieval(args.model, args.offline, args.top_k)
