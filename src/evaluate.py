"""Compare retrieval using fixed fictional documents and expected evidence."""
import argparse
import json
from pathlib import Path

from file_loader import load_documents
from chunk_text import chunk_documents
from tf_idf import build_index, search_index
from semantic_search import build_semantic_index, search_semantic, get_model, MODEL_NAME

BASE = Path(__file__).resolve().parent


def passage_matches(result, question):
    def normalize(text):
        return " ".join(text.split()).lower()
    return (result["source"] == question["source"]
            and normalize(question["expected_text"]) in normalize(result["content"]))


def measure(questions, search):
    if not questions:
        raise ValueError("Evaluation needs questions")
    rows = []
    for question in questions:
        results = search(question["query"])
        rank = next((rank for rank, result in enumerate(results, 1)
                     if passage_matches(result, question)), None)
        rows.append({**question, "rank": rank, "retrieved": [
            {"source": r["source"], "chunk_id": r["chunk_id"], "score": r["score"]}
            for r in results]})
    return {"questions": len(rows),
            "hit_at_1": sum(r["rank"] == 1 for r in rows) / len(rows),
            "hit_at_3": sum(r["rank"] is not None for r in rows) / len(rows),
            "mrr_at_3": sum(1 / r["rank"] if r["rank"] else 0 for r in rows) / len(rows),
            "details": rows}


def evaluate(method="both", model_name=MODEL_NAME, offline=False):
    chunks = chunk_documents(load_documents(BASE / "evaluation" / "documents"), 40, 10)
    questions = json.loads((BASE / "evaluation/questions.json").read_text(encoding="utf-8"))
    for question in questions:
        if not any(passage_matches(chunk, question) for chunk in chunks):
            raise ValueError(f"Expected evidence missing for question {question['id']}")
    report = {"corpus": "fictional six-document practice corpus", "chunk_size": 40,
              "overlap": 10, "top_k": 3, "model": model_name if method != "tfidf" else None,
              "results": {}}
    if method in {"both", "tfidf"}:
        vectorizer, vectors = build_index(chunks)
        report["results"]["tfidf"] = measure(questions, lambda q: search_index(chunks, vectorizer, vectors, q))
    if method in {"both", "semantic"}:
        model = get_model(model_name, local_files_only=offline)
        vectors = build_semantic_index(chunks, model=model)
        report["results"]["semantic"] = measure(questions, lambda q: search_semantic(chunks, vectors, q, model=model))
    name = "results.json" if method == "both" else f"results_{method}.json"
    output = BASE / "evaluation" / name
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, result in report["results"].items():
        print(f"{name}: hit@1={result['hit_at_1']:.1%}, hit@3={result['hit_at_3']:.1%}, MRR@3={result['mrr_at_3']:.3f}")
    print(f"Per-question results: {output}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["both", "tfidf", "semantic"], default="both")
    parser.add_argument("--model", default=MODEL_NAME, help="Model identifier or local folder")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    evaluate(args.method, args.model, args.offline)
