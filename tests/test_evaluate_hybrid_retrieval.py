import json
from types import SimpleNamespace

import evaluate_hybrid_retrieval as evaluator
import pytest


def result(source, chunk_id, content, score=0.8, **extra):
    return {
        "source": source,
        "page_number": None,
        "chunk_id": chunk_id,
        "content": content,
        "score": score,
        **extra,
    }


def test_evaluator_compares_semantic_and_hybrid_evidence_recall(
    tmp_path, monkeypatch
):
    questions = [
        {
            "id": "supported-01",
            "category": "supported",
            "question": "Question one?",
            "required_evidence": [{"source": "a.txt", "text": "Evidence A."}],
        },
        {
            "id": "supported-02",
            "category": "supported",
            "question": "Question two?",
            "required_evidence": [{"source": "b.txt", "text": "Evidence B."}],
        },
        {
            "id": "ambiguous-01",
            "category": "ambiguous",
            "question": "Question three?",
            "required_evidence": [
                {"source": "c.txt", "text": "Evidence C."},
                {"source": "d.txt", "text": "Evidence D."},
            ],
        },
        {
            "id": "unanswerable-01",
            "category": "unanswerable",
            "question": "Question four?",
            "required_evidence": [],
        },
    ]
    (tmp_path / "answer_questions.json").write_text(
        json.dumps(questions), encoding="utf-8"
    )

    semantic_results = {
        "Question one?": [result("a.txt", 1, "Evidence A.")],
        "Question two?": [],
        "Question three?": [result("c.txt", 1, "Evidence C.")],
        "Question four?": [],
    }
    hybrid_results = {
        "Question one?": [
            result("a.txt", 1, "Evidence A.", rrf_score=0.03)
        ],
        "Question two?": [
            result("b.txt", 1, "Evidence B.", rrf_score=0.03)
        ],
        "Question three?": [
            result("c.txt", 1, "Evidence C.", rrf_score=0.03),
            result("d.txt", 1, "Evidence D.", rrf_score=0.02),
        ],
        "Question four?": [],
    }

    monkeypatch.setattr(evaluator, "EVALUATION_DIR", tmp_path)
    monkeypatch.setattr(
        evaluator,
        "load_documents",
        lambda _path: [{"source": "fixture.txt", "content": "fixture"}],
    )
    monkeypatch.setattr(
        evaluator,
        "chunk_documents",
        lambda documents, chunk_size, overlap: documents,
    )
    monkeypatch.setattr(
        evaluator,
        "get_model",
        lambda model_name, local_files_only: object(),
    )
    monkeypatch.setattr(
        evaluator,
        "build_hybrid_index",
        lambda chunks, model: SimpleNamespace(semantic_vectors=object()),
    )
    monkeypatch.setattr(
        evaluator,
        "search_semantic",
        lambda chunks, vectors, query, top_k, model: semantic_results[query],
    )
    monkeypatch.setattr(
        evaluator,
        "search_hybrid",
        lambda chunks, index, query, top_k, candidate_k, rrf_k, model: hybrid_results[query],
    )

    report = evaluator.evaluate_hybrid_retrieval(
        model_name="fake-model",
        offline=True,
        top_k=3,
        candidate_k=5,
        rrf_k=60,
    )

    semantic = report["metrics"]["semantic"]
    hybrid = report["metrics"]["hybrid"]
    delta = report["metrics"]["delta_hybrid_minus_semantic"]

    assert semantic["all_required_evidence_recall_at_k"] == 1 / 3
    assert semantic["evidence_item_recall_at_k"] == 2 / 4
    assert hybrid["all_required_evidence_recall_at_k"] == 1.0
    assert hybrid["evidence_item_recall_at_k"] == 1.0
    assert delta["all_required_evidence_recall_at_k"] == pytest.approx(2 / 3)
    assert delta["evidence_item_recall_at_k"] == 0.5
    assert semantic["unanswerable_cases"] == 1
    assert (tmp_path / "hybrid_retrieval_results.json").is_file()


def test_evaluator_rejects_candidate_count_below_top_k():
    with pytest.raises(ValueError, match="candidate_k must be at least top_k"):
        evaluator.validate_settings(top_k=3, candidate_k=2, rrf_k=60)
