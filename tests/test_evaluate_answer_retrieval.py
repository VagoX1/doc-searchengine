import json

import evaluate_answer_retrieval as evaluator
import pytest


def test_evaluate_answer_retrieval_writes_metrics_without_loading_a_real_model(
    tmp_path, monkeypatch
):
    questions = [
        {
            "id": "supported-01",
            "category": "supported",
            "question": "When are books due?",
            "required_evidence": [
                {"source": "library.txt", "text": "Return books in fourteen days."}
            ],
        },
        {
            "id": "ambiguous-01",
            "category": "ambiguous",
            "question": "How long does shipping take?",
            "required_evidence": [
                {"source": "shipping.txt", "text": "Standard takes three days."},
                {"source": "shipping.txt", "text": "Express takes one day."},
            ],
        },
        {
            "id": "unanswerable-01",
            "category": "unanswerable",
            "question": "What is the phone number?",
            "required_evidence": [],
        },
    ]
    (tmp_path / "answer_questions.json").write_text(
        json.dumps(questions), encoding="utf-8"
    )

    documents = [{"source": "fixture.txt", "content": "fixture text"}]
    retrieved_by_question = {
        "When are books due?": [
            {
                "source": "library.txt",
                "page_number": None,
                "chunk_id": 1,
                "score": 0.9,
                "content": "Please return books in fourteen days.",
            }
        ],
        "How long does shipping take?": [
            {
                "source": "shipping.txt",
                "page_number": None,
                "chunk_id": 1,
                "score": 0.8,
                "content": "Standard takes three days.",
            }
        ],
        "What is the phone number?": [],
    }

    monkeypatch.setattr(evaluator, "EVALUATION_DIR", tmp_path)
    monkeypatch.setattr(evaluator, "load_documents", lambda _path: documents)
    monkeypatch.setattr(
        evaluator,
        "chunk_documents",
        lambda docs, chunk_size, overlap: docs,
    )
    monkeypatch.setattr(evaluator, "get_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        evaluator,
        "build_semantic_index",
        lambda docs, model: object(),
    )
    monkeypatch.setattr(
        evaluator,
        "search_semantic",
        lambda chunks, vectors, query, top_k, model: retrieved_by_question[query],
    )

    report = evaluator.evaluate_answer_retrieval(top_k=3)

    assert report["metrics"] == {
        "answerable_and_ambiguous_cases": 2,
        "all_required_evidence_recall_at_k": 0.5,
        "evidence_item_recall_at_k": 2 / 3,
        "unanswerable_cases": 1,
    }
    assert report["details"][1]["retrieval"]["missing_evidence"] == [
        {"source": "shipping.txt", "text": "Express takes one day."}
    ]
    assert report["details"][2]["retrieval"]["retrieval_applicable"] is False
    assert (tmp_path / "answer_retrieval_results.json").is_file()


def test_assess_retrieval_rejects_unknown_categories():
    question = {"category": "unexpected", "required_evidence": []}

    with pytest.raises(ValueError, match="Unknown question category"):
        evaluator.assess_retrieval(question, [])
