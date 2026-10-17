import json
from types import SimpleNamespace

import evaluate_answer_generation as evaluator


class FakeResponses:
    def __init__(self, payload):
        self.output_text = json.dumps(payload)
        self.call = None

    def create(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, payload):
        self.responses = FakeResponses(payload)


def make_question():
    return {
        "id": "supported-01",
        "category": "supported",
        "question": "How long may I keep a borrowed library book?",
        "expected_status": "answered",
        "required_evidence": [
            {
                "source": "library.txt",
                "text": "Books must be returned within fourteen days.",
            }
        ],
        "expected_answer_points": [
            "The return period is fourteen days."
        ],
        "must_ask_clarifying_question": False,
        "retrieval": {
            "retrieval_applicable": True,
            "retrieval_ok": True,
            "missing_evidence": [],
            "failure_layer": None,
        },
        "retrieved": [
            {
                "source": "library.txt",
                "page_number": None,
                "chunk_id": 1,
                "score": 0.9,
                "content": "Books must be returned within fourteen days.",
            }
        ],
    }


def passing_judgment():
    return {
        "answer_point_results": [{"point_id": "P1", "covered": True}],
        "clarification_asked": False,
        "unsupported_claims": [],
    }


def test_judge_answer_sends_expected_points_and_passages_to_responses_api():
    question = make_question()
    client = FakeClient(passing_judgment())

    result = evaluator.judge_answer(
        question,
        "Books are due within fourteen days [S1].",
        question["retrieved"],
        client,
        "gpt-6-luna",
    )

    assert result == passing_judgment()
    assert client.responses.call["model"] == "gpt-6-luna"
    assert client.responses.call["store"] is False
    payload = json.loads(client.responses.call["input"][0]["content"])
    assert payload["expected_answer_points"] == [
        {"point_id": "P1", "text": "The return period is fourteen days."}
    ]
    assert payload["retrieved_passages"][0]["citation_id"] == "S1"


def test_score_case_uses_judgment_and_separates_retrieval_failure():
    question = make_question()
    answer = SimpleNamespace(
        status="answered",
        citation_warnings=[],
        citations=[{"citation": "S1"}],
    )

    result = evaluator.score_case(
        question,
        answer,
        passing_judgment(),
        question["retrieval"],
    )

    assert result["generation_pass"] is True
    assert result["primary_failure"] is None

    question["retrieval"]["retrieval_ok"] = False
    result = evaluator.score_case(
        question,
        answer,
        passing_judgment(),
        question["retrieval"],
    )
    assert result["primary_failure"] == "retrieval"


def test_evaluation_defaults_answer_and_judge_models_to_gpt_6(tmp_path, monkeypatch):
    question = make_question()
    calls = {}

    monkeypatch.delenv("RAG_CHAT_MODEL", raising=False)
    monkeypatch.delenv("RAG_JUDGE_MODEL", raising=False)
    monkeypatch.setattr(evaluator, "EVALUATION_DIR", tmp_path)
    monkeypatch.setattr(
        evaluator,
        "evaluate_answer_retrieval",
        lambda **kwargs: {"details": [question]},
    )

    def fake_answer_from_context(question_text, retrieved, client, model_name):
        calls["chat_model"] = model_name
        return SimpleNamespace(
            answer="Books are due within fourteen days [S1].",
            status="answered",
            citation_warnings=[],
            citations=[{"citation": "S1"}],
        )

    def fake_judge_answer(question, answer, retrieved, client, judge_model):
        calls["judge_model"] = judge_model
        return passing_judgment()

    monkeypatch.setattr(evaluator, "answer_from_context", fake_answer_from_context)
    monkeypatch.setattr(evaluator, "judge_answer", fake_judge_answer)

    report = evaluator.evaluate_answer_generation(client=object())

    assert calls == {"chat_model": "gpt-6-luna", "judge_model": "gpt-6-luna"}
    assert report["metrics"]["generation_pass_rate_when_retrieval_succeeded"] == 1.0
    assert (tmp_path / "answer_generation_results.json").is_file()

def test_evaluation_handles_no_generation_evaluable_cases(
    tmp_path, monkeypatch, capsys
):
    question = make_question()

    # Retrieval failed for the only supported question.
    question["retrieval"]["retrieval_ok"] = False
    question["retrieval"]["missing_evidence"] = question["required_evidence"]

    monkeypatch.setattr(evaluator, "EVALUATION_DIR", tmp_path)

    monkeypatch.setattr(
        evaluator,
        "evaluate_answer_retrieval",
        lambda **kwargs: {"details": [question]}
    )

    monkeypatch.setattr(
        evaluator,
        "answer_from_context",
        lambda *args, **kwargs: SimpleNamespace(
            answer="Books are due within fourteen days [S1].",
            status="answered",
            citation_warnings=[],
            citations=[{"citation": "S1"}]
        )
    )

    monkeypatch.setattr(
        evaluator,
        "judge_answer",
        lambda *args, **kwargs: passing_judgment()
    )

    report = evaluator.evaluate_answer_generation(client=object())

    assert (
        report["metrics"]["generation_pass_rate_when_retrieval_succeeded"]
        is None
    )
    assert "n/a" in capsys.readouterr().out
    assert (tmp_path / "answer_generation_results.json").is_file()