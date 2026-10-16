import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANSWER_EVALUATION_DIR = PROJECT_ROOT / "evaluation"
QUESTION_SET = ANSWER_EVALUATION_DIR / "answer_questions.json"
DOCUMENT_DIR = ANSWER_EVALUATION_DIR / "answer_documents"


def normalize(text):
    return " ".join(text.split()).lower()


def load_question_set():
    return json.loads(QUESTION_SET.read_text(encoding="utf-8"))


def test_question_set_covers_each_expected_category():
    questions = load_question_set()
    categories = {question["category"] for question in questions}
    category_counts = {
        category: sum(question["category"] == category for question in questions)
        for category in categories
    }

    assert categories == {"supported", "ambiguous", "unanswerable"}
    assert category_counts == {"supported": 3, "ambiguous": 2, "unanswerable": 3}
    assert len({question["id"] for question in questions}) == len(questions)
    assert all(question["question"].strip() for question in questions)
    assert all(
        question["expected_answer_points"]
        and all(point.strip() for point in question["expected_answer_points"])
        for question in questions
    )


def test_expected_statuses_and_evidence_match_category():
    questions = load_question_set()
    expected_status = {
        "supported": "answered",
        "ambiguous": "needs_clarification",
        "unanswerable": "not_found",
    }

    for question in questions:
        category = question["category"]
        required_evidence = question["required_evidence"]

        assert question["expected_status"] == expected_status[category]
        assert isinstance(question["must_ask_clarifying_question"], bool)
        assert question["must_ask_clarifying_question"] is (category == "ambiguous")

        if category == "supported":
            assert required_evidence
        elif category == "ambiguous":
            assert len(required_evidence) >= 2
        else:
            assert required_evidence == []

        for evidence in required_evidence:
            assert evidence["source"].strip()
            assert evidence["text"].strip()


def test_all_required_evidence_exists_in_the_answer_fixtures():
    questions = load_question_set()
    document_text = {
        path.name: normalize(path.read_text(encoding="utf-8"))
        for path in DOCUMENT_DIR.glob("*.txt")
    }

    assert document_text, "Expected answer-evaluation fixture documents."

    for question in questions:
        for evidence in question["required_evidence"]:
            source = evidence["source"]
            expected_text = normalize(evidence["text"])

            assert source in document_text, (
                f"Question {question['id']} references missing source {source}"
            )
            assert expected_text in document_text[source], (
                f"Question {question['id']} has evidence absent from {source}"
            )
