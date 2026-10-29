import json
from types import SimpleNamespace

import pytest

import followup


class FakeResponses:
    def __init__(self, output_text):
        self.output_text = output_text
        self.call = None

    def create(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, output):
        self.responses = FakeResponses(json.dumps(output))


def exchange(question, answer, status="answered"):
    return {
        "question": question,
        "answer": answer,
        "status": status,
        "citations": [],
        "citation_warnings": [],
    }


def test_resolver_returns_original_question_without_history():
    client = FakeClient({"standalone_question": "unused"})

    result = followup.resolve_followup_question(
        "  Where is the account recovery link sent?  ",
        [],
        client=client,
    )

    assert result == "Where is the account recovery link sent?"
    assert client.responses.call is None


@pytest.mark.parametrize(
    ("query", "history", "standalone_question"),
    [
        (
            "Can I extend that?",
            [exchange("When must I return a book?", "Return it within fourteen days.")],
            "Can I extend the return period for a borrowed library book?",
        ),
        (
            "Express",
            [
                exchange(
                    "How long does shipping take?",
                    "Standard takes three to five days or express takes one to two. "
                    "Which option did you choose?",
                    status="needs_clarification",
                )
            ],
            "How long does express shipping take?",
        ),
        (
            "Where is the account recovery link sent?",
            [exchange("When must I return a book?", "Return it within fourteen days.")],
            "Where is the account recovery link sent?",
        ),
    ],
)
def test_resolver_uses_recent_turns_and_returns_standalone_question(
    monkeypatch, query, history, standalone_question
):
    monkeypatch.delenv("RAG_CHAT_MODEL", raising=False)
    client = FakeClient({"standalone_question": standalone_question})

    result = followup.resolve_followup_question(query, history, client=client)

    assert result == standalone_question
    assert client.responses.call["model"] == "gpt-6-luna"
    assert client.responses.call["store"] is False
    assert "clearly changes topic" in client.responses.call["instructions"]
    payload = json.loads(client.responses.call["input"][0]["content"])
    assert payload["latest_user_message"] == query
    assert payload["recent_conversation"] == [
        {
            "question": history[0]["question"],
            "answer": history[0]["answer"],
            "status": history[0]["status"],
        }
    ]
    assert "citations" not in payload["recent_conversation"][0]


def test_resolver_limits_context_to_the_last_four_exchanges():
    history = [
        exchange(f"Question {number}", f"Answer {number}")
        for number in range(6)
    ]
    client = FakeClient({"standalone_question": "What about answer five?"})

    followup.resolve_followup_question("What about that?", history, client=client)

    payload = json.loads(client.responses.call["input"][0]["content"])
    assert [turn["question"] for turn in payload["recent_conversation"]] == [
        "Question 2",
        "Question 3",
        "Question 4",
        "Question 5",
    ]


def test_resolver_rejects_invalid_structured_output():
    client = SimpleNamespace(
        responses=FakeResponses("not json")
    )

    with pytest.raises(ValueError, match="invalid structured output"):
        followup.resolve_followup_question(
            "Can I extend that?",
            [exchange("When must I return a book?", "Return in fourteen days.")],
            client=client,
        )


def test_resolve_search_and_answer_uses_only_fresh_retrieval(
    monkeypatch,
):
    history = [exchange("When must I return a book?", "Return it within fourteen days.")]
    chunks = [{"source": "library.txt", "content": "indexed text"}]
    hybrid_index = object()
    fresh_results = [
        {
            "source": "library.txt",
            "page_number": None,
            "chunk_id": 1,
            "score": 0.91,
            "content": "Books must be returned within fourteen days.",
        }
    ]
    answer = SimpleNamespace(answer="A fresh answer [S1].")
    calls = {}

    def fake_resolve(query, recent_history, client):
        calls["resolve"] = (query, recent_history, client)
        return "Can I extend the return period for a borrowed library book?"

    def fake_search(documents, index, query, top_k, candidate_k, rrf_k):
        calls["search"] = (
            documents,
            index,
            query,
            top_k,
            candidate_k,
            rrf_k,
        )
        return fresh_results

    def fake_answer(question, retrieved, client):
        calls["answer"] = (question, retrieved, client)
        return answer

    monkeypatch.setattr(followup, "resolve_followup_question", fake_resolve)
    monkeypatch.setattr(followup, "search_hybrid", fake_search)
    monkeypatch.setattr(followup, "answer_from_context", fake_answer)
    client = object()

    resolved, retrieved, result = followup.resolve_search_and_answer(
        "Can I extend that?",
        history,
        chunks,
        hybrid_index,
        client=client,
    )

    assert resolved == "Can I extend the return period for a borrowed library book?"
    assert calls["resolve"] == ("Can I extend that?", history, client)
    assert calls["search"] == (chunks, hybrid_index, resolved, 3, 20, 60)
    assert calls["answer"] == (resolved, fresh_results, client)
    assert retrieved is fresh_results
    assert result is answer
