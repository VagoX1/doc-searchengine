import json
from types import SimpleNamespace

import pytest

from document_qa import ABSTENTION, answer_from_context


class FakeResponses:
    def __init__(self, output_text):
        self.output_text = json.dumps({"status": "answered", "answer": output_text})
        self.call = None

    def create(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, output_text):
        self.responses = FakeResponses(output_text)


def test_answer_uses_retrieved_pdf_page_and_validates_its_citation():
    chunks = [
        {
            "source": "manual.pdf",
            "file_type": ".pdf",
            "page_number": 4,
            "chunk_id": 2,
            "content": "Return borrowed books within fourteen days.",
            "score": 0.8,
        }
    ]
    client = FakeClient(
        "Return the books within fourteen days [S1]."
    )

    result = answer_from_context(
        "When are books due?", chunks, client=client, model_name="test-model"
    )

    assert result.retrieved_chunks == 1
    assert result.citations[0]["source"] == "manual.pdf"
    assert result.citations[0]["page_number"] == 4
    assert result.citation_warnings == []
    assert "[S1]" in client.responses.call["input"][0]["content"]
    assert client.responses.call["store"] is False


def test_answer_defaults_to_gpt_6(monkeypatch):
    monkeypatch.delenv("RAG_CHAT_MODEL", raising=False)
    client = FakeClient("Fourteen days [S1].")
    chunk = {"source": "library.txt", "chunk_id": 1, "content": "Fourteen days."}

    answer_from_context("When are books due?", [chunk], client=client)

    assert client.responses.call["model"] == "gpt-6-luna"


def test_unavailable_citation_is_reported():
    chunks = [
        {
            "source": "guide.txt",
            "chunk_id": 1,
            "page_number": None,
            "content": "The guide explains setup.",
        }
    ]
    client = FakeClient("Install the package [S99].")

    result = answer_from_context("How do I install it?", chunks, client=client)

    assert result.citations == []
    assert any("unavailable citation" in warning for warning in result.citation_warnings)
    assert any("were not cited" in warning for warning in result.citation_warnings)


def test_empty_retrieval_abstains_without_calling_model():
    client = FakeClient("This should not be returned.")

    result = answer_from_context("Question?", [], client=client)

    assert result.answer == ABSTENTION
    assert result.retrieved_chunks == 0
    assert client.responses.call is None


def test_duplicate_citation_ids_are_rejected():
    chunk = {
        "source": "manual.pdf",
        "page_number": 4,
        "chunk_id": 1,
        "content": "A passage.",
    }
    client = FakeClient("Answer [S1].")

    with pytest.raises(ValueError, match="Duplicate citation ID"):
        answer_from_context("Question?", [chunk, chunk], client=client)


def test_abstention_with_retrieved_passages_does_not_warn():
    client = FakeClient("unused")
    client.responses.output_text = json.dumps({"status": "not_found", "answer": "No evidence."})
    result = answer_from_context("Unrelated?", [{"source": "a.txt", "chunk_id": 1, "content": "Books."}], client=client)
    assert result.status == "not_found"
    assert result.answer == ABSTENTION
    assert result.citation_warnings == []
    assert result.citations == []


def test_needs_clarification_returns_both_cited_alternatives():
    chunks = [
        {
            "source": "shipping-options.txt",
            "chunk_id": 1,
            "page_number": None,
            "content": "Standard shipping takes three to five business days.",
        },
        {
            "source": "shipping-options.txt",
            "chunk_id": 2,
            "page_number": None,
            "content": "Express shipping takes one to two business days.",
        },
    ]
    answer_text = (
        "Standard shipping takes three to five business days [S1], while express "
        "takes one to two business days [S2]. Which option did you choose?"
    )
    client = FakeClient(answer_text)
    client.responses.output_text = json.dumps(
        {"status": "needs_clarification", "answer": answer_text}
    )

    result = answer_from_context("How long does shipping take?", chunks, client=client)

    assert result.status == "needs_clarification"
    assert [citation["citation"] for citation in result.citations] == ["S1", "S2"]
    assert result.citation_warnings == []
    assert "needs_clarification" in client.responses.call["text"]["format"]["schema"]["properties"]["status"]["enum"]


def test_bracketed_filename_and_clickable_citation():
    from document_qa import link_answer_citations
    chunk = {"source": "manual [final].pdf", "page_number": 3, "chunk_id": 1, "content": "Fourteen days."}
    result = answer_from_context("When?", [chunk], client=FakeClient("Fourteen days [S1]."))
    assert result.citation_warnings == []
    assert result.citations[0]["source"] == chunk["source"]
    assert result.citations[0]["page_number"] == 3
    assert link_answer_citations(result) == "Fourteen days [S1](#passage-s1)."


def test_substantive_answer_without_citations_still_warns():
    chunk = {"source": "a.txt", "chunk_id": 1, "content": "Fourteen days."}
    result = answer_from_context("When?", [chunk], client=FakeClient("Fourteen days."))
    assert result.citation_warnings


@pytest.mark.parametrize("payload", ["not json", '{"status":"unknown","answer":"text"}', '{"status":"answered","answer":""}'])
def test_invalid_model_output_is_rejected(payload):
    client = FakeClient("unused")
    client.responses.output_text = payload
    with pytest.raises(ValueError):
        answer_from_context("When?", [{"source": "a", "chunk_id": 1, "content": "Text"}], client=client)
