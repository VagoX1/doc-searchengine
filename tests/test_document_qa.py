from types import SimpleNamespace

import pytest

from document_qa import ABSTENTION, answer_from_context


class FakeResponses:
    def __init__(self, output_text):
        self.output_text = output_text
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
        "Return the books within fourteen days [manual.pdf#page-4-chunk-2]."
    )

    result = answer_from_context(
        "When are books due?", chunks, client=client, model_name="test-model"
    )

    assert result.retrieved_chunks == 1
    assert result.citations[0]["source"] == "manual.pdf"
    assert result.citations[0]["page_number"] == 4
    assert result.citation_warnings == []
    assert "[manual.pdf#page-4-chunk-2]" in client.responses.call["input"][0]["content"]
    assert client.responses.call["store"] is False


def test_unavailable_citation_is_reported():
    chunks = [
        {
            "source": "guide.txt",
            "chunk_id": 1,
            "page_number": None,
            "content": "The guide explains setup.",
        }
    ]
    client = FakeClient("Install the package [other.txt#chunk-9].")

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
    client = FakeClient("Answer [manual.pdf#page-4-chunk-1].")

    with pytest.raises(ValueError, match="Duplicate citation ID"):
        answer_from_context("Question?", [chunk, chunk], client=client)
