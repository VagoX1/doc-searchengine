from types import SimpleNamespace
from pathlib import Path

import followup
import hybrid_search
import index_cache
import semantic_search
from streamlit.testing.v1 import AppTest


def test_app_uses_hybrid_search_for_search_and_answer(monkeypatch):
    chunks = [
        {
            "source": "library.txt",
            "page_number": None,
            "chunk_id": 1,
            "content": "Books must be returned within fourteen days.",
        }
    ]
    vectors = object()
    hybrid_index = hybrid_search.HybridIndex(
        tfidf_vectorizer=object(),
        tfidf_vectors=object(),
        semantic_vectors=vectors,
    )
    results = [
        {
            **chunks[0],
            "score": 1 / 61,
            "rrf_score": 1 / 61,
            "tfidf_rank": 1,
            "semantic_rank": None,
        }
    ]
    answer = SimpleNamespace(
        answer="Return it within fourteen days [S1].",
        status="answered",
        citations=[
            {
                "citation": "S1",
                "source": "library.txt",
                "page_number": None,
                "chunk_id": 1,
                "text": chunks[0]["content"],
            }
        ],
        citation_warnings=[],
    )
    search_calls = []
    answer_calls = []

    monkeypatch.setattr(index_cache, "load_index", lambda _key: None)
    monkeypatch.setattr(index_cache, "save_index", lambda *args: None)
    monkeypatch.setattr(
        semantic_search,
        "build_semantic_index",
        lambda _chunks: vectors,
    )
    monkeypatch.setattr(
        hybrid_search,
        "build_hybrid_index",
        lambda _chunks, semantic_vectors: hybrid_index,
    )

    def fake_search_hybrid(
        document_chunks,
        index,
        query,
        top_k,
        candidate_k,
        rrf_k,
    ):
        search_calls.append((document_chunks, index, query, top_k, candidate_k, rrf_k))
        return results

    def fake_resolve_search_and_answer(
        query,
        history,
        document_chunks,
        index,
        top_k,
        candidate_k,
        rrf_k,
    ):
        answer_calls.append(
            (query, history, document_chunks, index, top_k, candidate_k, rrf_k)
        )
        return "When must I return a book?", results, answer

    monkeypatch.setattr(hybrid_search, "search_hybrid", fake_search_hybrid)
    monkeypatch.setattr(
        followup,
        "resolve_search_and_answer",
        fake_resolve_search_and_answer,
    )

    app_path = Path(__file__).resolve().parents[1] / "src" / "app.py"
    app = AppTest.from_file(app_path).run()
    app.file_uploader[0].upload(
        "library.txt",
        b"Books must be returned within fourteen days.",
        "text/plain",
    ).run()
    assert not app.exception

    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["hybrid_index"] is hybrid_index

    app.text_input[0].set_value("book return")
    next(button for button in app.form_submit_button if button.label == "Search").click().run()
    assert not app.exception
    assert search_calls == [
        (app.session_state["chunks"], hybrid_index, "book return", 3, 20, 60)
    ]
    assert app.session_state["conversation_history"] == []

    app.text_input[0].set_value("When must I return a book?")
    next(
        button
        for button in app.form_submit_button
        if button.label == "Answer from document"
    ).click().run()
    assert not app.exception
    assert answer_calls[0][0] == "When must I return a book?"
    assert answer_calls[0][3] is hybrid_index
    assert answer_calls[0][4:] == (3, 20, 60)
    assert len(app.session_state["conversation_history"]) == 1

    next(button for button in app.button if button.label == "New conversation").click().run()
    assert not app.exception
    assert app.session_state["conversation_history"] == []
    assert app.session_state["hybrid_index"] is hybrid_index
