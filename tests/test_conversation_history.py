from contextlib import contextmanager
from types import SimpleNamespace

import conversation_history as conversation


class FakeStreamlit:
    def __init__(self):
        self.events = []

    @contextmanager
    def chat_message(self, role):
        self.events.append(("role", role))
        yield

    def markdown(self, value, **kwargs):
        self.events.append(("markdown", value, kwargs))

    def caption(self, value):
        self.events.append(("caption", value))

    def write(self, value):
        self.events.append(("write", value))

    def warning(self, value):
        self.events.append(("warning", value))


def make_answer():
    return SimpleNamespace(
        answer="Return it within fourteen days [S1].",
        status="answered",
        citations=[
            {
                "citation": "S1",
                "source": "library.txt",
                "page_number": None,
                "chunk_id": 1,
                "text": "Books must be returned within fourteen days.",
            }
        ],
        citation_warnings=["Example citation warning."],
    )


def test_make_exchange_stores_complete_answer_record():
    answer = make_answer()

    exchange = conversation.make_exchange(
        "When must I return a book?",
        answer,
    )

    assert exchange == {
        "question": "When must I return a book?",
        "answer": "Return it within fourteen days [S1].",
        "status": "answered",
        "citations": answer.citations,
        "citation_warnings": answer.citation_warnings,
    }
    assert exchange["citations"] is not answer.citations
    assert exchange["citation_warnings"] is not answer.citation_warnings


def test_history_renders_each_exchange_in_order_with_unique_citation_anchors(
    monkeypatch,
):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(conversation, "st", fake_st)

    first = conversation.make_exchange("When must I return a book?", make_answer())
    second = conversation.make_exchange("Where is the recovery link sent?", make_answer())

    conversation.render_conversation_history([first, second])

    roles = [value for kind, value, *_ in fake_st.events if kind == "role"]
    assert roles == ["user", "assistant", "user", "assistant"]

    rendered_markdown = [
        event[1] for event in fake_st.events if event[0] == "markdown"
    ]
    assert "When must I return a book?" in rendered_markdown
    assert "Where is the recovery link sent?" in rendered_markdown
    assert (
        "Return it within fourteen days [S1](#passage-exchange-1-s1)."
        in rendered_markdown
    )
    assert (
        "Return it within fourteen days [S1](#passage-exchange-2-s1)."
        in rendered_markdown
    )

    anchors = [
        event[1]
        for event in fake_st.events
        if event[0] == "markdown" and "<div id=" in event[1]
    ]
    assert anchors == [
        '<div id="passage-exchange-1-s1"></div>',
        '<div id="passage-exchange-2-s1"></div>',
    ]
    assert sum(kind == "warning" for kind, *_ in fake_st.events) == 2
