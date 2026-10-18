"""Session-scoped conversation records and Streamlit chat rendering."""
import streamlit as st

from document_qa import link_citation_tokens


def make_exchange(question, answer):
    """Copy the completed answer data into a session-safe exchange record."""
    return {
        "question": question,
        "answer": answer.answer,
        "status": answer.status,
        "citations": [dict(citation) for citation in answer.citations],
        "citation_warnings": list(answer.citation_warnings),
    }


def render_exchange(exchange, exchange_number):
    """Render one question/answer pair and its cited passages."""
    anchor_prefix = f"exchange-{exchange_number}"

    with st.chat_message("user"):
        st.markdown(exchange["question"])

    with st.chat_message("assistant"):
        st.markdown(
            link_citation_tokens(
                exchange["answer"],
                exchange["citations"],
                anchor_prefix=anchor_prefix,
            )
        )
        st.caption(f"Status: {exchange['status']}")

        if exchange["citations"]:
            st.markdown("**Cited passages**")
            for citation in exchange["citations"]:
                page_number = citation["page_number"]
                if page_number is None:
                    location = (
                        f"{citation['source']} - chunk {citation['chunk_id']}"
                    )
                else:
                    location = (
                        f"{citation['source']} - PDF page {page_number} "
                        f"- chunk {citation['chunk_id']}"
                    )

                anchor = (
                    f"passage-{anchor_prefix}-{citation['citation'].lower()}"
                )
                st.markdown(
                    f'<div id="{anchor}"></div>',
                    unsafe_allow_html=True,
                )
                st.write(f"**[{citation['citation']}] {location}**")
                st.write(citation["text"])

        for warning in exchange["citation_warnings"]:
            st.warning(warning)


def render_conversation_history(history):
    """Render saved exchanges in their original order."""
    for exchange_number, exchange in enumerate(history, start=1):
        render_exchange(exchange, exchange_number)
