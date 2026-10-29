"""Streamlit interface for uploading, indexing, searching, and chatting with documents."""

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib

from chunk_text import chunk_documents
from hybrid_search import build_hybrid_index, search_hybrid

from conversation_history import (
    make_exchange,
    render_conversation_history,
    render_exchange,
)

from followup import resolve_search_and_answer

import streamlit as st

from file_loader import extract_document_records

from index_cache import make_cache_key, load_index, save_index
from semantic_search import MODEL_NAME, build_semantic_index

if not st.user.is_logged_in:
    st.title("Document Search")
    st.button("Log in", on_click=st.login)
    st.stop()

# Use this stable authenticated ID for ownership checks in Stage 10.2.
authenticated_user_id = st.user.sub

if st.sidebar.button("Log out"):
    st.logout()

if "conversation_history" not in st.session_state:
    # Keep chat exchanges only for the current Streamlit browser session.
    st.session_state["conversation_history"] = []

st.title("Document Search")

uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    file_id = (
        uploaded_file.name,
        hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    )

    if st.session_state.get("file_id") != file_id:
        # A changed upload invalidates both the index and its conversation context.
        st.session_state["file_id"] = file_id
        for key in ("chunks", "vectors", "hybrid_index"):
            st.session_state.pop(key, None)
        st.session_state["conversation_history"] = []
    try:
        with TemporaryDirectory() as folder:
            suffix = Path(uploaded_file.name).suffix.lower()
            path = Path(folder) / f"document{suffix}"
            path.write_bytes(uploaded_file.getvalue())

            documents = extract_document_records(path)
        
        for document in documents:
            document["source"] = uploaded_file.name

        text = "\n\n".join(document["content"] for document in documents)

        if not text.strip():
            st.warning("No text was extracted. Scanned PDFs may need OCR.")
        else:
            st.success(f"Loaded {uploaded_file.name}")
            st.text_area("Extracted text", text, height=300)

            if st.button("index document"):
                # Reuse the cache when possible; otherwise build vectors for the upload.
                chunk_size = 100
                overlap = 20

                cache_key = make_cache_key(
                    uploaded_file.getvalue(),
                    MODEL_NAME,
                    chunk_size,
                    overlap,
                    file_extension=Path(uploaded_file.name).suffix,
                    chunk_strategy="paragraph",
                )

                with st.spinner("Preparing search index..."):
                    cached = load_index(cache_key)

                    if cached is not None:
                        chunks, vectors = cached

                        chunks = [
                            {**chunk, "source": uploaded_file.name}
                            for chunk in chunks
                        ]

                        st.info("Loaded the saved index.")
                    else:
                        chunks = chunk_documents(
                            documents,
                            chunk_size=chunk_size,
                            overlap=overlap
                        )
                        vectors = build_semantic_index(chunks)
                        save_index(cache_key, chunks, vectors)

                        st.info("Built and saved a new index.")

                    # Build TF-IDF once and reuse the semantic vectors from cache.
                    hybrid_index = build_hybrid_index(
                        chunks,
                        semantic_vectors=vectors,
                    )

                    st.session_state["chunks"] = chunks
                    st.session_state["vectors"] = vectors
                    st.session_state["hybrid_index"] = hybrid_index


            if "hybrid_index" in st.session_state:
                st.success("Document indexed and ready.")

                # A new conversation clears messages without discarding this index.
                if st.button("New conversation", key="new_conversation"):
                    st.session_state["conversation_history"] = []
                    st.rerun()

                render_conversation_history(
                    st.session_state["conversation_history"]
                )

                with st.form("search_form"):
                    # Search stays literal; answer requests use the conversation resolver.
                    query = st.text_input("Search")
                    submitted = st.form_submit_button("Search")
                    answer_submitted = st.form_submit_button("Answer from document")

                if submitted or answer_submitted:
                    results = None

                    if not query.strip():
                        st.warning("Enter a question.")
                    elif submitted:
                        results = search_hybrid(
                            st.session_state["chunks"],
                            st.session_state["hybrid_index"],
                            query,
                            top_k=3,
                            candidate_k=20,
                            rrf_k=60,
                        )
                    else:
                        try:
                            history = st.session_state["conversation_history"]

                            with st.spinner("Resolving question and searching documents..."):
                                # Resolution uses history only for wording; retrieval supplies evidence.
                                _, results, answer = resolve_search_and_answer(
                                    query,
                                    history,
                                    st.session_state["chunks"],
                                    st.session_state["hybrid_index"],
                                    top_k=3,
                                    candidate_k=20,
                                    rrf_k=60
                                )

                        except Exception as exc:
                            st.error(f"Answer generation failed: {exc}")

                        else:
                            # Save only after resolving, retrieving, and answering succeeded.
                            exchange = make_exchange(query, answer)
                            history.append(exchange)
                            render_exchange(exchange, len(history))

                    if results is not None:
                        st.subheader("Retrieved passages")

                        for result in results:
                            page_number = result.get("page_number")

                            if page_number is None:
                                location = (
                                    f"{result['source']} - chunk "
                                    f"{result['chunk_id']}"
                                )
                            else:
                                location = (
                                    f"{result['source']} - PDF page "
                                    f"{page_number} - chunk {result['chunk_id']}"
                                )

                            st.write(f"**{location}**")
                            st.caption(
                                f"RRF score: {result['rrf_score']:.3f} | "
                                f"TF-IDF rank: {result.get('tfidf_rank') or '—'} | "
                                f"semantic rank: {result.get('semantic_rank') or '—'}"
                            )
                            st.write(result["content"])

    except Exception as e:
        st.error(f"Document processing failed: {e}")

if uploaded_file is None:
    st.session_state["conversation_history"] = []
    for key in ("file_id", "chunks", "vectors", "hybrid_index"):
        st.session_state.pop(key, None)
