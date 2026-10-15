from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib

from chunk_text import chunk_documents
from semantic_search import build_semantic_index, search_semantic

import streamlit as st

from file_loader import extract_text_from_file

from index_cache import make_cache_key, load_index, save_index
from semantic_search import MODEL_NAME

st.title("Document Search")

uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    file_id = (
        uploaded_file.name,
        hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    )

    if st.session_state.get("file_id") != file_id:
        st.session_state["file_id"] = file_id
        st.session_state.pop("chunks", None)
        st.session_state.pop("vectors", None)


    try:
        with TemporaryDirectory() as folder:
            suffix = Path(uploaded_file.name).suffix.lower()
            path = Path(folder) / f"document{suffix}"
            path.write_bytes(uploaded_file.getvalue())
            text = extract_text_from_file(path)
        
        if not text.strip():
            st.warning("No text was extracted. Scanned PDFs may need OCR.")
        else:
            st.success(f"Loaded {uploaded_file.name}")
            st.text_area("Extracted text", text, height=300)

            if st.button("index document"):
                chunk_size = 100
                overlap = 20

                cache_key = make_cache_key(
                    uploaded_file.getvalue(),
                    MODEL_NAME,
                    chunk_size,
                    overlap,
                    file_extension=Path(uploaded_file.name).suffix
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
                        documents = [{
                            "source": uploaded_file.name,
                            "content": text
                        }]

                        chunks = chunk_documents(
                            documents,
                            chunk_size=chunk_size,
                            overlap=overlap
                        )
                        vectors = build_semantic_index(chunks)
                        save_index(cache_key, chunks, vectors)

                        st.info("Built and saved a new index.")

                    st.session_state["chunks"] = chunks
                    st.session_state["vectors"] = vectors

            if "vectors" in st.session_state:
                st.success("Document indexed and ready.")

                with st.form("search_form"):
                    query = st.text_input("Search")
                    submitted = st.form_submit_button("Search")

                if submitted:
                    if not query.strip():
                        st.warning("Enter a question.")
                    else:
                        results = search_semantic(
                            st.session_state["chunks"],
                            st.session_state["vectors"],
                            query,
                            top_k=3
                        )

                        for result in results:
                            st.write(
                                f"**{result['source']} - "
                                f"chunk {result['chunk_id']}**"
                            )
                            st.caption(f"similarity: {result['score']:.3f}")
                            st.write(result["content"])

    except Exception as e:
        st.error(f"Document processing failed: {e}")

if uploaded_file is None:
    for key in ("file_id", "chunks", "vectors"):
        st.session_state.pop(key, None)