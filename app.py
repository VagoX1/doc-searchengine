from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib

from chunk_text import chunk_documents
from semantic_search import build_semantic_index, search_semantic

import streamlit as st

from file_loader import extract_text_from_file

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
                documents = [{
                    "source": uploaded_file.name,
                    "content": text
                }]

                with st.spinner("Indexing document..."):
                    chunks = chunk_documents(documents)

                    if not chunks:
                        st.warning("No text available to index")
                    else:
                        vectors = build_semantic_index(chunks)
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

    except Exception as e:
        st.error(f"Could not read the document: {e}")