chunks = chunk_documents(documents, chunk_size=100, overlap=20)
model = get_model(model_name, local_files_only=offline)

# Build once, outside the loop.
hybrid_index = build_hybrid_index(chunks, model=model)

for question in questions:
    results = search_hybrid(
        chunks,
        hybrid_index,
        question["question"],
        top_k=3,
        candidate_k=20,
        rrf_k=60,
        model=model,
    )