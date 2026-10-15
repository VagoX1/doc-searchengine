from validation import validate_documents, validate_chunk_settings


def chunk_text(text, chunk_size=100, overlap=20):
    validate_chunk_settings(chunk_size, overlap)
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(documents, chunk_size=100, overlap=20):
    validate_chunk_settings(chunk_size, overlap)
    validate_documents(documents)
    records = []
    for document in documents:
        for number, content in enumerate(chunk_text(document["content"], chunk_size, overlap), start=1):
            records.append({**document, "chunk_id": number, "content": content})
    return records
