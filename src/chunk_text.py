from validation import validate_documents, validate_chunk_settings
import re


def chunk_text(text, chunk_size=100, overlap=20):
    validate_chunk_settings(chunk_size, overlap)

    if not isinstance(text, str):
        raise ValueError("Text must be a string")

    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n[ \t]*\n", normalized_text)
        if paragraph.strip()
    ]

    chunks = []
    current_paragraphs = []
    current_word_count = 0

    def flush_current():
        nonlocal current_paragraphs, current_word_count
        if current_paragraphs:
            chunks.append("\n\n".join(current_paragraphs))
            current_paragraphs = []
            current_word_count = 0
    
    for paragraph in paragraphs:
        paragraph_words = paragraph.split()

        if len(paragraph_words) > chunk_size:
            flush_current()

            paragraph_chunks = split_long_paragraph(
                paragraph,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            chunks.extend(paragraph_chunks[:-1])
            current_paragraphs = [paragraph_chunks[-1]]
            current_word_count = len(paragraph_chunks[-1].split())

            continue

        would_exceed_limit = (
            current_word_count + len(paragraph_words) > chunk_size
        )

        if current_paragraphs and would_exceed_limit:
            previous_text = "\n\n".join(current_paragraphs)
            flush_current()

            overlap_count = min(overlap, chunk_size - len(paragraph_words))
            overlap_words = (
                previous_text.split()[-overlap_count:]
                if overlap_count
                else []
            )

            if overlap_words:
                current_paragraphs = [" ".join(overlap_words)]
                current_word_count = len(overlap_words)
        
        current_paragraphs.append(paragraph)
        current_word_count += len(paragraph_words)

    flush_current()

    return chunks

def split_long_paragraph(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break
    
    return chunks

def chunk_documents(documents, chunk_size=100, overlap=20):
    validate_chunk_settings(chunk_size, overlap)
    validate_documents(documents)

    records = []

    for document in documents:
        chunks = chunk_text(
            document["content"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_number, content in enumerate(chunks, start=1):
            records.append({
                **document,
                "chunk_id": chunk_number,
                "content": content,
            })

    return records