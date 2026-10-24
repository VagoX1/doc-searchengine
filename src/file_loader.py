"""Extract text and source metadata from TXT, PDF, and DOCX files."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def extract_document_records(path):
    """Extract one or more text records from a supported document file.

    Text files and DOCX files produce a single record; PDFs produce one record
    per nonempty page so page citations can be retained.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    records = []

    if suffix == ".txt":
        text = path.read_text(encoding="utf-8-sig")
        if text.strip():
            records.append({
                "source": path.name,
                "file_type": suffix,
                "page_number": None,
                "content": text,
            })
        return records

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)

        # Keep page numbers attached to extracted text for later citations.
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.warning(
                    "Could not extract %s, PDF page %d: %s",
                    path.name,
                    page_number,
                    e
                )
                continue
            
            if not text.strip():
                logger.warning(
                    "No text extracted from %s, PDF page %d",
                    path.name,
                    page_number,
                )
                continue

            records.append({
                "source": path.name,
                "file_type": suffix,
                "page_number": page_number,
                "content": text,
            })

        return records

    if suffix == ".docx":
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        parts = []

        # Read body paragraphs and tables in their document order.
        for block in Document(path).iter_inner_content():
            if isinstance(block, Paragraph):
                if block.text.strip():
                    parts.append(block.text)

            elif isinstance(block, Table):
                for row in block.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    parts.append("\t".join(cells))

        text = "\n\n".join(parts)

        if text.strip():
            records.append({
                "source": path.name,
                "file_type": suffix,
                "page_number": None,
                "content": text,
            })

        return records

    raise ValueError(f"Unsupported file type: {suffix}")

def extract_text_from_file(path):
    """Return extracted text joined across all records in a file."""
    records = extract_document_records(path)
    return "\n\n".join(record["content"] for record in records)



def load_documents(folder_path):
    """Load supported files in a folder, warning and skipping unreadable files."""
    folder = Path(__file__).resolve().parent / folder_path
    if not folder.is_dir():
        raise NotADirectoryError(f"Document folder not found: {folder}")

    documents = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            documents.extend(extract_document_records(path))
        except ImportError:
            raise  # Missing dependencies require installation, not silent skipping.
        except Exception as exc:
            logger.warning(
                "Could not read %s: %s", 
                path.name,
                exc)
            continue
    return documents


if __name__ == "__main__":
    # Allow quick manual inspection of the default documents folder.
    logging.basicConfig(level=logging.INFO)
    print(load_documents("documents"))
