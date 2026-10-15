import logging
from pathlib import Path

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def extract_text_from_file(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8-sig")
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix == ".docx":
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        parts = []
        for block in Document(path).iter_inner_content():
            if isinstance(block, Paragraph):
                parts.append(block.text)
            elif isinstance(block, Table):
                parts.extend("\t".join(cell.text for cell in row.cells) for row in block.rows)
        return "\n".join(parts)
    raise ValueError(f"Unsupported file type: {suffix}")


def load_documents(folder_path):
    folder = Path(__file__).resolve().parent / folder_path
    if not folder.is_dir():
        raise NotADirectoryError(f"Document folder not found: {folder}")

    documents = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            text = extract_text_from_file(path)
        except ImportError:
            raise  # Missing dependencies require installation, not silent skipping.
        except Exception as exc:
            logger.warning("Could not read %s: %s", path.name, exc)
            continue
        if not text.strip():
            logger.warning("No extracted text in %s; scanned files may need OCR", path.name)
            continue
        documents.append({"source": path.name, "content": text, "file_type": path.suffix.lower()})
    return documents


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(load_documents("documents"))
