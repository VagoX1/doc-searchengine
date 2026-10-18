# Document search and conversational RAG

A Python learning project for searching TXT, PDF, and DOCX files, generating
answers from retrieved evidence, and asking follow-up questions in a local Streamlit app.

## Current features

- TF-IDF and local semantic retrieval; the original keyword baseline is retained.
- Paragraph-aware chunks with word-based splitting for long paragraphs.
- PDF page records preserved through chunking, retrieval, and displayed citations.
- Persistent document-index caching and session-scoped conversation history.
- Document-grounded answers with `answered`, `needs_clarification`, or `not_found` status.
- Clickable citation tokens such as S1, with unique anchors for each exchange.
- Follow-up rewriting using the last four exchanges, followed by fresh retrieval.
- Separate retrieval and answer-quality evaluation tools.

The embedding model is `sentence-transformers/all-MiniLM-L6-v2`.
The default answer and follow-up model is `gpt-6-luna`.

## Setup

Run these PowerShell commands from the project root (`doc-searchengine`):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install pytest
```

Activation is optional: using the environment's Python directly avoids PowerShell
activation-policy issues. The examples below use that interpreter.

For generated answers, set your API key in the same terminal before starting the app:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
# Optional overrides:
$env:RAG_CHAT_MODEL = "gpt-6-luna"
$env:RAG_JUDGE_MODEL = "gpt-6-luna"
```

The app does not automatically load a `.env` file. Do not commit keys.
Answer generation and evaluation use the configured API account and incur usage charges.
Search-only does not call the answer API. The embedding model downloads on first use;
subsequent use can use cached model files. Semantic model loading is lazy.

## Run the browser app

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/app.py
```

Open the local URL printed in the terminal, then:

1. Upload a TXT, PDF, or DOCX file.
2. Check the extracted text preview. Editing the preview does not change indexed content.
3. Click **index document** to build an index or load a matching saved index.
4. Enter a query and choose **Search** for passages or **Answer from document** for a cited answer.
5. Use **Answer from document** for follow-ups such as “Can I extend that?” or a
   clarification reply such as “Express”. Plain **Search** uses exactly the entered query.
6. Choose **New conversation** to clear history while keeping the document indexed.

Completed answers retain their question, status, citations, and warnings. Failed answer
requests do not create completed exchanges. Changing or removing the upload clears
both the active index and conversation history. History is separate for each Streamlit
session and is not persisted across server restarts or new sessions.

Follow-up requests send recent questions and answers to the API to resolve references.
Answer generation receives the rewritten question and the top three freshly retrieved
passages. History provides conversational context, not source evidence. Citation links
jump to supporting excerpts in the app, not to an embedded PDF viewer.

## Run command-line search

```powershell
.\.venv\Scripts\python.exe src/main.py "borrow books" --method tfidf --directory ../tests/documents
.\.venv\Scripts\python.exe src/main.py "How long can I keep a book?" --method semantic --directory ../tests/documents
```

`--directory` accepts an absolute folder path. Relative paths are resolved beside
`src/file_loader.py`, not from the terminal directory. Without that option it looks
for `src/documents`. Other options are `--chunk-size`, `--overlap`, and `--top-k`.
The CLI prints source, PDF page when available, chunk ID, score, and a passage preview.
It currently provides search only and rebuilds its in-memory index for each invocation.

## Index cache

The browser app saves chunks and vectors in `.index_cache/*.npz` under the project root.
The key includes file bytes, embedding model, extension, chunk size, overlap, chunk
strategy, and a version number. Renamed identical uploads reuse the index with updated
source labels. Cache contents include extracted text; the folder is excluded from Git.
Deleting a saved cache file makes the next indexing request rebuild it.

Increment the cache version when extraction or chunking behavior changes. An index
belongs to its exact ordered chunk list and embedding model; do not reorder records
or mix embedding models. Dimension checks cannot detect every such mismatch.
Caching avoids recomputing document embeddings; uploads are still extracted on reruns,
and query embeddings still require loading the embedding model.

## Automated checks

```powershell
$env:PYTHONPATH = "src"
$pytestRunPath = Join-Path $PWD (".pytest_cache/run-" + [guid]::NewGuid().ToString("N"))
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp "$pytestRunPath"
```

A unique temporary directory avoids the permission error observed with the default
Windows `pytest-of-Gerraldo` folder. Never point `--basetemp` at a folder containing
files you want to keep: pytest manages and may clear that directory.

The latest successful suite run had **37 passing tests**. Tests use substitute API
clients and embedding responses; they verify logic, not live model quality. Coverage
includes retrieval validation, evaluation fixtures, answer statuses, citations,
follow-up orchestration, history rendering, and exchange-specific anchors.

## Answer-quality evaluation

The fixed practice set in `evaluation/answer_questions.json` has eight questions:
three supported, two ambiguous, and three unanswerable. Its fictional corpus lives
in `evaluation/answer_documents/`, separate from uploaded files.

Retrieval-only evaluation (no answer API calls):

```powershell
.\.venv\Scripts\python.exe src/evaluate_answer_retrieval.py --offline
```

Omit `--offline` to allow embedding-model downloads, or use `--model` with a local
model directory. Reports go to `evaluation/answer_retrieval_results.json` and measure
all-required-evidence and evidence-item recall at k. Unanswerable cases are excluded
from retrieval recall.

Full answer generation and model-based judging:

```powershell
.\.venv\Scripts\python.exe src/evaluate_answer_generation.py --offline-embeddings
# Optional explicit models:
.\.venv\Scripts\python.exe src/evaluate_answer_generation.py --chat-model gpt-6-luna --judge-model gpt-6-luna
```

These commands make answer and judge API requests. `--offline-embeddings` affects
embedding downloads only; it does not make answer evaluation offline.
The judge defaults to `RAG_JUDGE_MODEL` or the selected answer model.
Results go to `evaluation/answer_generation_results.json`, including status accuracy,
answer-point coverage, clarification compliance, citation warnings, unsupported claims,
and separate retrieval/generation failure attribution. The conditional generation
pass rate also includes unanswerable cases, for which retrieval success is inapplicable.
Unavailable pass rates display `n/a`.

Model-based judgments are estimates. Manually inspect judgments and use held-out
questions before treating scores as independent evidence of quality. Citation-ID
validation proves that a reference exists, not that every claim follows from it.

### Earlier retrieval benchmark

`src/evaluate.py` contains the original 12-question TF-IDF/semantic comparison.
Its current `BASE` is the `src` folder although fixtures are under the project-root
`evaluation` folder. That path needs correcting before using this older runner.
Previously saved `evaluation/results*.json` files are historical, not a fresh run.
The answer-evaluation runners above resolve the project-root evaluation folder correctly.

## Source map

| File | Responsibility |
|---|---|
| `src/app.py` | Upload, indexing, search, and conversation UI |
| `src/file_loader.py` | TXT/PDF/DOCX extraction and PDF page metadata |
| `src/chunk_text.py` | Paragraph-aware chunking and metadata preservation |
| `src/semantic_search.py` | Lazy embedding model and semantic retrieval |
| `src/tf_idf.py` | TF-IDF indexing and retrieval |
| `src/index_cache.py` | Disk cache identity and persistence |
| `src/document_qa.py` | Grounded answer generation and citation validation |
| `src/followup.py` | Standalone-question rewriting and fresh retrieval |
| `src/conversation_history.py` | Exchange records and chat/citation rendering |
| `src/main.py` | Command-line search |
| `src/evaluate_answer_*.py` | Retrieval and answer evaluation |
| `src/search_doc.py` | Original keyword baseline |

## Document limits and current verification status

- Scanned PDFs require OCR; this project does not perform it. Blank/unreadable PDF
  pages are skipped with a server-side warning; an unreadable file is skipped by the folder loader.
- DOCX extraction covers body paragraphs and tables, not headers, footers, nested tables,
  or reliable page numbers. PDF columns, tables, and formulas may extract imperfectly.
- Similarity scores are not probabilities. Semantic search returns nearby passages even
  for unrelated questions; abstention depends on answer generation.
- Real PDF/DOCX extraction checks and browser checks for supported answers, abstention,
  bracketed filenames, and clickable citations have been completed.
- Conversation history and follow-up logic have unit tests. Live browser follow-up
  verification remains incomplete.
