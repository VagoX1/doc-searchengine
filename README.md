# Document search practice project

Search TXT, text-based PDF, and DOCX documents using TF-IDF or local embeddings,
then optionally generate answers from retrieved passages.
Scanned PDFs need OCR. PDF results retain page numbers; identify a passage by its source,
page (for PDFs), and chunk number.
DOCX body paragraphs and tables are extracted; headers and nested tables are not covered.

## Setup and search

Run from this folder:

```powershell
python -m pip install -r requirements.txt
python main.py "borrow books" --method tfidf
python main.py "How long can I keep a borrowed book?" --method semantic
python -m streamlit run src/app.py
```

The semantic model downloads on first use. Imports and TF-IDF searches do not load it.
Relative document folders are resolved beside file_loader.py; absolute paths also work.
Use --directory, --chunk-size, --overlap, and --top-k to change settings.
Semantic scores are similarities, not probabilities. Nearest passages may not answer a question.

## Document-grounded answers

The Streamlit app keeps retrieval local using its existing sentence-transformer index.
Choosing **Answer from document** sends only the top three retrieved passages to the
OpenAI Responses API for answer generation; the RAG project's document-only citation and
abstention approach is adapted in `src/document_qa.py`. Search-only does not call the API.

Set `OPENAI_API_KEY` in your environment before generating answers. The model defaults to
`gpt-6-luna` and can be changed with `RAG_CHAT_MODEL`.
Never put the API key in source code or commit it. Generated answers validate citation IDs
against the passages supplied to the model and show the cited source/page excerpts.
Citations such as [S1] link to the supporting excerpt and do not depend on filenames.
Structured answer status distinguishes unsupported questions from uncited answers;
valid abstentions do not trigger missing-citation warnings. Citation-ID checks do not
prove that every generated claim is supported by its excerpt.

## Stages 1 and 2

- validation.py: shared input checks.
- file_loader.py: supported file extraction; warns and skips unreadable or empty files.
- chunk_text.py: paragraph-aware, overlapping chunks and source metadata.
- tf_idf.py: build_index once, search_index repeatedly; search_tfidf is a convenience wrapper.
- semantic_search.py: lazy model loading and reusable in-memory document vectors.
- document_qa.py: document-only answer generation and citation validation, adapted from the RAG starter.
- main.py: command-line orchestration and output.
- search_doc.py: original keyword baseline, kept for learning.

An index belongs to the exact ordered records and embedding model used to create it.
Do not reorder records or mix models between indexing and querying. Row-count checks
cannot detect reordered records or different models with identical dimensions.
Each command starts a new process and rebuilds the index. Persistent caching is stage 4.

## Checks

```powershell
python -m unittest -v test_search
```

Unit checks use a fake embedding model to verify plumbing without downloads;
they do not measure semantic quality. PDF/DOCX extraction needs separate real-file checks.

## Retrieval evaluation

```powershell
python evaluate.py
python evaluate.py --method tfidf
python evaluate.py --offline --model "path/to/local/model"
```

The benchmark uses six fictional documents and 12 answerable questions, half literal
and half paraphrased, with fixed 40-word chunks and overlap 10. A hit requires both the
expected source and evidence text in the retrieved passage. Fixtures are checked before scoring.
Per-question rankings and aggregate metrics are saved in evaluation/results.json;
single-method runs use results_tfidf.json or results_semantic.json.

- Hit@1: fraction with expected evidence ranked first.
- Hit@3: fraction with expected evidence in the top three.
- MRR@3: average reciprocal evidence rank; zero if absent from the top three.

This is a small learning benchmark, not proof of production quality. It is separate from
your uploaded documents so results are reproducible. Do not tune on these questions and
then treat their scores as independent evidence. Add held-out questions for your own PDFs.
Unanswerable questions need separate threshold/abstention evaluation; this benchmark does
not measure that or generated answers.

## Answer-quality evaluation

`evaluation/answer_questions.json` is a separate fixed set of supported, ambiguous, and
unanswerable questions. Its small fictional corpus is in `evaluation/answer_documents/`.
Each case specifies the expected answer status and evidence; ambiguous cases specify the
alternative evidence that should be retrieved. Fixture-integrity tests verify the expected
statuses, clarification flags, answer points, and required passages. Run the Stage 7.3
retrieval-only evaluation from the project root:

```powershell
python src/evaluate_answer_retrieval.py
python src/evaluate_answer_retrieval.py --offline
```

It reports all-required-evidence@3 and evidence-item recall@3 for supported and ambiguous
questions, with per-question retrieval details in `evaluation/answer_retrieval_results.json`.
Unanswerable cases are excluded from retrieval recall. The evaluator does not yet run answer
generation or score claim-to-evidence support.
