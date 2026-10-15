# Document search practice project

Search TXT, text-based PDF, and DOCX documents using TF-IDF or local embeddings.
Scanned PDFs need OCR. PDF pages are currently combined; chunk IDs are not page numbers.
DOCX body paragraphs and tables are extracted; headers and nested tables are not covered.

## Setup and search

Run from this folder:

```powershell
python -m pip install -r requirements.txt
python main.py "borrow books" --method tfidf
python main.py "How long can I keep a borrowed book?" --method semantic
```

The semantic model downloads on first use. Imports and TF-IDF searches do not load it.
Relative document folders are resolved beside file_loader.py; absolute paths also work.
Use --directory, --chunk-size, --overlap, and --top-k to change settings.
Semantic scores are similarities, not probabilities. Nearest passages may not answer a question.

## Stages 1 and 2

- validation.py: shared input checks.
- file_loader.py: supported file extraction; warns and skips unreadable or empty files.
- chunk_text.py: overlapping word chunks and source metadata.
- tf_idf.py: build_index once, search_index repeatedly; search_tfidf is a convenience wrapper.
- semantic_search.py: lazy model loading and reusable in-memory document vectors.
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
