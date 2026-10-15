import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from chunk_text import chunk_text, chunk_documents
from file_loader import load_documents
from tf_idf import build_index, search_index, search_tfidf
from semantic_search import build_semantic_index, search_semantic
from main import run_search
from evaluate import passage_matches, measure


class FakeModel:
    def encode(self, texts, **kwargs):
        def vector(text):
            return [1., 0.] if "book" in text else [0., 1.]
        return np.array([vector(t) for t in texts] if isinstance(texts, list) else vector(texts))


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.chunks = chunk_documents([
            {"source": "books.txt", "content": "borrow books"},
            {"source": "code.txt", "content": "Python code"}], 4, 1)

    def test_chunk_boundaries(self):
        self.assertEqual(chunk_text("one two three four five six seven", 4, 1),
                         ["one two three four", "four five six seven"])
        self.assertEqual(chunk_text("one two three four", 4, 0), ["one two three four"])
        self.assertEqual(chunk_text(""), [])
        for size, overlap in [(0, 0), (4, 4), (4, -1), (2.5, 1)]:
            with self.assertRaises(ValueError):
                chunk_documents([], size, overlap)

    def test_loader(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder)
            (path / "sample.TXT").write_text("books", encoding="utf-8")
            (path / "empty.txt").write_text(" ", encoding="utf-8")
            (path / "broken.txt").write_bytes(b"\xff")
            (path / "ignore.bin").write_bytes(b"hello")
            (path / "directory.txt").mkdir()
            with self.assertLogs("file_loader", level="WARNING"):
                docs = load_documents(path)
            self.assertEqual([d["source"] for d in docs], ["sample.TXT"])
            with self.assertRaises(NotADirectoryError):
                load_documents(path / "absent")

    def test_tfidf_repeated_queries(self):
        vectorizer, vectors = build_index(self.chunks)
        for query, source in [("books?", "books.txt"), ("Python", "code.txt")]:
            self.assertEqual(search_index(self.chunks, vectorizer, vectors, query)[0]["source"], source)
        self.assertEqual(search_index(self.chunks, vectorizer, vectors, "zzzzzz"), [])
        self.assertEqual(search_tfidf(self.chunks, "  "), [])
        for content in ["", "!!!", "a b c"]:
            self.assertEqual(search_tfidf([{"source": "a", "content": content}], "books"), [])
        with self.assertRaises(ValueError):
            search_index(self.chunks, vectorizer, vectors[:1], "books")

    def test_semantic_validation_without_download(self):
        model = FakeModel()
        vectors = build_semantic_index(self.chunks, model)
        self.assertEqual(search_semantic(self.chunks, vectors, "books", 1, model)[0]["source"], "books.txt")
        self.assertEqual(search_semantic(self.chunks, vectors, " ", model=model), [])
        self.assertEqual(build_semantic_index([]).shape, (0, 0))
        for k in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                search_semantic(self.chunks, vectors, "books", k, model)
        for bad_vectors in [vectors[:1], np.zeros((2, 3))]:
            with self.assertRaises(ValueError):
                search_semantic(self.chunks, bad_vectors, "books", model=model)

    def test_bad_records(self):
        with self.assertRaises(ValueError):
            build_index([{"source": "a"}])
        with self.assertRaises(ValueError):
            build_semantic_index([{"source": "a", "content": None}], FakeModel())

    def test_main_reaches_search(self):
        with patch("main.load_documents", return_value=self.chunks), patch("semantic_search.get_model", return_value=FakeModel()), contextlib.redirect_stdout(io.StringIO()) as output:
            results = run_search(query="books", top_k=1)
        self.assertEqual(results[0]["source"], "books.txt")
        self.assertIn("books.txt", output.getvalue())

    def test_evaluation_requires_evidence(self):
        question = {"query": "books", "source": "books.txt", "expected_text": "borrow books"}
        self.assertFalse(passage_matches({"source": "books.txt", "content": "wrong passage"}, question))
        report = measure([question], lambda q: [{**self.chunks[1], "score": 1}, {**self.chunks[0], "score": .5}])
        self.assertEqual(report["hit_at_1"], 0)
        self.assertEqual(report["hit_at_3"], 1)
        self.assertEqual(report["mrr_at_3"], .5)


if __name__ == "__main__":
    unittest.main()
