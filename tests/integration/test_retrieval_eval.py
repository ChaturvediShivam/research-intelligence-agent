"""The evaluation harness itself.

Uses a deterministic test embedder so these run in milliseconds. Retrieval
*quality* is measured with the real model by scripts/run_retrieval_eval.py and
recorded in docs/evaluation.md; these tests check that the harness computes
what it claims to and is reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.runner import EvalDocument, load_dataset, run_retrieval_eval
from tests.fixtures.fake_embedder import TermOverlapEmbedder

DATASET = Path("evals/datasets/retrieval_v1")

VOCAB = [
    "premium",
    "pet",
    "solvency",
    "capital",
    "loyalty",
    "pricing",
    "veterinary",
    "motor",
    "claims",
    "compensation",
    "insurer",
    "investigation",
    "insurtech",
    "funding",
    "tax",
    "travel",
]


@pytest.fixture
def embedder() -> TermOverlapEmbedder:
    return TermOverlapEmbedder(VOCAB)


class TestDatasetIntegrity:
    def test_fixture_loads(self) -> None:
        documents, queries = load_dataset(DATASET)
        assert len(documents) == 20
        assert len(queries) == 10

    def test_document_ids_are_unique(self) -> None:
        documents, _ = load_dataset(DATASET)
        ids = [d.doc_id for d in documents]
        assert len(set(ids)) == len(ids)

    def test_query_ids_are_unique(self) -> None:
        _, queries = load_dataset(DATASET)
        ids = [q.query_id for q in queries]
        assert len(set(ids)) == len(ids)

    def test_every_query_has_at_least_one_relevant_document(self) -> None:
        """A query with no relevant document scores 0 and only drags the mean."""
        _, queries = load_dataset(DATASET)
        for query in queries:
            assert any(grade > 0 for grade in query.judgments.values()), query.query_id

    def test_grades_are_within_the_documented_scale(self) -> None:
        _, queries = load_dataset(DATASET)
        for query in queries:
            for doc_id, grade in query.judgments.items():
                assert grade in (0, 1, 2), f"{query.query_id}/{doc_id} = {grade}"

    def test_judgments_only_reference_documents_in_the_corpus(self) -> None:
        """An annotation typo would depress recall permanently and invisibly."""
        documents, queries = load_dataset(DATASET)
        known = {d.doc_id for d in documents}
        for query in queries:
            assert set(query.judgments) <= known, query.query_id

    def test_corpus_contains_hard_negatives(self) -> None:
        """Documents judged by no query — without them every score is 1.000."""
        documents, queries = load_dataset(DATASET)
        judged = {doc for q in queries for doc in q.judgments}
        unjudged = {d.doc_id for d in documents} - judged
        assert len(unjudged) >= 8, "fixture needs distractors to discriminate"

    def test_missing_fixture_raises_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="missing"):
            load_dataset(tmp_path)

    def test_a_judgment_for_an_unknown_document_is_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "corpus.jsonl").write_text(
            json.dumps({"doc_id": "d1", "url": "https://e.com/1", "title": "t", "text": "x"}) + "\n"
        )
        (tmp_path / "queries.jsonl").write_text(
            json.dumps({"query_id": "q1", "query": "q", "judgments": {"nope": 2}}) + "\n"
        )
        with pytest.raises(ValueError, match="absent from the"):
            load_dataset(tmp_path)


class TestHarness:
    def test_produces_a_complete_report(self, embedder: TermOverlapEmbedder) -> None:
        report = run_retrieval_eval(DATASET, embedder, k=5)
        assert report.documents == 20
        assert report.queries == 10
        assert report.chunks >= 20
        assert 0.0 <= report.recall_at_k <= 1.0
        assert 0.0 <= report.ndcg_at_k <= 1.0
        assert 0.0 <= report.mrr <= 1.0
        assert len(report.per_query) == 10

    def test_is_deterministic(self, embedder: TermOverlapEmbedder) -> None:
        """A drifting eval cannot detect a regression."""
        first = run_retrieval_eval(DATASET, embedder, k=5)
        second = run_retrieval_eval(DATASET, embedder, k=5)
        assert first.recall_at_k == second.recall_at_k
        assert first.mrr == second.mrr
        assert first.ndcg_at_k == second.ndcg_at_k
        assert [q.retrieved_docs for q in first.per_query] == [
            q.retrieved_docs for q in second.per_query
        ]

    def test_report_serialises_to_json(self, embedder: TermOverlapEmbedder) -> None:
        payload = json.loads(run_retrieval_eval(DATASET, embedder, k=3).to_json())
        assert payload["k"] == 3
        assert "per_query" in payload
        assert len(payload["per_query"]) == 10

    def test_dense_only_and_lexical_only_modes_run(self, embedder: TermOverlapEmbedder) -> None:
        dense = run_retrieval_eval(DATASET, embedder, k=5, dense_only=True, retriever_name="dense")
        lexical = run_retrieval_eval(
            DATASET, embedder, k=5, lexical_only=True, retriever_name="lexical"
        )
        assert dense.retriever == "dense"
        assert lexical.retriever == "lexical"
        assert all(q.top_hit_retrievers == ["dense"] for q in dense.per_query)

    def test_chunks_are_collapsed_to_documents(self, embedder: TermOverlapEmbedder) -> None:
        """Recall is measured per document, so a document must count once.

        Small chunks force multi-chunk documents; without the collapse, one
        document occupying the top three slots would be counted three times.
        """
        report = run_retrieval_eval(DATASET, embedder, k=5, chunk_tokens=30)
        assert report.chunks > report.documents, "chunking should have split documents"
        for result in report.per_query:
            assert len(result.retrieved_docs) == len(set(result.retrieved_docs))

    def test_retrieved_docs_respects_k(self, embedder: TermOverlapEmbedder) -> None:
        report = run_retrieval_eval(DATASET, embedder, k=3)
        assert all(len(q.retrieved_docs) <= 3 for q in report.per_query)

    def test_metrics_are_informative_at_k_below_corpus_size(
        self, embedder: TermOverlapEmbedder
    ) -> None:
        """At k >= |corpus| recall is trivially 1.0 and measures nothing."""
        documents, _ = load_dataset(DATASET)
        degenerate = run_retrieval_eval(DATASET, embedder, k=len(documents))
        assert degenerate.recall_at_k == pytest.approx(1.0)
        informative = run_retrieval_eval(DATASET, embedder, k=1)
        assert informative.recall_at_k < 1.0


class TestEvalDocument:
    def test_is_wrapped_as_a_fetched_source_for_identical_chunking(self) -> None:
        """The eval must use the same chunker as the pipeline, not a copy."""
        from app.evaluation.runner import _as_source

        document = EvalDocument(
            doc_id="d", url="https://example.com/d", title="T", text="Body text here."
        )
        source = _as_source(document)
        assert source.text == document.text
        assert source.verify_hash()
