"""Retrieval evaluation harness.

Builds an index from a fixture, runs every query, and reports recall@k, MRR
and nDCG@k. Deterministic end to end: the same code and fixture must produce
byte-identical numbers, because a drifting eval cannot detect a regression.

Judgments are per *document*; retrieval returns *chunks*. Chunks are mapped
back to their document by url, preserving rank order and de-duplicating — a
document whose chunks occupy the top three positions counts once, at rank 1.
Without that collapse, recall would be measured against chunk counts and would
change whenever chunk size changed, which would make the metric useless for
exactly the comparison it exists to support.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import structlog

from app.evaluation.metrics import (
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.retrieval.chunking import chunk_source
from app.retrieval.embeddings import Embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.store import SqliteVecStore
from app.schemas.source import FetchedSource, content_hash

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EvalDocument:
    doc_id: str
    url: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class EvalQuery:
    query_id: str
    query: str
    judgments: dict[str, int]


@dataclass(slots=True)
class QueryResult:
    """Per-query outcome, kept so a regression can be localised to a query."""

    query_id: str
    query: str
    retrieved_docs: list[str]
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    # Which retriever contributed the top hit: shows where hybrid earns its keep.
    top_hit_retrievers: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvalReport:
    """Aggregate result. Serialised verbatim into evals/results/."""

    dataset: str
    k: int
    embedding_model: str
    chunk_tokens: int
    overlap_tokens: int
    documents: int
    chunks: int
    queries: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    per_query: list[QueryResult] = field(default_factory=list)
    retriever: str = "hybrid"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def summary_line(self) -> str:
        return (
            f"{self.retriever:8} k={self.k:<3} "
            f"recall@{self.k}={self.recall_at_k:.3f}  "
            f"P@{self.k}={self.precision_at_k:.3f}  "
            f"MRR={self.mrr:.3f}  "
            f"nDCG@{self.k}={self.ndcg_at_k:.3f}"
        )


def load_dataset(directory: Path) -> tuple[list[EvalDocument], list[EvalQuery]]:
    """Load a fixture directory containing corpus.jsonl and queries.jsonl."""
    corpus_path = directory / "corpus.jsonl"
    queries_path = directory / "queries.jsonl"
    for path in (corpus_path, queries_path):
        if not path.is_file():
            raise FileNotFoundError(f"Eval fixture is missing {path}")

    documents = [
        EvalDocument(**json.loads(line))
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    queries = [
        EvalQuery(**json.loads(line))
        for line in queries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # A judgment naming a document that is not in the corpus is an annotation
    # error that would silently depress recall forever.
    known = {doc.doc_id for doc in documents}
    for query in queries:
        unknown = set(query.judgments) - known
        if unknown:
            raise ValueError(
                f"Query {query.query_id} judges documents absent from the corpus: {sorted(unknown)}"
            )
    return documents, queries


def _as_source(document: EvalDocument) -> FetchedSource:
    """Wrap a fixture document as a FetchedSource so chunking is identical.

    The eval must exercise the same chunker the pipeline uses; a separate
    code path for evaluation would measure the wrong thing.
    """
    return FetchedSource(
        url=document.url,
        final_url=document.url,
        title=document.title,
        text=document.text,
        content_hash=content_hash(document.text),
        status_code=200,
        byte_length=len(document.text.encode("utf-8")),
    )


def run_retrieval_eval(
    dataset_dir: Path,
    embedder: Embedder,
    *,
    k: int = 10,
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
    embedding_model_name: str = "unknown",
    retriever_name: str = "hybrid",
    dense_only: bool = False,
    lexical_only: bool = False,
) -> EvalReport:
    """Index the fixture, run every query, and report the metrics.

    `dense_only` and `lexical_only` run a single retriever, which is how the
    hybrid claim gets tested rather than asserted: if hybrid does not beat
    both, it is not earning its complexity.
    """
    documents, queries = load_dataset(dataset_dir)
    url_to_doc = {doc.url: doc.doc_id for doc in documents}

    store = SqliteVecStore(":memory:", dimension=embedder.dimension)
    try:
        total_chunks = 0
        for document in documents:
            source = _as_source(document)
            chunks = chunk_source(source, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
            vectors = embedder.embed_documents([c.text for c in chunks])
            store.add(chunks, vectors, content_hash=source.content_hash)
            total_chunks += len(chunks)

        retriever = HybridRetriever(store, embedder)
        results: list[QueryResult] = []

        for query in queries:
            if dense_only:
                hits = store.search_dense(embedder.embed_query(query.query), k * 3)
                chunk_map = store.get_chunks([h.chunk_id for h in hits])
                ranked = [chunk_map[h.chunk_id] for h in hits if h.chunk_id in chunk_map]
                contributors = ["dense"]
            elif lexical_only:
                hits = store.search_lexical(query.query, k * 3)
                chunk_map = store.get_chunks([h.chunk_id for h in hits])
                ranked = [chunk_map[h.chunk_id] for h in hits if h.chunk_id in chunk_map]
                contributors = ["lexical"]
            else:
                scored = retriever.retrieve(query.query, k=k * 3)
                ranked = [s.chunk for s in scored]
                contributors = list(scored[0].retrievers) if scored else []

            # Collapse chunks to documents, first occurrence wins.
            retrieved_docs: list[str] = []
            for chunk in ranked:
                doc_id = url_to_doc.get(str(chunk.source_url))
                if doc_id is not None and doc_id not in retrieved_docs:
                    retrieved_docs.append(doc_id)

            results.append(
                QueryResult(
                    query_id=query.query_id,
                    query=query.query,
                    retrieved_docs=retrieved_docs[:k],
                    recall_at_k=recall_at_k(retrieved_docs, query.judgments, k),
                    precision_at_k=precision_at_k(retrieved_docs, query.judgments, k),
                    reciprocal_rank=reciprocal_rank(retrieved_docs, query.judgments),
                    ndcg_at_k=ndcg_at_k(retrieved_docs, query.judgments, k),
                    top_hit_retrievers=contributors,
                )
            )

        report = EvalReport(
            dataset=dataset_dir.name,
            k=k,
            embedding_model=embedding_model_name,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            documents=len(documents),
            chunks=total_chunks,
            queries=len(queries),
            recall_at_k=round(mean([r.recall_at_k for r in results]), 6),
            precision_at_k=round(mean([r.precision_at_k for r in results]), 6),
            mrr=round(mean([r.reciprocal_rank for r in results]), 6),
            ndcg_at_k=round(mean([r.ndcg_at_k for r in results]), 6),
            per_query=results,
            retriever=retriever_name,
        )
        logger.info(
            "retrieval_eval_complete",
            **{
                "dataset": report.dataset,
                "retriever": report.retriever,
                "k": k,
                "recall": report.recall_at_k,
                "mrr": report.mrr,
                "ndcg": report.ndcg_at_k,
            },
        )
        return report
    finally:
        store.close()
