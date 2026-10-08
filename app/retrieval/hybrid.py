"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion.

**Why hybrid rather than dense alone.** Research questions carry proper nouns,
tickers, statute numbers and figures. Dense retrieval is strong on paraphrase
and weak on exact tokens; BM25 is the reverse. A question like "FCA general
insurance value measures 2023" needs the lexical half; "how concentrated is
this market" needs the dense half.

**Why RRF rather than a weighted score sum.** The two retrievers produce
scores on incomparable scales — a cosine-derived similarity in [0, 1] and a
BM25 score that is unbounded and corpus-dependent. Summing them requires
normalising both, and any normalisation is a tuning parameter that has to be
fitted per corpus. RRF uses only the *rank*, so it needs no normalisation and
no fitting:

    score(d) = Σ  1 / (k + rank_i(d))

k=60 is the value from the original Cormack et al. paper and the common
default. It damps the influence of top ranks enough that a single retriever
cannot dominate on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from app.retrieval.embeddings import Embedder
from app.retrieval.store import Hit, ScoredChunk, VectorStore

logger = structlog.get_logger(__name__)

# Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and individual
# rank learning methods" (2009).
RRF_K = 60


@dataclass(frozen=True, slots=True)
class FusionInput:
    """One retriever's ranked output."""

    name: str
    hits: list[Hit]


def reciprocal_rank_fusion(
    inputs: list[FusionInput], *, k: int = RRF_K
) -> list[tuple[int, float, tuple[str, ...]]]:
    """Fuse ranked lists into one ordering.

    Returns `(chunk_id, fused_score, contributing_retrievers)`, best first.
    Ranks are 1-based, which matters: a 0-based rank would make the first
    result's contribution 1/k instead of 1/(k+1) and skew the fusion.
    """
    scores: dict[int, float] = {}
    sources: dict[int, list[str]] = {}

    for retriever in inputs:
        for rank, hit in enumerate(retriever.hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            sources.setdefault(hit.chunk_id, []).append(retriever.name)

    # Sort by fused score, then by chunk_id so ties are deterministic — an
    # unstable order would make eval numbers irreproducible between runs.
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [(chunk_id, score, tuple(sources[chunk_id])) for chunk_id, score in ordered]


class HybridRetriever:
    """Dense + lexical retrieval over one corpus, fused with RRF."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        rrf_k: int = RRF_K,
        candidate_multiplier: int = 3,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._rrf_k = rrf_k
        # Each retriever is asked for more than k so fusion has material to
        # work with: if both return exactly k, fusion can only reorder the
        # intersection and the union is truncated before it is ranked.
        self._candidate_multiplier = candidate_multiplier

    def retrieve(self, query: str, *, k: int = 12) -> list[ScoredChunk]:
        """Retrieve the k best chunks for a query."""
        if k <= 0:
            return []
        candidates = k * self._candidate_multiplier

        dense = self._store.search_dense(self._embedder.embed_query(query), candidates)
        lexical = self._store.search_lexical(query, candidates)

        fused = reciprocal_rank_fusion(
            [FusionInput("dense", dense), FusionInput("lexical", lexical)],
            k=self._rrf_k,
        )[:k]

        chunks = self._store.get_chunks([chunk_id for chunk_id, _, _ in fused])
        results: list[ScoredChunk] = []
        for rank, (chunk_id, score, retrievers) in enumerate(fused, start=1):
            chunk = chunks.get(chunk_id)
            if chunk is None:  # pragma: no cover - would mean index corruption
                logger.warning("retrieved_chunk_missing", chunk_id=chunk_id)
                continue
            results.append(
                ScoredChunk(
                    chunk_id=chunk_id,
                    chunk=chunk,
                    score=score,
                    retrievers=retrievers,
                    rank=rank,
                )
            )

        logger.info(
            "retrieval_complete",
            query=query[:120],
            dense_hits=len(dense),
            lexical_hits=len(lexical),
            fused=len(results),
            k=k,
        )
        return results
