"""Stage 4 — RETRIEVE.

Indexes the chunks produced by stage 3 and retrieves the best candidates per
sub-question. The index is created per run and discarded with it (ADR-004).

No model call: embeddings are local (ADR-003), so this stage costs nothing and
reports zero cost rather than an estimate.
"""

from __future__ import annotations

import time

import structlog

from app.core.config import Settings
from app.core.errors import PipelineStageError
from app.retrieval.embeddings import Embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.store import ScoredChunk, SqliteVecStore
from app.schemas.research import ResearchPlan
from app.schemas.runs import Stage, StageMetric
from app.schemas.source import Chunk

logger = structlog.get_logger(__name__)


def build_index(
    chunks: list[Chunk], embedder: Embedder, *, path: str = ":memory:"
) -> SqliteVecStore:
    """Embed and index chunks, returning the populated store.

    The caller owns the store and must close it; the pipeline does so in a
    finally block so an index is not leaked when a later stage fails.
    """
    store = SqliteVecStore(path, dimension=embedder.dimension)
    if chunks:
        vectors = embedder.embed_documents([c.text for c in chunks])
        store.add(chunks, vectors)
    return store


async def run_retrieve_stage(
    plan: ResearchPlan,
    chunks: list[Chunk],
    *,
    embedder: Embedder,
    settings: Settings,
) -> tuple[dict[str, list[ScoredChunk]], StageMetric]:
    """Retrieve chunks per sub-question.

    Returns a mapping of sub-question id to its ranked chunks. Retrieval is
    per sub-question rather than per overall question because the plan's
    decomposition is the unit the later stages reason about — evidence is
    attributed to a sub-question, and a single blended retrieval would lose
    that attribution.
    """
    started = time.perf_counter()
    store: SqliteVecStore | None = None
    try:
        store = build_index(chunks, embedder)
        retriever = HybridRetriever(store, embedder)
        per_sub_question: dict[str, list[ScoredChunk]] = {}
        for sub_question in plan.ordered():
            per_sub_question[sub_question.id] = retriever.retrieve(
                sub_question.question, k=settings.retrieval_top_k
            )
    except Exception as exc:
        raise PipelineStageError(
            Stage.RETRIEVE.value, f"Retrieval failed: {exc}", cause=exc
        ) from exc
    finally:
        if store is not None:
            store.close()

    metric = StageMetric(
        stage=Stage.RETRIEVE,
        model=None,  # local embeddings: no API call, no cost
        duration_ms=int((time.perf_counter() - started) * 1000),
        calls=0,
    )
    logger.info(
        "stage_complete",
        stage=Stage.RETRIEVE.value,
        chunks_indexed=len(chunks),
        sub_questions=len(per_sub_question),
        duration_ms=metric.duration_ms,
    )
    return per_sub_question, metric
