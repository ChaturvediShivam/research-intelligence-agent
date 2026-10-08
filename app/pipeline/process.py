"""Stage 3 — PROCESS.

Fetch candidate sources, extract and normalise their text, and chunk them with
offsets preserved. This is the untrusted-content boundary: everything after it
is page text from the open web, and nothing downstream may treat it as
instruction.

One source failing must not fail the run. A research question answered from
five of eight sources is a usable result with a recorded gap; aborting because
one URL 404'd is not. Failures are collected and returned alongside the
successes so stage 8 (ASSESS) can account for them.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import structlog

from app.core.config import Settings
from app.core.errors import AppError
from app.retrieval.chunking import chunk_source, verify_chunks
from app.schemas.runs import Stage, StageMetric
from app.schemas.source import Chunk, FetchedSource, SourceCandidate
from app.tools.fetch import SourceFetcher

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class SourceFailure:
    """A candidate that could not be processed, and why."""

    url: str
    code: str
    message: str


@dataclass(slots=True)
class ProcessResult:
    """Everything stage 3 produced, including what it could not."""

    sources: list[FetchedSource] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    failures: list[SourceFailure] = field(default_factory=list)

    @property
    def chunks_by_source(self) -> dict[str, list[Chunk]]:
        grouped: dict[str, list[Chunk]] = {}
        for chunk in self.chunks:
            grouped.setdefault(str(chunk.source_url), []).append(chunk)
        return grouped


class OffsetIntegrityError(AppError):
    """Raised when a chunk's offsets do not select its own text.

    This should be impossible — chunks are slices of the source — so it is a
    programming error, not a data error, and it fails loudly rather than
    letting citation verification mysteriously degrade later (ADR-002).
    """

    code = "offset_integrity_error"


async def run_process_stage(
    candidates: list[SourceCandidate],
    *,
    fetcher: SourceFetcher,
    settings: Settings,
) -> tuple[ProcessResult, StageMetric]:
    """Fetch and chunk every candidate, up to the configured source ceiling."""
    started = time.perf_counter()
    result = ProcessResult()

    capped = candidates[: settings.max_sources_per_run]
    if len(candidates) > len(capped):
        logger.info(
            "candidates_capped",
            received=len(candidates),
            kept=len(capped),
            ceiling=settings.max_sources_per_run,
        )

    # Fetches are independent, so they run concurrently. `return_exceptions`
    # keeps one failure from cancelling the rest.
    fetched = await asyncio.gather(
        *(fetcher.fetch(str(c.url)) for c in capped), return_exceptions=True
    )

    for candidate, outcome in zip(capped, fetched, strict=True):
        if isinstance(outcome, BaseException):
            code = getattr(outcome, "code", type(outcome).__name__)
            message = getattr(outcome, "message", str(outcome))
            result.failures.append(
                SourceFailure(url=str(candidate.url), code=code, message=message[:300])
            )
            logger.info("source_skipped", url=str(candidate.url), code=code)
            continue

        chunks = chunk_source(
            outcome,
            chunk_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )

        # Assert the invariant in production, not only in tests.
        bad = verify_chunks(outcome, chunks)
        if bad:
            raise OffsetIntegrityError(
                "Chunk offsets do not match source text.",
                details={"url": str(outcome.final_url), "chunk_indexes": bad},
            )

        result.sources.append(outcome)
        result.chunks.extend(chunks)

    metric = StageMetric(
        stage=Stage.PROCESS,
        model=None,  # no model call in this stage
        duration_ms=int((time.perf_counter() - started) * 1000),
        calls=0,
    )
    logger.info(
        "stage_complete",
        stage=Stage.PROCESS.value,
        sources=len(result.sources),
        chunks=len(result.chunks),
        failures=len(result.failures),
        duration_ms=metric.duration_ms,
    )
    return result, metric
