"""Research endpoints.

`POST /research` returns a run id immediately and executes the pipeline in the
background; `GET /research/{id}` polls. That shape is deliberate: a research
run takes far longer than a reasonable HTTP request, and a caller should never
hold a connection open waiting for it.

The background task runs the real `ResearchOrchestrator` — every stage from
planning through citation validation to the assembled report. Until M10 this
route called `run_plan_stage` alone and then marked the run `completed`, so a
plan-only run was indistinguishable over HTTP from a finished one (F-017).
"""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, BackgroundTasks, status
from pydantic import BaseModel

from app.api.deps import ComponentsDep, LLMClientDep, RunRepositoryDep, SettingsDep
from app.core.config import Settings
from app.core.errors import PipelineStageError
from app.llm.client import LLMClient
from app.pipeline.orchestrator import ResearchOrchestrator
from app.schemas.evidence import Citation
from app.schemas.report import ResearchReport, SourceCoverage
from app.schemas.research import (
    ResearchPlan,
    ResearchRequest,
    ResearchRun,
    RunStatus,
)
from app.schemas.runs import RunTrace, StageOutcome
from app.storage.runs import RunRepository
from app.tools.registry import ToolContext

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


class RunAccepted(BaseModel):
    """202 body: the run exists and is being worked on."""

    run_id: str
    status: RunStatus
    poll_url: str


class CostSummary(BaseModel):
    """Measured cost of a run so far. Derived from real usage, never estimated."""

    total_usd: float
    by_stage: dict[str, float]
    total_duration_ms: int
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_hit_rate: float | None


class StageTiming(BaseModel):
    """One stage's measured cost and latency, from the persisted trace."""

    stage: str
    duration_ms: int
    cost_usd: float
    calls: int
    model: str | None = None


class RunDetail(BaseModel):
    """Full run state, including whatever has been produced so far.

    `report` is the deliverable and already nests every verified citation
    under its claim. `sources` and `citations` are flattened views of that
    same report — convenience for a caller that wants provenance without
    walking the tree, never a second source of truth.
    """

    run_id: str
    status: RunStatus
    question: str
    plan: ResearchPlan | None
    report: ResearchReport | None
    sources: SourceCoverage | None
    citations: list[Citation]
    stages: list[StageTiming]
    error: str | None
    cost: CostSummary | None


@router.post(
    "",
    response_model=RunAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a research question",
)
async def create_research_run(
    request: ResearchRequest,
    background: BackgroundTasks,
    repo: RunRepositoryDep,
    client: LLMClientDep,
    settings: SettingsDep,
    components: ComponentsDep,
) -> RunAccepted:
    """Create a run and schedule the full pipeline."""
    run = await repo.create(request)
    background.add_task(_execute_pipeline, run.id, request, repo, client, settings, components)
    return RunAccepted(
        run_id=run.id,
        status=run.status,
        poll_url=f"/research/{run.id}",
    )


@router.get("/{run_id}", response_model=RunDetail, summary="Poll a research run")
async def get_research_run(run_id: str, repo: RunRepositoryDep) -> RunDetail:
    """Return current status and everything persisted for this run so far."""
    run: ResearchRun = await repo.get(run_id)
    report = await repo.get_report(run_id)
    trace = await repo.get_trace(run_id)
    return RunDetail(
        run_id=run.id,
        status=run.status,
        question=run.request.question,
        plan=run.plan,
        report=report,
        sources=report.source_coverage if report else None,
        citations=_citations(report),
        stages=_stage_timings(trace),
        error=run.error,
        cost=_cost_summary(trace) if trace else None,
    )


def _citations(report: ResearchReport | None) -> list[Citation]:
    """Every verified citation in the report, flattened.

    Only from `supporting_claims`: a claim that failed verification keeps its
    citations in the report for audit, but surfacing them in a flat list
    labelled `citations` would present rejected provenance as accepted.
    """
    if report is None:
        return []
    return [
        citation
        for assessment in report.sub_questions
        for claim in assessment.supporting_claims
        for citation in claim.citations
    ]


def _stage_timings(trace: RunTrace | None) -> list[StageTiming]:
    if trace is None:
        return []
    return [
        StageTiming(
            stage=metric.stage.value,
            duration_ms=metric.duration_ms,
            cost_usd=metric.cost_usd,
            calls=metric.calls,
            model=metric.model,
        )
        for metric in trace.stages
    ]


def _cost_summary(trace: RunTrace) -> CostSummary:
    usage = trace.total_usage
    return CostSummary(
        total_usd=trace.total_cost_usd,
        by_stage=trace.cost_by_stage(),
        total_duration_ms=trace.total_duration_ms,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cache_hit_rate=usage.cache_hit_rate,
    )


async def _execute_pipeline(
    run_id: str,
    request: ResearchRequest,
    repo: RunRepository,
    client: LLMClient,
    settings: Settings,
    components: ToolContext,
) -> None:
    """Background execution of the full pipeline.

    Every exit path leaves the run in a terminal state. A background task that
    died silently would leave a run stuck mid-pipeline forever, which is
    indistinguishable from a slow run — so the broad `except` is deliberate:
    it exists to guarantee the run is marked failed, and it re-logs with the
    traceback rather than swallowing.
    """
    # The caller's per-request source cap, applied through the same mechanism
    # the MCP tool layer uses (app/tools/registry.py). The orchestrator reads
    # it from settings; a copy keeps one request from mutating app-wide state.
    run_settings = settings.model_copy(update={"max_sources_per_run": request.max_sources})

    # Every progress write scheduled by this run. Awaited before the terminal
    # status is written, so a late one cannot land on top of it.
    #
    # Completed tasks are deliberately kept rather than discarded in a done
    # callback: discarding them emptied this list before the drain could
    # retrieve their exceptions, which sent a failed write to asyncio's
    # default handler instead of the log. Bounded by the stage count.
    progress: list[asyncio.Task[None]] = []

    def observe(outcome: StageOutcome, run_status: RunStatus) -> None:
        """Record per-stage progress, to the log and to the run row.

        Persisting here is what makes `GET /research/{id}` agree with the
        logs. Previously nothing was written between the initial `planning`
        and the end of the run, so a caller polling a four-minute run saw
        `planning` with a null plan the entire time while the logs showed
        discovery and processing completing — and a run whose process was
        killed kept `planning` forever (F-019).

        The observer is synchronous because the orchestrator must not be
        slowed or failed by it, so the write is scheduled rather than
        awaited. `run_status` is the status of the stage that just finished,
        which is exactly what a diagnosis needs: the last stage a killed run
        got through.
        """
        logger.info(
            "run_stage",
            run_id=run_id,
            stage=outcome.stage.value,
            stage_status=outcome.status.value,
            run_status=run_status.value,
            duration_ms=outcome.metric.duration_ms if outcome.metric else 0,
            cost_usd=outcome.metric.cost_usd if outcome.metric else 0.0,
            errors=outcome.errors or None,
        )
        # The terminal status is written once, below, from the orchestrator's
        # verdict — never from a progress event.
        if run_status.is_terminal:
            return
        progress.append(asyncio.create_task(repo.set_status(run_id, run_status)))

    async def drain_progress() -> None:
        """Let scheduled progress writes finish before the final status.

        A failed progress write is logged and otherwise ignored: losing a
        status update is not a research failure, and raising here would
        replace the orchestrator's verdict with a persistence error.
        """
        if not progress:
            return
        for outcome in await asyncio.gather(*progress, return_exceptions=True):
            if isinstance(outcome, BaseException):
                logger.warning(
                    "run_progress_write_failed",
                    run_id=run_id,
                    exc_type=type(outcome).__name__,
                )

    orchestrator = ResearchOrchestrator(
        client=client,
        provider=components.provider,
        fetcher=components.fetcher,
        embedder=components.embedder,
        settings=run_settings,
        on_stage=observe,
    )

    try:
        await repo.set_status(run_id, RunStatus.PLANNING)
        # The run id created above is adopted, never regenerated.
        result = await orchestrator.run(request, run_id=run_id)

        if result.plan is not None:
            await repo.save_plan(run_id, result.plan)
        if result.report is not None:
            await repo.save_report(run_id, result.report)
        await repo.save_trace(run_id, result.trace)
        await drain_progress()
        # The orchestrator's own verdict. Never hard-coded: a run that failed
        # at discovery must not report the same status as one that finished.
        await repo.set_status(run_id, result.status, error=result.error)
    except PipelineStageError as exc:
        logger.warning("pipeline_stage_failed", run_id=run_id, stage=exc.stage)
        await drain_progress()
        await repo.set_status(run_id, RunStatus.FAILED, error=f"{exc.stage}: {exc.message}")
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.exception("pipeline_failed", run_id=run_id, exc_type=type(exc).__name__)
        await drain_progress()
        await repo.set_status(run_id, RunStatus.FAILED, error="Internal pipeline error.")
