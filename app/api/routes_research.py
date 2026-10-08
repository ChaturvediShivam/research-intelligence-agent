"""Research endpoints.

`POST /research` returns a run id immediately and executes the pipeline in the
background; `GET /research/{id}` polls. That shape is deliberate: a research
run takes far longer than a reasonable HTTP request, and a caller should never
hold a connection open waiting for it.

M1 executes stage 1 (PLAN) only. Later milestones extend the background task;
the API contract does not change.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, BackgroundTasks, status
from pydantic import BaseModel

from app.api.deps import LLMClientDep, RunRepositoryDep, SettingsDep
from app.core.config import Settings
from app.core.errors import PipelineStageError
from app.llm.client import LLMClient
from app.pipeline.plan import run_plan_stage
from app.schemas.research import (
    ResearchPlan,
    ResearchRequest,
    ResearchRun,
    RunStatus,
)
from app.schemas.runs import RunTrace
from app.storage.runs import RunRepository

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


class RunDetail(BaseModel):
    """Full run state, including whatever has been produced so far."""

    run_id: str
    status: RunStatus
    question: str
    plan: ResearchPlan | None
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
) -> RunAccepted:
    """Create a run and schedule the pipeline."""
    run = await repo.create(request)
    background.add_task(_execute_pipeline, run.id, request, repo, client, settings)
    return RunAccepted(
        run_id=run.id,
        status=run.status,
        poll_url=f"/research/{run.id}",
    )


@router.get("/{run_id}", response_model=RunDetail, summary="Poll a research run")
async def get_research_run(run_id: str, repo: RunRepositoryDep) -> RunDetail:
    """Return current status, the plan once available, and measured cost."""
    run: ResearchRun = await repo.get(run_id)
    trace = await repo.get_trace(run_id)
    return RunDetail(
        run_id=run.id,
        status=run.status,
        question=run.request.question,
        plan=run.plan,
        error=run.error,
        cost=_cost_summary(trace) if trace else None,
    )


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
) -> None:
    """Background execution. Every exit path leaves the run in a terminal state.

    A background task that dies silently would leave a run stuck in
    `planning` forever, which is indistinguishable from a slow run. The broad
    `except` is therefore deliberate: it exists to guarantee the run is marked
    failed, and it re-logs with the traceback rather than swallowing.
    """
    trace = RunTrace(run_id=run_id)
    try:
        await repo.set_status(run_id, RunStatus.PLANNING)
        plan, metric = await run_plan_stage(request, client=client, settings=settings)
        trace.stages.append(metric)
        await repo.save_plan(run_id, plan)
        await repo.save_trace(run_id, trace)
        # M1 ends after planning. Stages 2-10 extend this in later milestones.
        await repo.set_status(run_id, RunStatus.COMPLETED)
    except PipelineStageError as exc:
        logger.warning("pipeline_stage_failed", run_id=run_id, stage=exc.stage)
        await repo.save_trace(run_id, trace)
        await repo.set_status(run_id, RunStatus.FAILED, error=f"{exc.stage}: {exc.message}")
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.exception("pipeline_failed", run_id=run_id, exc_type=type(exc).__name__)
        await repo.save_trace(run_id, trace)
        await repo.set_status(run_id, RunStatus.FAILED, error="Internal pipeline error.")
