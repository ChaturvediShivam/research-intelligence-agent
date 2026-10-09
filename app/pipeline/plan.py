"""Stage 1 — PLAN.

Turns a research question into a ranked, testable decomposition. The output is
schema-constrained, so this stage cannot return prose where a plan is expected.

It is also the stage with the highest leverage on the whole run: the source
budget is spent in the rank order this stage produces, so a bad decomposition
wastes the entire run. Hence Opus at `high` effort (ADR-007).
"""

from __future__ import annotations

import time

import structlog

from app.core.config import Settings
from app.core.errors import PipelineStageError
from app.llm.client import LLMClient
from app.llm.context import build_plan_user_content, load_prompt
from app.schemas.research import ResearchPlan, ResearchRequest
from app.schemas.runs import Stage, StageMetric

logger = structlog.get_logger(__name__)

PROMPT_VERSION = "plan.v2"


async def run_plan_stage(
    request: ResearchRequest,
    *,
    client: LLMClient,
    settings: Settings,
) -> tuple[ResearchPlan, StageMetric]:
    """Produce a `ResearchPlan`, plus the measured cost of producing it.

    Returns the metric alongside the value rather than writing it to a global
    collector, so the stage stays a pure function of its inputs and is
    testable without a trace object.
    """
    started = time.perf_counter()
    try:
        result = await client.structured(
            model=settings.planning_model,
            output_model=ResearchPlan,
            system=load_prompt(PROMPT_VERSION),
            user_content=build_plan_user_content(
                request.question,
                request.context,
                max_sources=settings.max_sources_per_run,
            ),
            effort="high",
        )
    except PipelineStageError:
        raise
    except Exception as exc:
        # Wrapped so a failure names the stage it came from. The original is
        # chained, so nothing about the cause is lost.
        raise PipelineStageError(
            Stage.PLAN.value,
            f"Planning failed: {exc}",
            cause=exc,
        ) from exc

    plan = result.value
    # The budget rule is a prompt instruction, so it is a request rather than
    # a guarantee. Enforcing it in the schema would fail an otherwise good run
    # over a quality preference, so an over-budget plan is logged instead and
    # the extra sub-questions are reported as gaps by stage 8 as before.
    if len(plan.sub_questions) > settings.max_sources_per_run:
        logger.warning(
            "plan_exceeds_source_budget",
            sub_questions=len(plan.sub_questions),
            max_sources=settings.max_sources_per_run,
            prompt_version=PROMPT_VERSION,
        )

    metric = StageMetric(
        stage=Stage.PLAN,
        model=result.model,
        duration_ms=int((time.perf_counter() - started) * 1000),
        usage=result.usage,
        cost_usd=result.cost_usd,
        calls=1,
    )
    logger.info(
        "stage_complete",
        stage=Stage.PLAN.value,
        prompt_version=PROMPT_VERSION,
        sub_questions=len(plan.sub_questions),
        max_sources=settings.max_sources_per_run,
        duration_ms=metric.duration_ms,
        cost_usd=metric.cost_usd,
    )
    return plan, metric
