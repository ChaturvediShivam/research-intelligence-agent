"""Live verification of stage 1 against the real API.

Marked `live`: deselected by default and never run in CI, because these calls
cost money. This is the test that satisfies M1's exit criterion — "a real
ResearchPlan from a real question" — and it is the only thing that can.

Run with:  uv run pytest -m live
"""

from __future__ import annotations

import os

import pytest

from app.core.config import Settings
from app.llm.client import LLMClient
from app.pipeline.plan import run_plan_stage
from app.schemas.research import ResearchRequest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY is not set; live verification cannot run",
    ),
]

QUESTION = (
    "How concentrated is the UK pet insurance market, and which insurers hold the largest shares?"
)


async def test_real_plan_from_a_real_question() -> None:
    """Stage 1 produces a usable plan from the live API.

    Asserts the properties the prompt is responsible for, not just that a
    response arrived: a decomposition exists, ranks are coherent, and the
    planner did not answer the question it was asked to plan.
    """
    settings = Settings(environment="local")
    client = LLMClient(settings)

    plan, metric = await run_plan_stage(
        ResearchRequest(question=QUESTION),
        client=client,
        settings=settings,
    )

    # Structure
    assert 1 <= len(plan.sub_questions) <= 8
    assert plan.restated_question.strip()
    ranks = sorted(sq.rank for sq in plan.sub_questions)
    assert ranks == list(range(1, len(ranks) + 1)), "ranks should be 1..n"

    # Every sub-question carries the fields the pipeline depends on.
    for sq in plan.sub_questions:
        assert sq.expected_source_types, f"{sq.id} named no source type"
        assert sq.answerable_if.strip(), f"{sq.id} has no answerability criterion"
        assert sq.rationale.strip()

    # The planner must not have answered the question. A planner that asserts
    # market shares has made up facts, having seen no sources.
    plan_text = plan.model_dump_json().lower()
    for leak in ("market share of", "holds approximately", "% of the market"):
        assert leak not in plan_text, f"planner appears to have answered: {leak!r}"

    # Measurement is real, not estimated.
    assert metric.usage.input_tokens > 0
    assert metric.usage.output_tokens > 0
    assert metric.cost_usd > 0
    assert metric.duration_ms > 0

    print(
        f"\nLIVE PLAN: {len(plan.sub_questions)} sub-questions · "
        f"{metric.usage.input_tokens} in / {metric.usage.output_tokens} out · "
        f"${metric.cost_usd:.4f} · {metric.duration_ms}ms"
    )


async def test_prompt_caching_actually_caches() -> None:
    """ADR-008 requires this: a broken cache is invisible except on the invoice.

    Two calls with an identical system prefix. The second must report
    cache_read_input_tokens > 0, or a silent invalidator is at work.
    """
    settings = Settings(environment="local")
    client = LLMClient(settings)
    request = ResearchRequest(question=QUESTION)

    _, first = await run_plan_stage(request, client=client, settings=settings)
    _, second = await run_plan_stage(request, client=client, settings=settings)

    wrote = first.usage.cache_creation_input_tokens
    read = second.usage.cache_read_input_tokens
    print(f"\nLIVE CACHE: wrote={wrote} read={read}")

    assert read > 0, (
        "Second identical-prefix call read nothing from cache. "
        "Check for a varying prefix (timestamp, uuid, unsorted keys)."
    )
