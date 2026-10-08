"""Stage 2 — DISCOVER. The one model-driven step.

Everything else in the pipeline is a fixed sequence. Discovery is not: how
many searches a question needs, which phrasings to try, and when enough
sources have been found all depend on what the web returns. That is the
definition of a step that cannot be pre-scripted, and the only one in this
system (ADR-009).

The loop is the SDK's `tool_runner`, not a hand-written
`while stop_reason == "tool_use"`. The model chooses queries; the tool it
calls goes through the existing `SourceProvider` (ADR-006), so discovery
cannot bypass the domain policy that abstraction enforces.

Bounded on three axes, because an unbounded agentic loop is a spend risk:
`max_sources_per_run` caps results, `max_iterations` caps turns, and the
provider itself caps `max_uses` on the server-side tool.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog
from anthropic import beta_async_tool

from app.core.config import Settings
from app.core.errors import PipelineStageError, UpstreamError
from app.llm.client import LLMClient
from app.schemas.research import ResearchPlan
from app.schemas.runs import Stage, StageMetric, TokenUsage
from app.schemas.source import SourceCandidate
from app.tools.search import SourceProvider

logger = structlog.get_logger(__name__)

PROMPT_VERSION = "discover.v1"

DISCOVER_SYSTEM = (
    "You are the source-discovery step of a research pipeline. You are given "
    "a research plan whose sub-questions are ranked by importance.\n\n"
    "Call the search_sources tool to find sources, working through the "
    "sub-questions in rank order. Choose query wording yourself: use the "
    "terminology the kind of document you want would actually use, not the "
    "wording of the sub-question. Prefer searches likely to surface "
    "regulatory filings, official statistics and primary company documents.\n\n"
    "Issue one search per sub-question first. Only search again for the same "
    "sub-question if the first attempt returned nothing useful, and then "
    "change the wording rather than repeating it.\n\n"
    "Stop as soon as every sub-question has candidate sources, or when "
    "further searching is clearly not helping. Do not answer the research "
    "question, do not summarise the sources, and do not comment on them — "
    "finding them is the entire task. When you are finished, reply with the "
    "single word DONE."
)


@dataclass(slots=True)
class DiscoveryResult:
    """Candidates found, and a record of how they were found.

    `queries_issued` exists for debugging and evaluation: a run that returned
    poor sources is usually a run that issued poor queries, and without this
    there is no way to tell that from a retrieval problem.
    """

    candidates: list[SourceCandidate] = field(default_factory=list)
    queries_issued: list[str] = field(default_factory=list)
    search_failures: list[str] = field(default_factory=list)

    @property
    def domains(self) -> set[str]:
        return {candidate.domain for candidate in self.candidates}


def _usage_of(message: Any) -> TokenUsage:
    usage = getattr(message, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


def _plan_brief(plan: ResearchPlan) -> str:
    """The plan, rendered for the discovery model."""
    lines = [
        f"Research question (restated): {plan.restated_question}",
        "",
        "Sub-questions in rank order:",
    ]
    for sub_question in plan.ordered():
        source_types = ", ".join(t.value for t in sub_question.expected_source_types)
        lines.append(
            f"  {sub_question.rank}. [{sub_question.id}] {sub_question.question}\n"
            f"     expected source types: {source_types}"
        )
    return "\n".join(lines)


async def run_discover_stage(
    plan: ResearchPlan,
    *,
    provider: SourceProvider,
    client: LLMClient,
    settings: Settings,
    max_iterations: int = 8,
) -> tuple[DiscoveryResult, StageMetric]:
    """Discover candidate sources for a plan.

    The tool closes over `result`, so candidates accumulate as the model
    searches. A search that fails is recorded and reported back to the model
    as text rather than raised: one bad query should change the next query,
    not end the stage.
    """
    started = time.perf_counter()
    result = DiscoveryResult()
    seen_urls: set[str] = set()

    @beta_async_tool
    async def search_sources(query: str, sub_question_id: str) -> str:
        """Search the web for sources that would answer a sub-question.

        Args:
            query: The search query. Use the terminology the kind of document
                you want would use, not the wording of the sub-question.
            sub_question_id: The id of the sub-question this search is for,
                e.g. SQ1.
        """
        remaining = settings.max_sources_per_run - len(result.candidates)
        if remaining <= 0:
            return (
                f"Source budget reached ({settings.max_sources_per_run}). "
                "Stop searching and reply DONE."
            )

        result.queries_issued.append(query)
        try:
            found = await provider.search(
                query,
                max_results=min(remaining, 8),
                sub_question_id=sub_question_id,
            )
        except UpstreamError as exc:
            # Reported to the model, not raised: it can try different wording.
            result.search_failures.append(f"{query}: {exc.message}")
            logger.warning("discovery_search_failed", query=query[:120])
            return f"Search failed: {exc.message}. Try different wording."

        added: list[SourceCandidate] = []
        for candidate in found:
            url = str(candidate.url)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            result.candidates.append(candidate)
            added.append(candidate)

        if not added:
            return "No new sources found for that query. Try different wording."

        listing = "\n".join(f"- {c.domain}: {c.title[:100]}" for c in added)
        return (
            f"Found {len(added)} new source(s) for {sub_question_id} "
            f"({len(result.candidates)}/{settings.max_sources_per_run} total):\n"
            f"{listing}"
        )

    usage = TokenUsage()
    calls = 0

    try:
        runner = client.client.beta.messages.tool_runner(
            model=settings.planning_model,
            max_tokens=4096,
            system=DISCOVER_SYSTEM,
            messages=[{"role": "user", "content": _plan_brief(plan)}],
            tools=[search_sources],
            max_iterations=max_iterations,
            # Forced tool choice is a 400 on this model; the system prompt
            # names the tool instead (ADR-006 amendment).
            tool_choice={"type": "auto"},
        )
        async for message in runner:
            calls += 1
            usage = usage + _usage_of(message)
    except Exception as exc:
        raise PipelineStageError(
            Stage.DISCOVER.value, f"Source discovery failed: {exc}", cause=exc
        ) from exc

    from app.llm.pricing import cost_usd

    cost = cost_usd(settings.planning_model, usage)
    metric = StageMetric(
        stage=Stage.DISCOVER,
        model=settings.planning_model,
        duration_ms=int((time.perf_counter() - started) * 1000),
        usage=usage,
        cost_usd=cost,
        calls=calls,
    )
    logger.info(
        "stage_complete",
        stage=Stage.DISCOVER.value,
        prompt_version=PROMPT_VERSION,
        queries=len(result.queries_issued),
        candidates=len(result.candidates),
        domains=len(result.domains),
        search_failures=len(result.search_failures),
        iterations=calls,
        cost_usd=cost,
        duration_ms=metric.duration_ms,
    )
    return result, metric
