"""Stage 6 — SYNTHESISE. Cited prose from verified evidence.

Two properties make this stage safe rather than merely useful:

1. **It can only see sources whose evidence already verified.** The
   orchestrator passes the sources backing verified evidence, so a source
   whose extracted quotes failed verification is not available to be cited.

2. **It cannot mark anything verified.** Every `Claim` it emits is built with
   the schema defaults — `status=UNKNOWN`, `confidence=UNKNOWN`,
   `verified_citations=0`. Only `CitationVerifier` ever changes those, and it
   does so by re-slicing the stored source. There is no code path by which
   synthesis promotes its own citation, which is a property of the types
   rather than a convention to remember.

Uses native document citations (ADR-001), so each claim arrives with character
offsets that stage 7 re-checks independently. The call is deliberately not
schema-constrained, because `output_config.format` and citations are mutually
exclusive — the two-pass design ADR-001 documents.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from app.core.config import Settings
from app.core.errors import PipelineStageError
from app.llm.citations import (
    parse_cited_response,
    request_cited_answer,
    source_ids_in_order,
)
from app.llm.client import LLMClient
from app.llm.pricing import cost_usd
from app.schemas.evidence import Claim
from app.schemas.research import ResearchPlan
from app.schemas.runs import Stage, StageMetric, TokenUsage
from app.schemas.source import FetchedSource

logger = structlog.get_logger(__name__)

PROMPT_VERSION = "synthesize.v1"


@dataclass(slots=True)
class SynthesisResult:
    """Claims produced, and which sub-questions produced none."""

    claims: list[Claim] = field(default_factory=list)
    # Sub-questions that had verified evidence but yielded no cited claim.
    # Kept so stage 8 can distinguish "no evidence" from "evidence that did
    # not support a statement" — different findings for a reader.
    unanswered: list[str] = field(default_factory=list)

    @property
    def cited_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.citations]


def _usage_of(response: object) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:  # pragma: no cover - defensive
        return TokenUsage()
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


async def synthesise_sub_question(
    sub_question_id: str,
    question: str,
    sources: list[FetchedSource],
    *,
    client: LLMClient,
    settings: Settings,
) -> tuple[list[Claim], TokenUsage, float]:
    """Produce cited claims for one sub-question."""
    response = await request_cited_answer(
        client.client,
        model=settings.synthesis_model,
        sources=sources,
        question=(
            f"{question}\n\n"
            "Answer only from the attached documents. Write plain declarative "
            "sentences, each citing the passage it rests on. If the documents "
            "do not answer the question, say exactly that and cite nothing."
        ),
    )

    blocks = parse_cited_response(list(response.content), source_ids_in_order(sources))

    claims: list[Claim] = []
    for position, block in enumerate(blocks):
        text = block.text.strip()
        # Schema requires 4 characters; connective fragments are not claims.
        if len(text) < 4:
            continue
        claims.append(
            Claim(
                claim_id=f"{sub_question_id}_C{position}",
                sub_question_id=sub_question_id,
                text=text,
                citations=block.citations,
                # status, confidence and verified_citations are left at their
                # defaults. Only the verifier sets them.
            )
        )

    usage = _usage_of(response)
    return claims, usage, cost_usd(settings.synthesis_model, usage)


async def run_synthesise_stage(
    plan: ResearchPlan,
    sources_by_sub_question: dict[str, list[FetchedSource]],
    *,
    client: LLMClient,
    settings: Settings,
) -> tuple[SynthesisResult, StageMetric]:
    """Synthesise cited claims, one call per sub-question that has sources.

    Per sub-question rather than one blended call, so a claim carries the
    sub-question it answers. A single synthesis over everything would lose
    that attribution, and stage 8 needs it to report which sub-questions went
    unanswered.
    """
    started = time.perf_counter()
    result = SynthesisResult()
    usage = TokenUsage()
    cost = 0.0
    calls = 0

    questions = {sq.id: sq.question for sq in plan.ordered()}

    for sub_question_id in [sq.id for sq in plan.ordered()]:
        sources = sources_by_sub_question.get(sub_question_id, [])
        if not sources:
            # No verified evidence reached this sub-question. Not an error:
            # stage 8 reports it as an information gap.
            result.unanswered.append(sub_question_id)
            continue

        try:
            claims, call_usage, call_cost = await synthesise_sub_question(
                sub_question_id,
                questions.get(sub_question_id, ""),
                sources,
                client=client,
                settings=settings,
            )
        except Exception as exc:
            raise PipelineStageError(
                Stage.SYNTHESISE.value,
                f"Synthesis failed for {sub_question_id}: {exc}",
                cause=exc,
            ) from exc

        calls += 1
        usage = usage + call_usage
        cost += call_cost

        if not claims:
            result.unanswered.append(sub_question_id)
        result.claims.extend(claims)

    metric = StageMetric(
        stage=Stage.SYNTHESISE,
        model=settings.synthesis_model if calls else None,
        duration_ms=int((time.perf_counter() - started) * 1000),
        usage=usage,
        cost_usd=round(cost, 6),
        calls=calls,
    )
    logger.info(
        "stage_complete",
        stage=Stage.SYNTHESISE.value,
        prompt_version=PROMPT_VERSION,
        calls=calls,
        claims=len(result.claims),
        cited_claims=len(result.cited_claims),
        unanswered=len(result.unanswered),
        cost_usd=metric.cost_usd,
        duration_ms=metric.duration_ms,
    )
    return result, metric
