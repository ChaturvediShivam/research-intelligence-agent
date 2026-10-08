"""M5's exit criterion against the live API.

One realistic research question through every stage, with nothing stood in
for: real web search, real fetching behind the real SSRF guard, real local
embeddings, real hybrid retrieval, real extraction on Haiku, real native
citations on Opus, and the real deterministic verifier.

This is the only test that can establish the exit criterion. The offline
orchestration suite proves the wiring, the ordering and the failure
semantics; it cannot prove that a plan produced by the real planner yields
queries the real search tool can use, whose results the real fetcher can
read, whose text the real extractor will quote verbatim, at offsets the real
verifier accepts. Every one of those seams is a place the pipeline could fail
in a way no fake would reveal.

Billable. Expect roughly $0.10–0.40 and a minute or two of wall clock.

Run with:  uv run pytest -m live tests/integration/test_live_e2e.py -s
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from app.core.config import Settings
from app.llm.client import LLMClient
from app.pipeline.orchestrator import ResearchOrchestrator
from app.retrieval.embeddings import FastEmbedEmbedder
from app.schemas.research import ResearchRequest, RunStatus
from app.schemas.runs import Stage
from app.tools.fetch import SourceFetcher
from app.tools.search import AnthropicSearchProvider


def _key_available() -> bool:
    try:
        Settings(environment="local").require_anthropic_key()
    except Exception:
        return False
    return True


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _key_available(), reason="ANTHROPIC_API_KEY not resolvable"),
]

QUESTION = (
    "How concentrated is the UK pet insurance market, and what is driving claims costs upward?"
)

RESULTS_DIR = Path("evals/results")


async def test_one_complete_live_research_run() -> None:
    """The M5 exit criterion: one full run, measured."""
    settings = Settings(
        environment="local",
        # Bounded so one run cannot become an expensive one.
        max_sources_per_run=5,
        retrieval_top_k=4,
    )

    client = LLMClient(settings)
    provider = AnthropicSearchProvider(settings)
    fetcher = SourceFetcher(settings)
    embedder = FastEmbedEmbedder()

    orchestrator = ResearchOrchestrator(
        client=client,
        provider=provider,
        fetcher=fetcher,
        embedder=embedder,
        settings=settings,
    )

    started = time.perf_counter()
    try:
        result = await orchestrator.run(ResearchRequest(question=QUESTION))
    finally:
        await client.aclose()
        await provider.aclose()
        await fetcher.aclose()
        # Drain pending transport callbacks (F-006).
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    wall_ms = int((time.perf_counter() - started) * 1000)

    summary = result.summary()

    # ---------------------------------------------------------------- report
    print("\n" + "=" * 72)
    print("LIVE END-TO-END RESEARCH RUN")
    print("=" * 72)
    print(f"question:            {QUESTION}")
    print(f"run_id:              {result.run_id}")
    print(f"status:              {result.status.value}")
    if result.error:
        print(f"error:               {result.error}")
    print("-" * 72)
    print(f"plan sub-questions:  {summary['plan_sub_questions']}")
    print(f"queries issued:      {summary['queries_issued']}")
    print(f"sources discovered:  {summary['sources_discovered']}")
    print(f"sources fetched:     {summary['sources_fetched']}")
    print(f"sources failed:      {summary['sources_failed']}")
    print(f"chunks indexed:      {summary['chunks_indexed']}")
    print(f"chunks retrieved:    {summary['chunks_retrieved']}")
    print(f"evidence items:      {summary['evidence_items']}")
    print(f"evidence unlocatable:{summary['evidence_unlocatable']}")
    print(f"evidence verified:   {summary['evidence_verified']}")
    print(f"claims synthesised:  {summary['claims']}")
    print(f"citations verified:  {summary['citations_verified']}")
    print(f"citations rejected:  {summary['citations_rejected']}")
    print(f"claims supported:    {summary['claims_supported']}")
    print(f"claims UNKNOWN:      {summary['claims_unknown']}")
    print("-" * 72)
    print(f"LLM calls:           {summary['total_llm_calls']}")
    usage = result.trace.total_usage
    print(
        f"tokens:              {usage.input_tokens} in / {usage.output_tokens} out"
        f" / {usage.cache_read_input_tokens} cache-read"
    )
    hit_rate = usage.cache_hit_rate
    print(
        f"cache hit rate:      {hit_rate:.1%}"
        if hit_rate is not None
        else "cache hit rate:      n/a"
    )
    print(f"total cost:          ${summary['total_cost_usd']:.6f}")
    print(f"stage latency total: {summary['total_duration_ms']} ms")
    print(f"wall clock:          {wall_ms} ms")
    print("-" * 72)
    print("per-stage:")
    for outcome in result.stages:
        metric = outcome.metric
        print(
            f"  {outcome.stage.value:<12} {outcome.status.value:<8} "
            f"{outcome.duration_ms:>7} ms  ${outcome.cost_usd:.6f}  "
            f"in={outcome.inputs:<3} out={outcome.outputs:<3} "
            f"calls={metric.calls if metric else 0}"
        )
        for warning in outcome.warnings[:3]:
            print(f"      warn: {warning[:90]}")
        for error in outcome.errors[:3]:
            print(f"      err:  {error[:90]}")
    print("-" * 72)
    print("sources:")
    for reference in result.sources.values():
        print(f"  [{reference.credibility.value:<22}] {reference.domain}")
    print("-" * 72)
    if result.claim_validation is not None:
        print("supported claims:")
        for claim in result.claim_validation.supported_claims:
            print(
                f"  conf={claim.confidence.value:<8} corrob={claim.corroboration} "
                f"cites={claim.verified_citations}  [{claim.sub_question_id}]"
            )
            print(f"    {claim.text[:150]}")
            print(f"    basis: {claim.confidence_basis}")
            for citation in claim.citations[:2]:
                print(
                    f"    cite  {citation.source_id} "
                    f"[{citation.start_char}:{citation.end_char}] "
                    f"{citation.cited_text[:70]!r}"
                )
        if result.claim_validation.unknown_claims:
            print("UNKNOWN claims:")
            for claim in result.claim_validation.unknown_claims:
                print(f"  failures={[f.value for f in claim.failures]}  {claim.text[:110]}")
    print("=" * 72)

    # Persist the measured run so the report is reproducible evidence rather
    # than something quoted from a terminal.
    out = RESULTS_DIR / f"live_e2e_{result.run_id}.json"
    await asyncio.to_thread(RESULTS_DIR.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(
        out.write_text,
        json.dumps(
            {
                "summary": summary,
                "wall_clock_ms": wall_ms,
                "stages": [
                    {
                        "stage": o.stage.value,
                        "status": o.status.value,
                        "duration_ms": o.duration_ms,
                        "cost_usd": o.cost_usd,
                        "inputs": o.inputs,
                        "outputs": o.outputs,
                        "calls": o.metric.calls if o.metric else 0,
                        "warnings": o.warnings,
                        "errors": o.errors,
                    }
                    for o in result.stages
                ],
                "sources": [
                    {"domain": r.domain, "credibility": r.credibility.value, "url": str(r.url)}
                    for r in result.sources.values()
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"written: {out}")

    # ------------------------------------------------------------ assertions
    assert result.status is RunStatus.COMPLETED, (
        f"run did not complete: {result.error}. Stage statuses: {summary['stages']}"
    )

    # Every stage reached a usable state.
    for stage in (
        Stage.PLAN,
        Stage.DISCOVER,
        Stage.PROCESS,
        Stage.RETRIEVE,
        Stage.EXTRACT,
        Stage.VALIDATE,
        Stage.SYNTHESISE,
        Stage.REPORT,
    ):
        outcome = result.stage(stage)
        assert outcome is not None, f"{stage.value} did not run"
        assert outcome.status.is_usable, f"{stage.value} was {outcome.status.value}"

    # The pipeline actually did work at each step.
    assert summary["plan_sub_questions"] >= 1
    assert summary["sources_discovered"] >= 1
    assert summary["sources_fetched"] >= 1
    assert summary["chunks_retrieved"] >= 1
    assert summary["evidence_items"] >= 1
    assert summary["claims_supported"] >= 1

    # The guarantee: every supported claim's citation slices the stored text
    # exactly. Verified here independently of the verifier's own verdict.
    assert result.claim_validation is not None
    for claim in result.claim_validation.supported_claims:
        assert claim.citations, claim.claim_id
        assert claim.verified_citations >= 1
        assert claim.failures == []
        for citation in claim.citations:
            stored = result.source_texts[citation.source_id]
            assert stored[citation.start_char : citation.end_char] == citation.cited_text, (
                f"{claim.claim_id}: citation does not slice the stored source"
            )

    # Evidence provenance holds back to the fetched sources.
    assert result.extraction is not None
    for item in result.extraction.items:
        assert item.source_id in result.sources
        quote = item.quote
        stored = result.source_texts[quote.source_id]
        assert stored[quote.start_char : quote.end_char] == quote.text

    # Measurement is real, and within the configured ceiling.
    assert summary["total_llm_calls"] > 0
    assert summary["total_cost_usd"] > 0
    assert summary["total_cost_usd"] <= settings.max_cost_usd_per_run
    assert wall_ms > 0
