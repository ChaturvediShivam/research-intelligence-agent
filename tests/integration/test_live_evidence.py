"""Live verification of M4's two real integration risks.

Marked `live`: billable. These exist because mocked tests cannot establish
either of the following, and both are assumptions the whole M4 design rests on.

**1. Does the extractor actually quote verbatim?** The offline tests prove that
an unlocatable quote is discarded. They cannot prove the rate at which a real
model produces one. If Haiku paraphrases under this prompt, evidence is
silently lost and the pipeline quietly returns less than it found.

**2. Do Anthropic's native citation offsets index into the text I stored?**
This is the load-bearing assumption of ADR-001. Their `start_char_index` is an
offset into the document *they* received; my verifier re-slices the text *I*
hold. If those two strings are not byte-identical the offsets are meaningless,
and every citation would fail verification for a reason that looks like a model
error and is not.

Run with:  uv run pytest -m live
"""

from __future__ import annotations

import asyncio
import time

import anthropic
import pytest

from app.core.config import Settings
from app.llm.citations import (
    parse_cited_response,
    request_cited_answer,
    source_ids_in_order,
)
from app.llm.client import LLMClient, aclose_client
from app.llm.pricing import cost_usd
from app.pipeline.extract import run_extract_stage
from app.pipeline.validate import CitationVerifier, classify_credibility
from app.schemas.evidence import Claim, SourceRef, source_id_for
from app.schemas.runs import TokenUsage
from app.schemas.source import Chunk, FetchedSource, content_hash

pytestmark = pytest.mark.live


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


# Realistic source text, written to contain quotable factual sentences.
FCA_TEXT = (
    "General insurance value measures\n\n"
    "The Financial Conduct Authority publishes value measures data twice "
    "yearly for home, motor, travel and add-on products.\n\n"
    "Claims acceptance rates for home emergency cover averaged 61% across "
    "reporting firms in the most recent period.\n\n"
    "Add-on products showed materially lower claims acceptance than core "
    "cover, with several firms reporting rates below 40%.\n\n"
    "The regulator has stated that a low acceptance rate is not by itself "
    "evidence of poor value where exclusions are clearly disclosed."
)

ABI_TEXT = (
    "Pet insurance market data\n\n"
    "Gross written premium for UK pet insurance reached 1,650 million pounds "
    "in the reporting year.\n\n"
    "Lifetime policies accounted for 78% of that premium, with time-limited "
    "and accident-only cover making up the remainder.\n\n"
    "Average premiums rose by 12.4% year on year, faster than general "
    "inflation over the same period."
)


def _source(url: str, text: str, title: str) -> FetchedSource:
    return FetchedSource(
        url=url,
        final_url=url,
        title=title,
        text=text,
        content_hash=content_hash(text),
        status_code=200,
        byte_length=len(text.encode("utf-8")),
    )


SOURCES = [
    _source("https://www.fca.org.uk/data/value-measures", FCA_TEXT, "Value measures"),
    _source("https://www.abi.org.uk/data/pet-insurance", ABI_TEXT, "Pet insurance data"),
]


def _verifier() -> CitationVerifier:
    refs: dict[str, SourceRef] = {}
    texts: dict[str, str] = {}
    for source in SOURCES:
        source_id = source_id_for(str(source.final_url))
        refs[source_id] = SourceRef(
            source_id=source_id,
            url=str(source.final_url),
            title=source.title,
            domain=source.domain,
            content_hash=source.content_hash,
            credibility=classify_credibility(source.domain),
            publisher=source.domain,
        )
        texts[source_id] = source.text
    return CitationVerifier(refs, texts)


def _chunks_for(source: FetchedSource) -> list[Chunk]:
    """Whole-document chunks, so offsets must be absolute to work."""
    from app.retrieval.chunking import chunk_source

    return chunk_source(source, chunk_tokens=200, overlap_tokens=0)


class TestLiveExtraction:
    async def test_real_extraction_produces_locatable_verifiable_quotes(self) -> None:
        """Haiku must quote verbatim, and every quote must verify."""
        settings = Settings(environment="local")
        client = LLMClient(settings)
        chunks = {
            "SQ1": _chunks_for(SOURCES[0]),
            "SQ2": _chunks_for(SOURCES[1]),
        }
        questions = {
            "SQ1": "What were claims acceptance rates for home emergency cover?",
            "SQ2": "What was UK pet insurance gross written premium?",
        }

        started = time.perf_counter()
        try:
            result, metric = await run_extract_stage(
                chunks, questions, client=client, settings=settings
            )
        finally:
            await client.aclose()
            await asyncio.sleep(0)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        print(
            f"\nLIVE EXTRACT: {metric.calls} chunk calls · "
            f"{len(result.items)} evidence items · "
            f"{len(result.unlocatable)} unlocatable "
            f"({result.unlocatable_rate:.1%}) · "
            f"{metric.usage.input_tokens} in / {metric.usage.output_tokens} out · "
            f"${metric.cost_usd:.6f} · {elapsed_ms}ms"
        )
        for item in result.items:
            print(
                f"  [{item.grade.value:8}] {item.quote.start_char:>4}-"
                f"{item.quote.end_char:<4} {item.statement[:70]}"
            )

        assert result.items, "extraction produced no evidence from relevant sources"

        # Every emitted quote must verify. Stage 5 computes offsets from a real
        # match, so this failing would mean an offset-arithmetic bug.
        verifier = _verifier()
        for item in result.items:
            verdict = verifier.verify_evidence(item)
            assert verdict.ok, f"{item.evidence_id}: {verdict.failure} {verdict.detail}"

        # A high unlocatable rate means the prompt is inviting paraphrase.
        assert result.unlocatable_rate < 0.5, (
            f"{result.unlocatable_rate:.0%} of quotes could not be located; "
            f"the extraction prompt is producing paraphrase: {result.unlocatable}"
        )
        assert metric.cost_usd > 0
        assert metric.model == settings.extraction_model

    async def test_extractor_does_not_invent_evidence_for_an_irrelevant_question(
        self,
    ) -> None:
        """Returning nothing must be a real outcome, not a theoretical one.

        An extractor that always finds something is an extractor that invents.
        """
        settings = Settings(environment="local")
        client = LLMClient(settings)
        try:
            result, metric = await run_extract_stage(
                {"SQX": _chunks_for(SOURCES[0])},
                {"SQX": "What is the average rainfall in the Atacama Desert?"},
                client=client,
                settings=settings,
            )
        finally:
            await client.aclose()
            await asyncio.sleep(0)

        print(
            f"\nLIVE EXTRACT (irrelevant question): {len(result.items)} items, "
            f"${metric.cost_usd:.6f}"
        )
        for item in result.items:
            print(f"  UNEXPECTED: {item.statement[:100]}")

        assert result.items == [], (
            "extractor produced evidence for a question the source cannot "
            "answer, which is fabrication"
        )


class TestLiveNativeCitations:
    async def test_native_citation_offsets_verify_against_stored_text(self) -> None:
        """ADR-001's load-bearing assumption, tested rather than assumed.

        Their offsets index into the document they received; my verifier
        re-slices the text I hold. If those differ by a byte, the whole
        citation design does not work.
        """
        settings = Settings(environment="local")
        client = anthropic.AsyncAnthropic(api_key=settings.require_anthropic_key())

        started = time.perf_counter()
        try:
            response = await request_cited_answer(
                client,
                model=settings.synthesis_model,
                sources=SOURCES,
                question=(
                    "What were home emergency claims acceptance rates, and what "
                    "was UK pet insurance gross written premium? Cite the "
                    "passages you rely on."
                ),
            )
        finally:
            await aclose_client(client)
            await asyncio.sleep(0)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        usage = TokenUsage(
            input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        )
        cost = cost_usd(settings.synthesis_model, usage)

        blocks = parse_cited_response(list(response.content), source_ids_in_order(SOURCES))
        total_citations = sum(len(b.citations) for b in blocks)

        print(
            f"\nLIVE NATIVE CITATIONS: {len(blocks)} text blocks · "
            f"{total_citations} citations · "
            f"{usage.input_tokens} in / {usage.output_tokens} out · "
            f"${cost:.6f} · {elapsed_ms}ms"
        )

        assert total_citations > 0, (
            "the model returned no char_location citations; the document "
            "blocks or the citations flag are not reaching the API correctly"
        )

        # The assertion that matters: every returned citation verifies against
        # the text this service stored, with no adjustment.
        verifier = _verifier()
        verified = 0
        for block in blocks:
            for citation in block.citations:
                verdict = verifier.verify_citation(citation)
                marker = "OK " if verdict.ok else "BAD"
                print(
                    f"  {marker} [{citation.start_char:>4}:{citation.end_char:<4}] "
                    f"{citation.cited_text[:60]!r}"
                    + ("" if verdict.ok else f"  <- {verdict.failure}")
                )
                assert verdict.ok, (
                    f"native citation failed verification: {verdict.failure} {verdict.detail}"
                )
                verified += 1

        assert verified == total_citations
        print(f"  all {verified} native citations verified against stored text")

    async def test_claims_built_from_native_citations_are_supported(self) -> None:
        """The full chain: cited response -> Claim -> verifier -> SUPPORTED."""
        settings = Settings(environment="local")
        client = anthropic.AsyncAnthropic(api_key=settings.require_anthropic_key())
        try:
            response = await request_cited_answer(
                client,
                model=settings.synthesis_model,
                sources=SOURCES,
                question="What was UK pet insurance gross written premium?",
            )
        finally:
            await aclose_client(client)
            await asyncio.sleep(0)

        blocks = parse_cited_response(list(response.content), source_ids_in_order(SOURCES))
        cited = [b for b in blocks if b.citations]
        assert cited, "no block carried a citation"

        verifier = _verifier()
        statuses = []
        for index, block in enumerate(cited):
            claim, _ = verifier.verify_claim(
                Claim(
                    claim_id=f"LC{index}",
                    text=block.text.strip(),
                    citations=block.citations,
                )
            )
            statuses.append(claim)
            print(
                f"\n  {claim.status.value:9} conf={claim.confidence.value:8} "
                f"corrob={claim.corroboration} "
                f"verified={claim.verified_citations} "
                f"failures={[f.value for f in claim.failures]}"
            )
            print(f"    {claim.text[:140]}")
            print(f"    basis: {claim.confidence_basis}")

        # At least one cited claim must come out supported, or the chain is
        # producing citations the verifier cannot accept.
        assert any(c.is_supported for c in statuses), (
            "no claim built from native citations survived verification"
        )
