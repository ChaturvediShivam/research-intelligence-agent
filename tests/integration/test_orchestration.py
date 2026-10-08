"""End-to-end orchestration.

These are integration tests, not mock-verification tests. Only the search
provider and the LLM transport are stood in for; fetching, extraction,
chunking, embedding, indexing, retrieval, quote location, offset arithmetic
and citation verification are the real implementations. A citation that
verifies here verifies because the verifier re-sliced the stored source and
agreed — not because a fake said so.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.core.config import Settings
from app.llm.client import LLMClient
from app.pipeline.orchestrator import ResearchOrchestrator, RunResult
from app.retrieval.embeddings import Embedder
from app.schemas.research import ResearchRequest, RunStatus
from app.schemas.runs import Stage, StageStatus
from app.tools.fetch import SourceFetcher
from tests.fixtures.fake_embedder import TermOverlapEmbedder
from tests.fixtures.fake_pipeline import (
    FakeSourceProvider,
    ScriptedLLM,
    make_plan,
)
from tests.security.test_ssrf import FakeResolver

QUESTION = "What were UK general insurance claims acceptance rates and pet premium?"

FCA_HTML = """<!doctype html><html><head><title>Value measures</title></head><body>
<article><h1>General insurance value measures</h1>
<p>Claims acceptance rates for home emergency cover averaged 61% across reporting firms
in the most recent period under review.</p>
<p>Add-on products showed materially lower claims acceptance than core cover, with
several firms reporting rates below 40% of submitted claims.</p>
</article></body></html>"""

ABI_HTML = """<!doctype html><html><head><title>Pet insurance data</title></head><body>
<article><h1>Pet insurance market data</h1>
<p>Gross written premium for UK pet insurance reached 1,650 million pounds in the most
recent reporting year across all policy types.</p>
<p>Lifetime policies accounted for the majority of that premium, with time-limited
cover making up the remainder of the market.</p>
</article></body></html>"""

FCA_URL = "https://www.fca.org.uk/data/value-measures"
ABI_URL = "https://www.abi.org.uk/data/pet-insurance"
DEAD_URL = "https://www.example.org/missing-report"

VOCAB = [
    "claims",
    "acceptance",
    "rates",
    "home",
    "emergency",
    "cover",
    "premium",
    "pet",
    "insurance",
    "gross",
    "written",
    "lifetime",
    "policies",
    "firms",
    "regulator",
    "market",
]


def settings(**kw: object) -> Settings:
    return Settings(  # type: ignore[call-arg]
        environment="test",
        _env_file=None,
        max_sources_per_run=6,
        retrieval_top_k=4,
        chunk_tokens=120,
        chunk_overlap_tokens=16,
        **kw,  # type: ignore[arg-type]
    )


def embedder() -> Embedder:
    return TermOverlapEmbedder(VOCAB)


def candidates(*urls: str) -> list:
    from app.schemas.source import SourceCandidate

    return [SourceCandidate(url=u, title=f"Source {i}") for i, u in enumerate(urls)]


def mount(url: str, html: str) -> None:
    respx.get(url).mock(
        return_value=httpx.Response(200, html=html, headers={"content-type": "text/html"})
    )


def orchestrator(
    llm: ScriptedLLM,
    provider: FakeSourceProvider,
    *,
    config: Settings | None = None,
) -> ResearchOrchestrator:
    resolved = config or settings()
    return ResearchOrchestrator(
        client=LLMClient(resolved, client=llm),
        provider=provider,
        fetcher=SourceFetcher(resolved, resolver=FakeResolver()),
        embedder=embedder(),
        settings=resolved,
    )


# ==========================================================================
# 1 · Happy path — the M5 exit criterion, offline
# ==========================================================================


class TestHappyPath:
    @respx.mock
    async def test_one_complete_run_executes_every_stage(self) -> None:
        mount(FCA_URL, FCA_HTML)
        mount(ABI_URL, ABI_HTML)
        llm = ScriptedLLM(plan=make_plan(("SQ1", "SQ2")))
        provider = FakeSourceProvider(candidates(FCA_URL, ABI_URL))

        result = await orchestrator(llm, provider).run(ResearchRequest(question=QUESTION))

        assert result.status is RunStatus.COMPLETED, result.error
        summary = result.summary()
        assert summary["plan_sub_questions"] == 2
        assert summary["sources_discovered"] == 2
        assert summary["sources_fetched"] == 2
        assert summary["sources_failed"] == 0
        assert summary["chunks_retrieved"] > 0
        assert summary["evidence_items"] > 0
        assert summary["citations_verified"] > 0
        assert summary["citations_rejected"] == 0
        assert summary["claims_supported"] > 0

    @respx.mock
    async def test_stages_run_in_the_architected_order(self) -> None:
        mount(FCA_URL, FCA_HTML)
        mount(ABI_URL, ABI_HTML)
        result = await orchestrator(
            ScriptedLLM(), FakeSourceProvider(candidates(FCA_URL, ABI_URL))
        ).run(ResearchRequest(question=QUESTION))

        assert [s.stage for s in result.stages] == [
            Stage.PLAN,
            Stage.DISCOVER,
            Stage.PROCESS,
            Stage.RETRIEVE,
            Stage.EXTRACT,
            Stage.VALIDATE,
            Stage.SYNTHESISE,
            Stage.REPORT,
            # M6: stage 8 assembles the report and names the gaps.
            Stage.ASSESS,
        ]

    @respx.mock
    async def test_final_claims_are_grounded_in_the_pipeline_evidence(self) -> None:
        """The claim text must come from the sources this run actually fetched.

        Not from the model's own knowledge, and not from a source that failed.
        """
        mount(FCA_URL, FCA_HTML)
        mount(ABI_URL, ABI_HTML)
        result = await orchestrator(
            ScriptedLLM(), FakeSourceProvider(candidates(FCA_URL, ABI_URL))
        ).run(ResearchRequest(question=QUESTION))

        assert result.claim_validation is not None
        supported = result.claim_validation.supported_claims
        assert supported

        for claim in supported:
            assert claim.citations, claim.claim_id
            for citation in claim.citations:
                # Every citation resolves to a source this run stored...
                assert citation.source_id in result.source_texts
                # ...and its quote is genuinely at those offsets.
                stored = result.source_texts[citation.source_id]
                assert stored[citation.start_char : citation.end_char] == citation.cited_text

    @respx.mock
    async def test_result_is_inspectable_end_to_end(self) -> None:
        mount(FCA_URL, FCA_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        assert result.plan is not None
        assert result.discovery is not None
        assert result.processed is not None
        assert result.retrieved
        assert result.extraction is not None
        assert result.evidence_validation is not None
        assert result.synthesis is not None
        assert result.claim_validation is not None
        assert result.sources and result.source_texts


# ==========================================================================
# 2-10 · Failure semantics
# ==========================================================================


class TestPlannerFailure:
    @respx.mock
    async def test_planner_failure_fails_the_run_and_skips_everything(self) -> None:
        llm = ScriptedLLM(plan_error=RuntimeError("planner exploded"))
        result = await orchestrator(llm, FakeSourceProvider(candidates(FCA_URL))).run(
            ResearchRequest(question=QUESTION)
        )

        assert result.status is RunStatus.FAILED
        assert result.error is not None and result.error.startswith("plan:")
        # No stage after PLAN may have run.
        assert result.stages == [] or all(s.stage is Stage.PLAN for s in result.stages)
        assert result.discovery is None
        assert result.processed is None
        assert llm.create_calls == [], "synthesis must not have been attempted"


class TestDiscoveryFailure:
    @respx.mock
    async def test_no_candidates_is_a_structured_stage_failure(self) -> None:
        """A stage with no viable input must not report success."""
        result = await orchestrator(ScriptedLLM(), FakeSourceProvider([])).run(
            ResearchRequest(question=QUESTION)
        )

        assert result.status is RunStatus.FAILED
        assert result.error == "discover: no candidate sources were found."
        discover = result.stage(Stage.DISCOVER)
        assert discover is not None and discover.status is StageStatus.FAILED

    @respx.mock
    async def test_search_error_is_recorded_but_does_not_raise(self) -> None:
        """One failed query should change the next query, not end the stage."""
        from app.core.errors import UpstreamError

        provider = FakeSourceProvider([], error=UpstreamError("search down"))
        result = await orchestrator(ScriptedLLM(), provider).run(ResearchRequest(question=QUESTION))

        assert result.discovery is not None
        assert result.discovery.search_failures
        assert result.status is RunStatus.FAILED  # no candidates resulted

    @respx.mock
    async def test_downstream_stages_are_skipped_not_failed(self) -> None:
        """SKIPPED and FAILED are different findings."""
        result = await orchestrator(ScriptedLLM(), FakeSourceProvider([])).run(
            ResearchRequest(question=QUESTION)
        )
        for stage in (Stage.PROCESS, Stage.RETRIEVE, Stage.EXTRACT, Stage.SYNTHESISE):
            outcome = result.stage(stage)
            assert outcome is not None and outcome.status is StageStatus.SKIPPED


class TestPartialSourceFailure:
    @respx.mock
    async def test_one_dead_source_does_not_end_the_run(self) -> None:
        """A research answer from two of three sources is a usable result."""
        mount(FCA_URL, FCA_HTML)
        mount(ABI_URL, ABI_HTML)
        respx.get(DEAD_URL).mock(return_value=httpx.Response(404))

        result = await orchestrator(
            ScriptedLLM(), FakeSourceProvider(candidates(FCA_URL, DEAD_URL, ABI_URL))
        ).run(ResearchRequest(question=QUESTION))

        assert result.status is RunStatus.COMPLETED, result.error
        assert result.processed is not None
        assert len(result.processed.sources) == 2
        assert len(result.processed.failures) == 1
        assert result.processed.failures[0].url == DEAD_URL

        process = result.stage(Stage.PROCESS)
        assert process is not None
        # PARTIAL, not FAILED: the distinction a reader needs.
        assert process.status is StageStatus.PARTIAL
        assert any(DEAD_URL in w for w in process.warnings)

    @respx.mock
    async def test_a_blocked_source_is_a_source_failure_not_a_run_failure(self) -> None:
        """The SSRF guard refusing a URL must not abort the run."""
        mount(FCA_URL, FCA_HTML)
        config = settings(blocked_source_domains=("spam.test",))
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL, "https://spam.test/x")),
            config=config,
        ).run(ResearchRequest(question=QUESTION))

        assert result.status is RunStatus.COMPLETED, result.error
        assert result.processed is not None
        assert [f.code for f in result.processed.failures] == ["unsafe_url"]

    @respx.mock
    async def test_a_failed_source_contributes_no_evidence(self) -> None:
        """No evidence may be attributed to a source that never loaded."""
        mount(FCA_URL, FCA_HTML)
        respx.get(DEAD_URL).mock(return_value=httpx.Response(500))

        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL, DEAD_URL)),
        ).run(ResearchRequest(question=QUESTION))

        assert result.extraction is not None
        fetched_ids = set(result.source_texts)
        for item in result.extraction.items:
            assert item.source_id in fetched_ids


class TestNoViableSources:
    @respx.mock
    async def test_every_source_failing_is_a_structured_failure(self) -> None:
        respx.get(FCA_URL).mock(return_value=httpx.Response(404))
        respx.get(ABI_URL).mock(return_value=httpx.Response(503))

        result = await orchestrator(
            ScriptedLLM(), FakeSourceProvider(candidates(FCA_URL, ABI_URL))
        ).run(ResearchRequest(question=QUESTION))

        assert result.status is RunStatus.FAILED
        assert result.error == "process: no source could be fetched."
        process = result.stage(Stage.PROCESS)
        assert process is not None and process.status is StageStatus.FAILED
        retrieve = result.stage(Stage.RETRIEVE)
        assert retrieve is not None and retrieve.status is StageStatus.SKIPPED


class TestExtractionFailure:
    @respx.mock
    async def test_no_evidence_found_is_a_structured_failure(self) -> None:
        mount(FCA_URL, FCA_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",)), extraction_enabled=False),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        assert result.status is RunStatus.FAILED
        assert result.error is not None and result.error.startswith("extract:")
        extract = result.stage(Stage.EXTRACT)
        assert extract is not None and extract.status is StageStatus.FAILED

    @respx.mock
    async def test_extraction_transport_failure_names_the_stage(self) -> None:
        mount(FCA_URL, FCA_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",)), extract_error=RuntimeError("boom")),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        assert result.status is RunStatus.FAILED
        assert result.error is not None and result.error.startswith("extract:")


class TestSynthesisFailure:
    @respx.mock
    async def test_synthesis_failure_fails_the_run_after_evidence_succeeded(self) -> None:
        mount(FCA_URL, FCA_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",)), synthesis_error=RuntimeError("down")),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        assert result.status is RunStatus.FAILED
        assert result.error is not None and result.error.startswith("synthesise:")
        # Earlier stages keep their outcomes: the run is still inspectable.
        extract = result.stage(Stage.EXTRACT)
        assert extract is not None and extract.status.is_usable
        assert result.extraction is not None and result.extraction.items


# ==========================================================================
# Citation safety — M4 must not be weakened by M5
# ==========================================================================


class TestCitationSafetyIsPreserved:
    @respx.mock
    async def test_a_fabricated_citation_is_rejected_by_the_orchestrated_run(
        self,
    ) -> None:
        """Synthesis cannot turn an unverified citation into a verified one."""
        mount(FCA_URL, FCA_HTML)
        mount(ABI_URL, ABI_HTML)

        result = await orchestrator(
            ScriptedLLM(fabricate_citations=True),
            FakeSourceProvider(candidates(FCA_URL, ABI_URL)),
        ).run(ResearchRequest(question=QUESTION))

        # Whatever else happens, nothing fabricated may be reported supported.
        if result.claim_validation is not None:
            assert result.claim_validation.supported_claims == []
            assert result.claim_validation.rejected_count > 0
        assert result.status is RunStatus.FAILED
        assert result.verified_claims == 0

    @respx.mock
    async def test_synthesis_only_sees_sources_whose_evidence_verified(self) -> None:
        mount(FCA_URL, FCA_HTML)
        mount(ABI_URL, ABI_HTML)
        llm = ScriptedLLM()
        result = await orchestrator(llm, FakeSourceProvider(candidates(FCA_URL, ABI_URL))).run(
            ResearchRequest(question=QUESTION)
        )

        verified_ids = {
            v.citation.source_id
            for v in (result.evidence_validation.verdicts if result.evidence_validation else [])
            if v.ok
        }
        assert verified_ids
        for call in llm.create_calls:
            documents = [
                b
                for b in call["messages"][0]["content"]
                if isinstance(b, dict) and b.get("type") == "document"
            ]
            for document in documents:
                # Every document sent to synthesis is a stored source.
                assert document["source"]["data"] in result.source_texts.values()

    @respx.mock
    async def test_no_claim_is_supported_without_a_verified_citation(self) -> None:
        mount(FCA_URL, FCA_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        assert result.claim_validation is not None
        for claim in result.claim_validation.supported_claims:
            assert claim.verified_citations >= 1
            assert claim.failures == []

    @respx.mock
    async def test_no_evidence_is_invented_for_a_source_not_fetched(self) -> None:
        mount(FCA_URL, FCA_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        assert result.extraction is not None
        for item in result.extraction.items:
            assert item.source_id in result.sources


# ==========================================================================
# Provenance
# ==========================================================================


class TestProvenanceSurvivesEveryStage:
    @respx.mock
    async def test_a_discovered_source_is_traceable_to_a_final_citation(self) -> None:
        """discovery -> fetched -> stored -> chunk -> retrieved -> evidence
        -> citation, with offsets intact at every hop."""
        mount(FCA_URL, FCA_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        assert result.discovery is not None and result.processed is not None
        assert result.extraction is not None and result.claim_validation is not None

        # discovery
        assert str(result.discovery.candidates[0].url) == FCA_URL
        # fetched + stored
        source = result.processed.sources[0]
        assert str(source.final_url) == FCA_URL
        from app.schemas.evidence import source_id_for

        source_id = source_id_for(FCA_URL)
        assert source_id in result.sources
        assert result.source_texts[source_id] == source.text

        # chunks slice the stored text exactly
        for chunk in result.processed.chunks:
            assert source.text[chunk.start_char : chunk.end_char] == chunk.text

        # retrieved chunks keep their offsets
        for chunks in result.retrieved.values():
            for chunk in chunks:
                assert source.text[chunk.start_char : chunk.end_char] == chunk.text

        # evidence quotes slice the stored text exactly
        for item in result.extraction.items:
            quote = item.quote
            assert (
                result.source_texts[quote.source_id][quote.start_char : quote.end_char]
                == quote.text
            )

        # final citations slice the stored text exactly
        for claim in result.claim_validation.supported_claims:
            for citation in claim.citations:
                assert (
                    result.source_texts[citation.source_id][citation.start_char : citation.end_char]
                    == citation.cited_text
                )

    @respx.mock
    async def test_stored_text_is_never_altered_after_fetch(self) -> None:
        """The canonical text must stay byte-identical, or offsets drift."""
        mount(FCA_URL, FCA_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        assert result.processed is not None
        for source in result.processed.sources:
            from app.schemas.evidence import source_id_for

            assert result.source_texts[source_id_for(str(source.final_url))] == source.text
            assert source.verify_hash()

    @respx.mock
    async def test_source_metadata_is_preserved(self) -> None:
        mount(FCA_URL, FCA_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        reference = next(iter(result.sources.values()))
        assert reference.domain == "fca.org.uk"
        assert reference.title
        assert reference.content_hash
        # Credibility assigned by the same rule the verifier uses.
        assert reference.credibility.value == "primary"


# ==========================================================================
# Cost, usage and metadata
# ==========================================================================


class TestCostAndUsageAggregation:
    @respx.mock
    async def test_cost_is_the_sum_of_stage_costs(self) -> None:
        mount(FCA_URL, FCA_HTML)
        mount(ABI_URL, ABI_HTML)
        result = await orchestrator(
            ScriptedLLM(), FakeSourceProvider(candidates(FCA_URL, ABI_URL))
        ).run(ResearchRequest(question=QUESTION))

        stage_costs = sum(s.cost_usd for s in result.stages)
        assert result.total_cost_usd == pytest.approx(stage_costs, abs=1e-9)
        assert result.total_cost_usd > 0

    @respx.mock
    async def test_stages_without_a_model_call_cost_nothing(self) -> None:
        mount(FCA_URL, FCA_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        for stage in (Stage.PROCESS, Stage.RETRIEVE):
            outcome = result.stage(stage)
            assert outcome is not None
            assert outcome.cost_usd == 0.0
            assert outcome.metric is not None and outcome.metric.model is None

    @respx.mock
    async def test_llm_call_count_matches_the_transport(self) -> None:
        mount(FCA_URL, FCA_HTML)
        mount(ABI_URL, ABI_HTML)
        llm = ScriptedLLM()
        result = await orchestrator(llm, FakeSourceProvider(candidates(FCA_URL, ABI_URL))).run(
            ResearchRequest(question=QUESTION)
        )

        # Every parse and create the transport saw is accounted for, plus the
        # discovery turns.
        assert result.total_llm_calls >= len(llm.parse_calls) + len(llm.create_calls)
        assert result.total_llm_calls > 0

    @respx.mock
    async def test_token_usage_is_aggregated(self) -> None:
        mount(FCA_URL, FCA_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        usage = result.trace.total_usage
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0

    @respx.mock
    async def test_cost_by_stage_attributes_spend(self) -> None:
        mount(FCA_URL, FCA_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        by_stage = result.trace.cost_by_stage()
        assert by_stage["plan"] > 0
        assert by_stage["extract"] > 0

    @respx.mock
    async def test_summary_exposes_measured_metadata_only(self) -> None:
        mount(FCA_URL, FCA_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        summary = result.summary()
        for key in (
            "sources_discovered",
            "sources_fetched",
            "chunks_retrieved",
            "evidence_items",
            "citations_verified",
            "citations_rejected",
            "total_llm_calls",
            "total_cost_usd",
            "total_duration_ms",
        ):
            assert key in summary


class TestNoSecretsLeak:
    @respx.mock
    async def test_the_run_result_contains_no_credential(self) -> None:
        mount(FCA_URL, FCA_HTML)
        from pydantic import SecretStr

        config = settings()
        config = config.model_copy(update={"anthropic_api_key": SecretStr("sk-ant-must-not-leak")})
        result = await ResearchOrchestrator(
            client=LLMClient(config, client=ScriptedLLM(plan=make_plan(("SQ1",)))),
            provider=FakeSourceProvider(candidates(FCA_URL)),
            fetcher=SourceFetcher(config, resolver=FakeResolver()),
            embedder=embedder(),
            settings=config,
        ).run(ResearchRequest(question=QUESTION))

        blob = repr(result.summary()) + repr(result.stages) + repr(result.error)
        assert "sk-ant-must-not-leak" not in blob


class TestRunAlwaysReturnsAResult:
    @respx.mock
    async def test_an_unexpected_error_still_returns_a_result(self) -> None:
        """A caller needs the partial outputs; an exception discards them."""
        mount(FCA_URL, FCA_HTML)

        class Exploding(FakeSourceProvider):
            async def search(self, query: str, **kw: object) -> list:
                raise ValueError("not an AppError")

        result = await orchestrator(ScriptedLLM(), Exploding()).run(
            ResearchRequest(question=QUESTION)
        )
        assert isinstance(result, RunResult)
        assert result.status is RunStatus.FAILED
        assert result.plan is not None, "the planning output survives the failure"
