"""Per-stage trace assertions (M9).

Tracing must observe execution without controlling it, and must be safe to
ship. These tests run the real pipeline — succeeding, partially failing and
failing outright — and assert against the exported trace.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.core.config import Settings
from app.llm.client import LLMClient
from app.observability.trace import (
    TRACE_VERSION,
    Measurement,
    RunTraceRecord,
    build_trace,
)
from app.pipeline.orchestrator import ResearchOrchestrator
from app.retrieval.embeddings import FastEmbedEmbedder
from app.schemas.research import ResearchRequest
from app.schemas.runs import Stage, StageStatus
from app.schemas.source import SourceCandidate
from app.tools.fetch import SourceFetcher
from tests.fixtures.fake_pipeline import FakeSourceProvider, ScriptedLLM
from tests.security.test_ssrf import FakeResolver

URL = "https://www.fca.org.uk/value-measures"
SECRET = "sk-ant-this-must-never-appear"

HTML = """
<html><head><title>FCA value measures</title></head><body>
<h1>General insurance value measures</h1>
<p>The Financial Conduct Authority publishes general insurance value measures
data covering claims frequencies and claims acceptance rates.</p>
<p>Claims acceptance rates for home emergency products averaged 78 per cent
across the reporting period, the lowest of any product line reported.</p>
</body></html>
"""

CANONICAL_ORDER = [
    Stage.PLAN,
    Stage.DISCOVER,
    Stage.PROCESS,
    Stage.RETRIEVE,
    Stage.EXTRACT,
    Stage.VALIDATE,
    Stage.SYNTHESISE,
    Stage.REPORT,
]


@pytest.fixture(scope="module")
def embedder() -> FastEmbedEmbedder:
    return FastEmbedEmbedder()


@pytest.fixture
def settings(tmp_path) -> Settings:  # type: ignore[no-untyped-def]
    return Settings(
        anthropic_api_key=SECRET,
        environment="test",
        _env_file=None,  # type: ignore[call-arg]
        sqlite_path=str(tmp_path / "obs.db"),
        chunk_tokens=160,
        max_sources_per_run=2,
    )


async def run(settings: Settings, embedder, *, response, llm=None):  # type: ignore[no-untyped-def]
    orchestrator = ResearchOrchestrator(
        client=LLMClient(
            settings,
            client=llm or ScriptedLLM(sub_question_ids=["SQ1"], discovery_turns=1),
        ),
        provider=FakeSourceProvider([SourceCandidate(url=URL, title="FCA")]),
        fetcher=SourceFetcher(settings, resolver=FakeResolver()),
        embedder=embedder,
        settings=settings,
    )
    with respx.mock:
        respx.get(URL).mock(return_value=response)
        return await orchestrator.run(
            ResearchRequest(question="What were UK claims acceptance rates?")
        )


# ==========================================================================
# Each executed stage produces a trace record
# ==========================================================================


class TestStageRecords:
    async def test_every_executed_stage_appears(self, settings, embedder) -> None:
        result = await run(settings, embedder, response=httpx.Response(200, html=HTML))
        trace = build_trace(result)
        traced = {s.stage for s in trace.stages}
        for stage in CANONICAL_ORDER:
            assert stage in traced, f"{stage.value} missing from trace"

    async def test_stage_order_is_preserved(self, settings, embedder) -> None:
        """The trace reads in execution order, so a reordering is visible."""
        result = await run(settings, embedder, response=httpx.Response(200, html=HTML))
        trace = build_trace(result)
        traced = [s.stage for s in trace.stages if s.stage in set(CANONICAL_ORDER)]
        assert traced == CANONICAL_ORDER

    async def test_a_successful_stage_is_observable(self, settings, embedder) -> None:
        result = await run(settings, embedder, response=httpx.Response(200, html=HTML))
        trace = build_trace(result)
        plan = next(s for s in trace.stages if s.stage is Stage.PLAN)
        assert plan.status is StageStatus.PASSED
        assert plan.duration_ms >= 0
        assert plan.calls >= 1
        assert plan.model

    async def test_model_calls_and_tokens_are_recorded(self, settings, embedder) -> None:
        result = await run(settings, embedder, response=httpx.Response(200, html=HTML))
        trace = build_trace(result)
        calling = [s for s in trace.stages if s.calls]
        assert calling
        for stage in calling:
            assert stage.output_tokens > 0
            assert stage.tokens_provenance is Measurement.MEASURED

    async def test_counts_describe_the_run(self, settings, embedder) -> None:
        result = await run(settings, embedder, response=httpx.Response(200, html=HTML))
        counts = build_trace(result).counts
        assert counts["sources_fetched"] == 1
        assert counts["chunks_indexed"] >= 1
        assert counts["evidence_items"] >= 1
        assert counts["citations_verified"] >= 1
        assert "citations_rejected" in counts

    async def test_sources_are_identified_by_hash(self, settings, embedder) -> None:
        result = await run(settings, embedder, response=httpx.Response(200, html=HTML))
        trace = build_trace(result)
        assert len(trace.sources) == 1
        source = trace.sources[0]
        assert len(source.content_hash) == 64
        assert source.text_chars > 0
        assert source.domain == "fca.org.uk"
        assert source.credibility == "primary"

    async def test_run_level_status_matches_stage_level(self, settings, embedder) -> None:
        result = await run(settings, embedder, response=httpx.Response(200, html=HTML))
        trace = build_trace(result)
        assert trace.status == "completed"
        assert not [s for s in trace.stages if s.status is StageStatus.FAILED]


# ==========================================================================
# Failures stay visible
# ==========================================================================


class TestFailuresAreObservable:
    async def test_a_partial_stage_is_observable(self, settings, embedder) -> None:
        """A source that could not be fetched must show as a partial stage."""
        result = await run(settings, embedder, response=httpx.Response(403))
        trace = build_trace(result)
        process = next(s for s in trace.stages if s.stage is Stage.PROCESS)
        assert process.status in {StageStatus.PARTIAL, StageStatus.FAILED}
        assert trace.failed_stages()

    async def test_an_exception_does_not_vanish_from_the_trace(self, settings, embedder) -> None:
        """A stage that raised must be diagnosable from the trace alone."""
        llm = ScriptedLLM(
            sub_question_ids=["SQ1"],
            discovery_turns=1,
            plan_error=RuntimeError("plan exploded"),
        )
        result = await run(settings, embedder, response=httpx.Response(200, html=HTML), llm=llm)
        trace = build_trace(result)
        assert trace.status == "failed"
        assert trace.error
        plan = next(s for s in trace.stages if s.stage is Stage.PLAN)
        assert plan.status is StageStatus.FAILED

    async def test_a_partial_run_remains_diagnosable(self, settings, embedder) -> None:
        """Counts must still be present when the run did not fully succeed."""
        result = await run(settings, embedder, response=httpx.Response(404))
        trace = build_trace(result)
        assert trace.counts["sources_failed"] >= 1
        assert trace.run_id
        assert trace.total_duration_ms >= 0
        assert trace.stage_order

    async def test_stages_that_never_ran_are_still_listed(self, settings, embedder) -> None:
        """A skipped stage is a fact worth seeing, not an absence."""
        llm = ScriptedLLM(
            sub_question_ids=["SQ1"],
            discovery_turns=1,
            plan_error=RuntimeError("plan exploded"),
        )
        result = await run(settings, embedder, response=httpx.Response(200, html=HTML), llm=llm)
        trace = build_trace(result)
        statuses = {s.stage: s.status for s in trace.stages}
        assert statuses[Stage.PLAN] is StageStatus.FAILED
        assert StageStatus.SKIPPED in set(statuses.values())


# ==========================================================================
# Measured vs derived vs unavailable
# ==========================================================================


class TestMeasurementProvenance:
    async def test_duration_is_measured(self, settings, embedder) -> None:
        trace = build_trace(await run(settings, embedder, response=httpx.Response(200, html=HTML)))
        assert trace.total_duration_provenance is Measurement.MEASURED
        assert trace.total_duration_ms > 0

    async def test_cost_is_derived_not_measured(self, settings, embedder) -> None:
        """The API returns tokens, not dollars. Saying otherwise would be false."""
        trace = build_trace(await run(settings, embedder, response=httpx.Response(200, html=HTML)))
        assert trace.total_cost_provenance is Measurement.DERIVED
        for stage in trace.stages:
            if stage.calls:
                assert stage.cost_provenance is Measurement.DERIVED

    async def test_a_stage_with_no_model_call_reports_tokens_unavailable(
        self, settings, embedder
    ) -> None:
        """Not zero. A zero that means "never seen" is a different fact."""
        trace = build_trace(await run(settings, embedder, response=httpx.Response(200, html=HTML)))
        retrieve = next(s for s in trace.stages if s.stage is Stage.RETRIEVE)
        assert retrieve.calls == 0
        assert retrieve.tokens_provenance is Measurement.UNAVAILABLE
        assert "retrieve.tokens" in trace.unavailable

    async def test_cache_hit_rate_is_none_when_there_is_no_input(self, settings, embedder) -> None:
        trace = build_trace(await run(settings, embedder, response=httpx.Response(200, html=HTML)))
        retrieve = next(s for s in trace.stages if s.stage is Stage.RETRIEVE)
        assert retrieve.cache_hit_rate is None

    async def test_the_trace_is_versioned(self, settings, embedder) -> None:
        trace = build_trace(await run(settings, embedder, response=httpx.Response(200, html=HTML)))
        assert trace.trace_version == TRACE_VERSION


# ==========================================================================
# The trace is safe to ship
# ==========================================================================


class TestNoLeakage:
    async def test_the_api_key_never_appears(self, settings, embedder) -> None:
        trace = build_trace(await run(settings, embedder, response=httpx.Response(200, html=HTML)))
        assert SECRET not in trace.model_dump_json()

    async def test_source_text_is_not_dumped(self, settings, embedder) -> None:
        """Observability must not reproduce documents in a log line."""
        trace = build_trace(await run(settings, embedder, response=httpx.Response(200, html=HTML)))
        blob = trace.model_dump_json()
        assert "averaged 78 per cent" not in blob
        assert "Financial Conduct Authority" not in blob

    async def test_no_prompt_text_is_dumped(self, settings, embedder) -> None:
        trace = build_trace(await run(settings, embedder, response=httpx.Response(200, html=HTML)))
        blob = trace.model_dump_json()
        assert "You are the source-discovery step" not in blob
        assert "retrieved source material" not in blob

    async def test_the_question_is_not_carried(self, settings, embedder) -> None:
        """The question can carry caller data; the trace does not need it."""
        trace = build_trace(await run(settings, embedder, response=httpx.Response(200, html=HTML)))
        assert "claims acceptance rates" not in trace.model_dump_json().lower()

    async def test_the_trace_serialises_and_round_trips(self, settings, embedder) -> None:
        """It has to survive being written to an artifact and read back."""
        trace = build_trace(await run(settings, embedder, response=httpx.Response(200, html=HTML)))
        restored = RunTraceRecord.model_validate_json(trace.model_dump_json())
        assert restored.run_id == trace.run_id
        assert restored.stage_order == trace.stage_order


# ==========================================================================
# Tracing observes; it does not control
# ==========================================================================


class TestTracingDoesNotControl:
    def test_the_orchestrator_does_not_import_the_trace_exporter(self) -> None:
        """If tracing could change behaviour, it would be business logic."""
        import inspect

        from app.pipeline import orchestrator

        assert "observability" not in inspect.getsource(orchestrator)

    async def test_building_a_trace_twice_is_identical_and_harmless(
        self, settings, embedder
    ) -> None:
        result = await run(settings, embedder, response=httpx.Response(200, html=HTML))
        first = build_trace(result)
        second = build_trace(result)
        assert first.model_dump_json() == second.model_dump_json()

    async def test_building_a_trace_does_not_mutate_the_result(self, settings, embedder) -> None:
        result = await run(settings, embedder, response=httpx.Response(200, html=HTML))
        before = (result.status, len(result.stages), result.total_cost_usd)
        build_trace(result)
        assert (result.status, len(result.stages), result.total_cost_usd) == before
