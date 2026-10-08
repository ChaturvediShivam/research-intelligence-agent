"""The HTTP surface drives the real pipeline.

These are the tests whose absence let the bug ship. `test_research_api.py`
asserted `status == "completed"` after planning alone, which encoded the M1
contract as correct — so every suite run since M1 actively confirmed that a
plan-only run was a finished run (F-017).

Only the search provider and the LLM transport are stood in for. Fetching,
chunking, embedding, indexing, retrieval, quote location, offset arithmetic
and citation verification are the real implementations, reused from the M5
orchestration fixtures. A citation that verifies here verifies because the
verifier re-sliced the stored source and agreed.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import respx
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.llm.client import LLMClient
from app.main import create_app
from app.pipeline.orchestrator import ResearchOrchestrator, StageObserver
from app.schemas.research import ResearchRequest, RunStatus
from app.schemas.runs import StageOutcome
from app.schemas.source import SourceCandidate
from app.tools.fetch import SourceFetcher
from app.tools.registry import ToolContext
from tests.fixtures.fake_embedder import TermOverlapEmbedder
from tests.fixtures.fake_pipeline import FakeSourceProvider, ScriptedLLM, make_plan
from tests.integration.test_orchestration import (
    ABI_HTML,
    ABI_URL,
    FCA_HTML,
    FCA_URL,
    QUESTION,
    VOCAB,
)
from tests.security.test_ssrf import FakeResolver


def api_settings(tmp_path: Path, **kw: object) -> Settings:
    return Settings(  # type: ignore[call-arg]
        environment="test",
        _env_file=None,
        database_path=tmp_path / "runs.db",
        max_sources_per_run=6,
        retrieval_top_k=4,
        chunk_tokens=120,
        chunk_overlap_tokens=16,
        **kw,  # type: ignore[arg-type]
    )


def build_client(
    settings: Settings,
    *,
    llm: ScriptedLLM | None = None,
    provider: FakeSourceProvider | None = None,
) -> tuple[TestClient, FakeSourceProvider]:
    """An app whose pipeline components are fakes, wired exactly as production.

    The route resolves components from `app.state`, so overriding them here
    exercises the production route and the production orchestrator.
    """
    resolved_provider = provider or FakeSourceProvider(
        [
            SourceCandidate(url=FCA_URL, title="Value measures"),
            SourceCandidate(url=ABI_URL, title="Pet insurance data"),
        ]
    )
    app = create_app(settings)
    app.state.llm_client = LLMClient(
        settings, client=llm or ScriptedLLM(plan=make_plan(("SQ1", "SQ2")))
    )
    app.state.components = ToolContext(
        settings=settings,
        _provider=resolved_provider,
        _fetcher=SourceFetcher(settings, resolver=FakeResolver()),
        _embedder=TermOverlapEmbedder(VOCAB),
    )
    return TestClient(app), resolved_provider


def build_orchestrator(
    settings: Settings, *, on_stage: StageObserver | None = None
) -> ResearchOrchestrator:
    """The orchestrator alone, for contract tests that need no HTTP layer."""
    return ResearchOrchestrator(
        client=LLMClient(settings, client=ScriptedLLM(plan=make_plan(("SQ1", "SQ2")))),
        provider=FakeSourceProvider(
            [
                SourceCandidate(url=FCA_URL, title="Value measures"),
                SourceCandidate(url=ABI_URL, title="Pet insurance data"),
            ]
        ),
        fetcher=SourceFetcher(settings, resolver=FakeResolver()),
        embedder=TermOverlapEmbedder(VOCAB),
        settings=settings,
        on_stage=on_stage,
    )


def mount_sources() -> None:
    for url, html in ((FCA_URL, FCA_HTML), (ABI_URL, ABI_HTML)):
        respx.get(url).mock(
            return_value=httpx.Response(200, html=html, headers={"content-type": "text/html"})
        )


def submit(client: TestClient, **body: object) -> str:
    """POST a question and return the run id. BackgroundTasks run on exit."""
    response = client.post("/research", json={"question": QUESTION, **body})
    assert response.status_code == 202, response.text
    run_id: str = response.json()["run_id"]
    return run_id


class TestFullPipelineOverHttp:
    @respx.mock
    def test_api_run_reaches_report(self, tmp_path: Path) -> None:
        """The M10 exit criterion: POST /research runs every stage, not just PLAN."""
        mount_sources()
        client, _ = build_client(api_settings(tmp_path))

        with client as c:
            run_id = submit(c)
            # TestClient runs the BackgroundTask before the context exits, so
            # polling is not needed: by here the pipeline has finished.
            detail = c.get(f"/research/{run_id}").json()

        assert detail["status"] == RunStatus.COMPLETED.value, detail["error"]

        # The persisted trace carries every stage that recorded a measurement.
        # REPORT and ASSESS are deterministic and record no StageMetric, so
        # they are absent here by design — their evidence is the report below.
        stages = [s["stage"] for s in detail["stages"]]
        for stage in ("plan", "discover", "process", "retrieve", "extract", "synthesise"):
            assert stage in stages, f"{stage} missing from {stages}"

        # The bug this file exists to prevent: the trace used to hold `plan`
        # and nothing else, because the route never called the orchestrator.
        assert stages != ["plan"], "pipeline stopped after planning"
        assert len(stages) >= 6, f"pipeline stopped early: {stages}"

        # Stage 8 ran: a report only exists if REPORT and ASSESS completed.
        assert detail["report"] is not None
        # Cost is attributed across stages, not booked entirely to planning.
        assert len(detail["cost"]["by_stage"]) > 1, detail["cost"]["by_stage"]

    @respx.mock
    def test_get_returns_the_persisted_report_with_citations(self, tmp_path: Path) -> None:
        mount_sources()
        client, _ = build_client(api_settings(tmp_path))

        with client as c:
            run_id = submit(c)
            detail = c.get(f"/research/{run_id}").json()

        report = detail["report"]
        assert report is not None, "the report was not persisted or not returned"
        assert report["run_id"] == run_id
        assert report["executive_summary"]
        assert report["sub_questions"]

        # Citations are real, verified, and traceable to a fetched source.
        assert detail["citations"], "no citations surfaced"
        assert report["verified_citations"] > 0
        for citation in detail["citations"]:
            assert citation["source_id"]
            assert citation["cited_text"]
            assert citation["end_char"] > citation["start_char"]

        # Source provenance reached the response.
        assert detail["sources"]["fetched"] == 2
        assert detail["sources"]["domains"]

    @respx.mock
    def test_the_report_outlives_the_process(self, tmp_path: Path) -> None:
        """A second app on the same database still serves the finished run."""
        mount_sources()
        settings = api_settings(tmp_path)
        client, _ = build_client(settings)
        with client as c:
            run_id = submit(c)
            assert c.get(f"/research/{run_id}").json()["report"] is not None

        # A fresh app — new process, same SQLite file on the mounted disk.
        with TestClient(create_app(settings)) as c2:
            detail = c2.get(f"/research/{run_id}").json()

        assert detail["status"] == RunStatus.COMPLETED.value
        assert detail["report"]["executive_summary"]
        assert detail["citations"]

    @respx.mock
    def test_the_run_id_is_adopted_not_regenerated(self, tmp_path: Path) -> None:
        """The orchestrator must not mint a second id for an existing row."""
        mount_sources()
        client, _ = build_client(api_settings(tmp_path))

        with client as c:
            run_id = submit(c)
            detail = c.get(f"/research/{run_id}").json()

        assert detail["run_id"] == run_id
        # The report and the trace are keyed to the id the caller was given.
        assert detail["report"]["run_id"] == run_id
        assert detail["cost"]["total_usd"] > 0


class TestLimitsAreEnforced:
    @respx.mock
    def test_api_run_honours_max_sources(self, tmp_path: Path) -> None:
        """`request.max_sources` reaches discovery, instead of being ignored."""
        mount_sources()
        client, provider = build_client(api_settings(tmp_path))

        with client as c:
            run_id = submit(c, max_sources=1)
            detail = c.get(f"/research/{run_id}").json()

        # One source fetched, though two were available and mounted.
        assert detail["sources"]["fetched"] == 1, detail["sources"]
        assert detail["status"] == RunStatus.COMPLETED.value

    @respx.mock
    def test_api_run_honours_the_cost_ceiling(self, tmp_path: Path) -> None:
        """A ceiling below the run's cost fails the run instead of overspending."""
        mount_sources()
        # Far below the cheapest possible run, so the ceiling must trigger.
        client, _ = build_client(api_settings(tmp_path, max_cost_usd_per_run=0.000001))

        with client as c:
            run_id = submit(c)
            detail = c.get(f"/research/{run_id}").json()

        assert detail["status"] == RunStatus.FAILED.value
        assert detail["error"] is not None
        assert "ceiling" in detail["error"].lower()

    @respx.mock
    def test_a_generous_ceiling_does_not_fail_the_run(self, tmp_path: Path) -> None:
        """Control for the test above: the ceiling, not the wiring, failed it."""
        mount_sources()
        client, _ = build_client(api_settings(tmp_path, max_cost_usd_per_run=100.0))

        with client as c:
            run_id = submit(c)
            detail = c.get(f"/research/{run_id}").json()

        assert detail["status"] == RunStatus.COMPLETED.value, detail["error"]


class TestFailuresArePersisted:
    @respx.mock
    def test_a_discovery_failure_is_recorded_not_reported_as_completed(
        self, tmp_path: Path
    ) -> None:
        """A run that found nothing must not claim success."""
        mount_sources()
        client, _ = build_client(
            api_settings(tmp_path),
            provider=FakeSourceProvider([]),
        )

        with client as c:
            run_id = submit(c)
            detail = c.get(f"/research/{run_id}").json()

        assert detail["status"] != RunStatus.COMPLETED.value
        # The plan survives even though the run did not finish.
        assert detail["plan"] is not None


class TestOrchestratorContract:
    """The two new orchestrator affordances, pinned directly."""

    @respx.mock
    async def test_a_supplied_run_id_is_adopted(self, tmp_path: Path) -> None:
        mount_sources()
        settings = api_settings(tmp_path)
        result = await build_orchestrator(settings).run(
            ResearchRequest(question=QUESTION), run_id="run_caller0000000001"
        )
        assert result.run_id == "run_caller0000000001"
        assert result.trace.run_id == "run_caller0000000001"
        assert result.report is not None
        assert result.report.run_id == "run_caller0000000001"

    @respx.mock
    async def test_an_omitted_run_id_is_still_generated(self, tmp_path: Path) -> None:
        """Existing callers (MCP, the eval runner) are unaffected."""
        mount_sources()
        result = await build_orchestrator(api_settings(tmp_path)).run(
            ResearchRequest(question=QUESTION)
        )
        assert result.run_id.startswith("run_")
        assert result.run_id != "run_caller0000000001"

    @respx.mock
    async def test_a_failing_observer_does_not_fail_the_run(self, tmp_path: Path) -> None:
        """An observer is for telemetry; it must not be able to break research."""
        mount_sources()
        seen: list[str] = []

        def explode(outcome: StageOutcome, status: RunStatus) -> None:
            seen.append(outcome.stage.value)
            raise RuntimeError("observer is broken")

        result = await build_orchestrator(api_settings(tmp_path), on_stage=explode).run(
            ResearchRequest(question=QUESTION)
        )

        assert result.status is RunStatus.COMPLETED, result.error
        assert seen, "the observer was never called"

    @respx.mock
    async def test_the_observer_sees_every_recorded_stage(self, tmp_path: Path) -> None:
        mount_sources()
        seen: list[tuple[str, str]] = []
        result = await build_orchestrator(
            api_settings(tmp_path),
            on_stage=lambda outcome, status: seen.append(
                (outcome.stage.value, outcome.status.value)
            ),
        ).run(ResearchRequest(question=QUESTION))

        assert [stage for stage, _ in seen] == [outcome.stage.value for outcome in result.stages]
        assert ("report", "passed") in seen
        assert ("assess", "passed") in seen
