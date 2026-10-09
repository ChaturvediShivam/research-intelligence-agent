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

import asyncio
import sqlite3
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.llm.client import LLMClient
from app.main import create_app
from app.pipeline.orchestrator import ResearchOrchestrator, StageObserver
from app.schemas.research import ResearchRequest, RunStatus
from app.schemas.runs import StageOutcome
from app.schemas.source import SourceCandidate
from app.storage.runs import RunRepository
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


def record_statuses(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture every status written to the database, in order.

    Asserting on the sequence rather than on a final value is the point: the
    bug was that nothing at all was written between the opening `planning`
    and the end of the run.
    """
    written: list[str] = []
    original = RunRepository.set_status

    async def recording(
        self: RunRepository, run_id: str, status: RunStatus, *, error: str | None = None
    ) -> None:
        written.append(status.value)
        await original(self, run_id, status, error=error)

    monkeypatch.setattr(RunRepository, "set_status", recording)
    return written


class TestProgressIsPersisted:
    """F-019.

    `GET /research/{id}` reported `planning` with a null plan for the whole
    of a four-minute run while the logs showed discovery and processing
    completing. Nothing was wrong with the read path — nothing had been
    written. The stage observer logged progress and persisted none of it.
    """

    @respx.mock
    def test_intermediate_statuses_reach_the_database(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mount_sources()
        written = record_statuses(monkeypatch)
        client, _ = build_client(api_settings(tmp_path))
        with client as c:
            submit(c)

        assert written[0] == RunStatus.PLANNING.value
        # The stages the logs showed, now persisted rather than only logged.
        for status in (RunStatus.DISCOVERING, RunStatus.PROCESSING, RunStatus.RETRIEVING):
            assert status.value in written, written

    @respx.mock
    def test_the_terminal_status_is_written_last(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A progress write landing after the verdict would resurrect the run.

        Progress writes are scheduled, not awaited, so without the drain
        before the final write this ordering is a race.
        """
        mount_sources()
        written = record_statuses(monkeypatch)
        client, _ = build_client(api_settings(tmp_path))
        with client as c:
            run_id = submit(c)
            final = c.get(f"/research/{run_id}").json()

        assert RunStatus(written[-1]).is_terminal, written
        assert (
            written.count(RunStatus.COMPLETED.value) + written.count(RunStatus.FAILED.value) == 1
        ), f"the verdict must be written exactly once: {written}"
        assert final["status"] == written[-1]

    @respx.mock
    def test_a_failed_run_also_records_where_it_got_to(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run that fails at discovery must still show it planned."""
        written = record_statuses(monkeypatch)
        client, _ = build_client(api_settings(tmp_path), provider=FakeSourceProvider([]))
        with client as c:
            run_id = submit(c)
            body = c.get(f"/research/{run_id}").json()

        assert body["status"] == RunStatus.FAILED.value
        assert body["error"].startswith("discover:")
        assert written[0] == RunStatus.PLANNING.value
        assert written[-1] == RunStatus.FAILED.value


class TestAPartialStageDoesNotStrandTheRun:
    """`stage_status=partial` on PROCESS is a recorded gap, not a halt.

    The production logs stopped at a PARTIAL process stage, which made the
    stage status look like the cause. It was not — PARTIAL continues. This
    pins that, so the next investigation does not re-examine it.
    """

    @respx.mock
    def test_one_unreachable_source_still_produces_a_report(self, tmp_path: Path) -> None:
        respx.get(FCA_URL).mock(
            return_value=httpx.Response(200, html=FCA_HTML, headers={"content-type": "text/html"})
        )
        respx.get(ABI_URL).mock(return_value=httpx.Response(404))

        client, _ = build_client(api_settings(tmp_path))
        with client as c:
            run_id = submit(c)
            body = c.get(f"/research/{run_id}").json()

        assert RunStatus(body["status"]).is_terminal, body["status"]
        # The run went past PROCESS: stages after it were measured.
        reached = {stage["stage"] for stage in body["stages"]}
        assert "retrieve" in reached, reached
        assert body["report"] is not None
        # And the failed source is accounted for rather than hidden.
        assert body["sources"]["failed"] >= 1


class TestInterruptedRunsAreFailedOnStartup:
    """The guarantee the in-process handler cannot make.

    A SIGKILL runs no `except` block. These assert through the real lifespan,
    because the reap has to be wired into startup to be worth anything.
    """

    def test_a_run_stranded_by_a_dead_process_is_failed_on_the_next_boot(
        self, tmp_path: Path
    ) -> None:
        settings = api_settings(tmp_path)

        # A run left mid-pipeline, exactly as an OOM kill leaves one.
        orphan = RunRepository(settings.database_path)
        run = asyncio.run(orphan.create(ResearchRequest(question=QUESTION)))
        asyncio.run(orphan.set_status(run.id, RunStatus.PROCESSING))
        orphan.close()

        client, _ = build_client(settings)
        with client as c:
            body = c.get(f"/research/{run.id}").json()

        assert body["status"] == RunStatus.FAILED.value
        assert "interrupted" in body["error"]
        assert "processing" in body["error"]

    @respx.mock
    def test_a_completed_run_is_not_disturbed_by_a_restart(self, tmp_path: Path) -> None:
        """The reap must not touch finished work on the mounted disk."""
        mount_sources()
        settings = api_settings(tmp_path)
        client, _ = build_client(settings)
        with client as c:
            run_id = submit(c)
            before = c.get(f"/research/{run_id}").json()
        assert before["status"] == RunStatus.COMPLETED.value

        restarted, _ = build_client(settings)
        with restarted as c:
            after = c.get(f"/research/{run_id}").json()

        assert after["status"] == RunStatus.COMPLETED.value
        assert after["error"] is None
        assert after["report"] is not None


class TestExceptionsAreNeverLost:
    """A run that raises must end up FAILED in the database, with a reason.

    Two paths, kept apart: a stage raising inside the orchestrator (which
    returns a FAILED result), and the route's own persistence failing (which
    must not be able to resurrect or strand the run).
    """

    @respx.mock
    def test_a_raising_stage_is_persisted_as_failed(self, tmp_path: Path) -> None:
        mount_sources()
        client, _ = build_client(
            api_settings(tmp_path),
            llm=ScriptedLLM(
                plan=make_plan(("SQ1", "SQ2")),
                plan_error=RuntimeError("the planner exploded"),
            ),
        )
        with client as c:
            run_id = submit(c)
            body = c.get(f"/research/{run_id}").json()

        assert body["status"] == RunStatus.FAILED.value
        assert body["error"], "a failed run must say why"
        assert body["error"].startswith("plan:")
        # Every later stage is recorded as skipped with the run already
        # FAILED, so the observer must not write any of them as progress.
        assert body["status"] == RunStatus.FAILED.value

    @respx.mock
    def test_a_failed_progress_write_does_not_change_the_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Progress is best-effort; the orchestrator's verdict is not.

        Progress writes are scheduled and not awaited, so an exception in one
        would otherwise surface at the `gather` and be mistaken for a
        pipeline failure.
        """
        mount_sources()
        original = RunRepository.set_status
        failed_writes: list[str] = []

        async def flaky(
            self: RunRepository, run_id: str, status: RunStatus, *, error: str | None = None
        ) -> None:
            # Fail every progress write, let the terminal verdict through.
            if not status.is_terminal and status is not RunStatus.PLANNING:
                failed_writes.append(status.value)
                raise sqlite3.OperationalError("database is locked")
            await original(self, run_id, status, error=error)

        monkeypatch.setattr(RunRepository, "set_status", flaky)
        client, _ = build_client(api_settings(tmp_path))
        with client as c:
            run_id = submit(c)
            body = c.get(f"/research/{run_id}").json()

        assert failed_writes, "the test did not exercise the failure path"
        assert body["status"] == RunStatus.COMPLETED.value
        assert body["error"] is None
        assert body["report"] is not None

    @respx.mock
    def test_a_persistence_failure_still_leaves_the_run_terminal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The route's own last-resort handler.

        The orchestrator never raises — it catches everything and returns a
        FAILED result — so this handler can only be reached by one of the
        route's own writes failing. That is exactly when it matters: a run
        left non-terminal here is a run stranded until the next restart.
        """

        async def boom(self: RunRepository, run_id: str, report: object) -> None:
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(RunRepository, "save_report", boom)
        client, _ = build_client(api_settings(tmp_path))
        with client as c:
            run_id = submit(c)
            body = c.get(f"/research/{run_id}").json()

        assert body["status"] == RunStatus.FAILED.value
        # Generic on purpose: the sqlite message must not reach the caller.
        assert body["error"] == "Internal pipeline error."
        assert "disk I/O" not in (body["error"] or "")


class TestUncitedProseNeverBecomesAClaim:
    """F-020, end to end over HTTP.

    The API returns a cited answer as interleaved cited and uncited text
    blocks. Every fixture before this one emitted cited blocks only, so the
    whole suite was blind to the shape production returns — 900+ tests passed
    while the deployed service reported 20 claims of which 11 were fake
    UNKNOWNs. `prose_blocks=True` reproduces the real shape.
    """

    @respx.mock
    def test_the_report_carries_no_no_citation_claims(self, tmp_path: Path) -> None:
        mount_sources()
        client, _ = build_client(
            api_settings(tmp_path),
            llm=ScriptedLLM(plan=make_plan(("SQ1", "SQ2")), prose_blocks=True),
        )
        with client as c:
            run_id = submit(c)
            body = c.get(f"/research/{run_id}").json()

        report = body["report"]
        assert report is not None
        every_claim = [
            claim
            for assessment in report["sub_questions"]
            for key in ("supporting_claims", "unknown_claims")
            for claim in assessment.get(key) or []
        ]
        assert every_claim, "the run produced no claims at all"
        # The artefact signature: an UNKNOWN claim whose only failure is that
        # it never carried a citation.
        manufactured = [c for c in every_claim if c["failures"] == ["no_citation"]]
        assert manufactured == [], manufactured
        # And no claim text is one of the prose fragments.
        texts = {c["text"] for c in every_claim}
        assert "- **Drivers:**" not in texts
        assert not any(t.startswith("**The attached documents") for t in texts)

    @respx.mock
    def test_every_reported_claim_carries_a_citation(self, tmp_path: Path) -> None:
        """The invariant the filter establishes, asserted at the HTTP surface."""
        mount_sources()
        client, _ = build_client(
            api_settings(tmp_path),
            llm=ScriptedLLM(plan=make_plan(("SQ1", "SQ2")), prose_blocks=True),
        )
        with client as c:
            run_id = submit(c)
            body = c.get(f"/research/{run_id}").json()

        for assessment in body["report"]["sub_questions"]:
            for claim in assessment["supporting_claims"]:
                assert claim["citations"], claim["text"]
                assert claim["verified_citations"] >= 1

    @respx.mock
    def test_unsupported_claim_rate_is_not_inflated_by_formatting(self, tmp_path: Path) -> None:
        """Metric 4 read 0.000 offline and 0.55 in production on the same code.

        The offline figure was right about the model and wrong about the
        pipeline. This pins the quantity the metric actually measures.
        """
        mount_sources()
        client, _ = build_client(
            api_settings(tmp_path),
            llm=ScriptedLLM(plan=make_plan(("SQ1", "SQ2")), prose_blocks=True),
        )
        with client as c:
            run_id = submit(c)
            report = c.get(f"/research/{run_id}").json()["report"]

        total = report["total_claims"]
        unsupported = report["unknown_claims"]
        assert total > 0
        rate = unsupported / total
        assert rate == 0.0, f"{unsupported}/{total} claims unsupported"
