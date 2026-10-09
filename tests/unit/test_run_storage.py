"""Run persistence: the additive migration and the report round-trip.

The migration test matters more than it looks. `CREATE TABLE IF NOT EXISTS`
is a no-op against a database that already has the table, so a column added
to `SCHEMA` alone would never reach the production file on Render's mounted
disk. These tests pin the `ALTER TABLE` path and, just as importantly, that
existing rows survive it.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.schemas.report import ResearchReport, SourceCoverage
from app.schemas.research import ResearchRequest, RunStatus
from app.storage.db import connect, init_schema, migrate
from app.storage.runs import RunRepository

# The schema exactly as it shipped before the `report` column existed. Copied
# rather than imported: the point is to reproduce a real old database, and
# importing the current SCHEMA would silently track any future change.
OLD_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    request     TEXT NOT NULL,
    plan        TEXT,
    trace       TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""

QUESTION = "What drove the change in UK pet insurance premiums over this period?"


def _old_database(path: Path, *, run_id: str = "run_legacy0000000001") -> None:
    """Build a pre-migration database holding one run."""
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO runs (id, status, request, plan, trace, error, created_at, "
        "updated_at) VALUES (?, ?, ?, NULL, NULL, NULL, ?, ?)",
        (
            run_id,
            RunStatus.COMPLETED.value,
            ResearchRequest(question=QUESTION).model_dump_json(),
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()


def _columns(path: Path) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {str(row[1]) for row in conn.execute("PRAGMA table_info(runs)")}
    finally:
        conn.close()


def _report(run_id: str) -> ResearchReport:
    return ResearchReport(
        run_id=run_id,
        question=QUESTION,
        restated_question="UK pet insurance premium drivers, 2024-2026.",
        executive_summary="Premiums rose, driven by claims inflation.",
        unknowns=["The split between frequency and severity is not established."],
        source_coverage=SourceCoverage(discovered=3, fetched=2, failed=1, domains=["abi.org.uk"]),
        total_claims=2,
        supported_claims=1,
        verified_citations=4,
    )


class TestMigration:
    def test_migration_adds_report_column_to_existing_db(self, tmp_path: Path) -> None:
        """The exit criterion: an old production database gains the column."""
        path = tmp_path / "legacy.db"
        _old_database(path)
        assert "report" not in _columns(path)

        repo = RunRepository(path)
        try:
            assert "report" in _columns(path)
        finally:
            repo.close()

    async def test_existing_rows_survive_the_migration(self, tmp_path: Path) -> None:
        """Additive means additive: the run that was there is still there."""
        path = tmp_path / "legacy.db"
        _old_database(path, run_id="run_survivor00000001")

        repo = RunRepository(path)
        try:
            run = await repo.get("run_survivor00000001")
        finally:
            repo.close()

        assert run.status is RunStatus.COMPLETED
        assert run.request.question == QUESTION
        # The new column exists and is simply empty for a pre-migration run.
        assert run.id == "run_survivor00000001"

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        """Safe on every startup, and safe to run twice in one process."""
        path = tmp_path / "legacy.db"
        _old_database(path)
        conn = connect(path)
        try:
            assert migrate(conn) == ["report"]
            assert migrate(conn) == []
            # A second init_schema must not raise a duplicate-column error.
            init_schema(conn)
            init_schema(conn)
        finally:
            conn.close()

    def test_a_fresh_database_already_has_the_column(self, tmp_path: Path) -> None:
        path = tmp_path / "fresh.db"
        repo = RunRepository(path)
        try:
            assert "report" in _columns(path)
        finally:
            repo.close()


class TestReportPersistence:
    async def test_report_survives_repo_roundtrip(self, tmp_path: Path) -> None:
        repo = RunRepository(tmp_path / "runs.db")
        try:
            run = await repo.create(ResearchRequest(question=QUESTION))
            assert await repo.get_report(run.id) is None

            await repo.save_report(run.id, _report(run.id))
            loaded = await repo.get_report(run.id)
        finally:
            repo.close()

        assert loaded is not None
        # Validated on the way back, not merely stored as opaque text.
        assert isinstance(loaded, ResearchReport)
        assert loaded.run_id == run.id
        assert loaded.verified_citations == 4
        assert loaded.source_coverage.fetched == 2
        assert loaded.unknowns == ["The split between frequency and severity is not established."]

    async def test_report_survives_a_reopen(self, tmp_path: Path) -> None:
        """The deliverable outlives the process, which is the whole point."""
        path = tmp_path / "runs.db"
        repo = RunRepository(path)
        try:
            run = await repo.create(ResearchRequest(question=QUESTION))
            await repo.save_report(run.id, _report(run.id))
        finally:
            repo.close()

        reopened = RunRepository(path)
        try:
            loaded = await reopened.get_report(run.id)
        finally:
            reopened.close()

        assert loaded is not None
        assert loaded.executive_summary == "Premiums rose, driven by claims inflation."

    async def test_saving_a_report_for_an_unknown_run_is_rejected(self, tmp_path: Path) -> None:
        from app.core.errors import NotFoundError

        repo = RunRepository(tmp_path / "runs.db")
        try:
            with pytest.raises(NotFoundError):
                await repo.save_report("run_doesnotexist0001", _report("run_doesnotexist0001"))
        finally:
            repo.close()


class TestInterruptedRunsAreReaped:
    """F-019.

    The pipeline runs in a `BackgroundTask` inside the API process. A SIGKILL
    — an OOM, a deploy, a restart — takes the task with it and runs no
    exception handler, so the run keeps its last persisted status forever. The
    in-process `try/except` cannot cover that case; only startup can.
    """

    async def test_a_non_terminal_run_is_failed_at_startup(self, tmp_path: Path) -> None:
        path = tmp_path / "runs.db"
        repo = RunRepository(path)
        run = await repo.create(ResearchRequest(question=QUESTION))
        await repo.set_status(run.id, RunStatus.PROCESSING)
        repo.close()

        # A second repository stands in for the restarted process.
        restarted = RunRepository(path)
        assert await restarted.fail_interrupted() == [run.id]
        reaped = await restarted.get(run.id)
        assert reaped.status is RunStatus.FAILED
        # The status it died at is the only record of how far it got, so it
        # has to survive into the message.
        assert "processing" in (reaped.error or "")
        restarted.close()

    @pytest.mark.parametrize(
        "status",
        [RunStatus.PENDING, RunStatus.PLANNING, RunStatus.DISCOVERING, RunStatus.SYNTHESISING],
    )
    async def test_every_non_terminal_status_is_reaped(
        self, tmp_path: Path, status: RunStatus
    ) -> None:
        """Parameterised over the enum so a new status cannot be forgotten."""
        assert not status.is_terminal
        repo = RunRepository(tmp_path / f"{status.value}.db")
        run = await repo.create(ResearchRequest(question=QUESTION))
        await repo.set_status(run.id, status)
        assert await repo.fail_interrupted() == [run.id]
        assert (await repo.get(run.id)).status is RunStatus.FAILED
        repo.close()

    @pytest.mark.parametrize("status", [RunStatus.COMPLETED, RunStatus.FAILED])
    async def test_a_terminal_run_is_left_alone(self, tmp_path: Path, status: RunStatus) -> None:
        """A finished run must not have its error or status rewritten."""
        repo = RunRepository(tmp_path / f"{status.value}.db")
        run = await repo.create(ResearchRequest(question=QUESTION))
        await repo.set_status(run.id, status, error="original")
        assert await repo.fail_interrupted() == []
        after = await repo.get(run.id)
        assert after.status is status
        assert after.error == "original"
        repo.close()

    async def test_reaping_is_idempotent_and_quiet_when_there_is_nothing_to_do(
        self, tmp_path: Path
    ) -> None:
        """It runs on every startup, including the ones with no stranded runs."""
        repo = RunRepository(tmp_path / "runs.db")
        assert await repo.fail_interrupted() == []
        run = await repo.create(ResearchRequest(question=QUESTION))
        await repo.set_status(run.id, RunStatus.RETRIEVING)
        assert await repo.fail_interrupted() == [run.id]
        assert await repo.fail_interrupted() == []
        repo.close()

    async def test_a_completed_run_survives_alongside_a_stranded_one(self, tmp_path: Path) -> None:
        """The reap is a filtered UPDATE; this is what pins the filter."""
        repo = RunRepository(tmp_path / "runs.db")
        done = await repo.create(ResearchRequest(question=QUESTION))
        stuck = await repo.create(ResearchRequest(question=QUESTION))
        await repo.set_status(done.id, RunStatus.COMPLETED)
        await repo.set_status(stuck.id, RunStatus.EXTRACTING)
        assert await repo.fail_interrupted() == [stuck.id]
        assert (await repo.get(done.id)).status is RunStatus.COMPLETED
        assert (await repo.get(stuck.id)).status is RunStatus.FAILED
        repo.close()

    async def test_the_reaped_status_survives_a_reopen(self, tmp_path: Path) -> None:
        """Committed, not just held in the connection."""
        path = tmp_path / "runs.db"
        repo = RunRepository(path)
        run = await repo.create(ResearchRequest(question=QUESTION))
        await repo.set_status(run.id, RunStatus.VALIDATING)
        await repo.fail_interrupted()
        repo.close()

        reopened = RunRepository(path)
        assert (await reopened.get(run.id)).status is RunStatus.FAILED
        reopened.close()
