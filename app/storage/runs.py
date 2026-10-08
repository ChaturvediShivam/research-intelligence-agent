"""Run repository.

Thin persistence for research runs. Every method runs the blocking sqlite3 call
in a thread so the event loop is not stalled — `asyncio.to_thread` rather than
an async driver, because the stdlib driver is adequate at one writer and adding
`aiosqlite` would be a dependency for no measurable gain.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import structlog

from app.core.errors import NotFoundError
from app.schemas.research import ResearchPlan, ResearchRequest, ResearchRun, RunStatus
from app.schemas.runs import RunTrace
from app.storage.db import connect, init_schema

logger = structlog.get_logger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class RunRepository:
    """CRUD for research runs, keyed by opaque run id."""

    def __init__(self, database_path: Path) -> None:
        self._path = database_path
        self._conn = connect(database_path)
        init_schema(self._conn)
        # Serialises writers within this process. SQLite handles cross-process
        # locking itself; this avoids needless `database is locked` retries.
        self._write_lock = asyncio.Lock()

    # -- writes -------------------------------------------------------------

    async def create(self, request: ResearchRequest) -> ResearchRun:
        run = ResearchRun(request=request, status=RunStatus.PENDING)
        await asyncio.to_thread(self._insert, run)
        logger.info("run_created", run_id=run.id, status=run.status.value)
        return run

    def _insert(self, run: ResearchRun) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO runs (id, status, request, plan, trace, error, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, NULL, NULL, ?, ?)",
                (
                    run.id,
                    run.status.value,
                    run.request.model_dump_json(),
                    run.plan.model_dump_json() if run.plan else None,
                    run.created_at.isoformat(),
                    run.updated_at.isoformat(),
                ),
            )

    async def set_status(self, run_id: str, status: RunStatus, *, error: str | None = None) -> None:
        async with self._write_lock:
            await asyncio.to_thread(self._update_status, run_id, status, error)
        logger.info("run_status", run_id=run_id, status=status.value)

    def _update_status(self, run_id: str, status: RunStatus, error: str | None) -> None:
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE runs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (status.value, error, _now(), run_id),
            )
        if cursor.rowcount == 0:
            raise NotFoundError(f"Run {run_id!r} does not exist.", details={"run_id": run_id})

    async def save_plan(self, run_id: str, plan: ResearchPlan) -> None:
        async with self._write_lock:
            await asyncio.to_thread(self._update_plan, run_id, plan)

    def _update_plan(self, run_id: str, plan: ResearchPlan) -> None:
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE runs SET plan = ?, updated_at = ? WHERE id = ?",
                (plan.model_dump_json(), _now(), run_id),
            )
        if cursor.rowcount == 0:
            raise NotFoundError(f"Run {run_id!r} does not exist.", details={"run_id": run_id})

    async def save_trace(self, run_id: str, trace: RunTrace) -> None:
        async with self._write_lock:
            await asyncio.to_thread(self._update_trace, run_id, trace)

    def _update_trace(self, run_id: str, trace: RunTrace) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET trace = ?, updated_at = ? WHERE id = ?",
                (trace.model_dump_json(), _now(), run_id),
            )

    # -- reads --------------------------------------------------------------

    async def get(self, run_id: str) -> ResearchRun:
        row = await asyncio.to_thread(self._select, run_id)
        if row is None:
            raise NotFoundError(f"Run {run_id!r} does not exist.", details={"run_id": run_id})
        return self._to_run(row)

    def _select(self, run_id: str) -> sqlite3.Row | None:
        cursor = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row: sqlite3.Row | None = cursor.fetchone()
        return row

    async def get_trace(self, run_id: str) -> RunTrace | None:
        row = await asyncio.to_thread(self._select, run_id)
        if row is None:
            raise NotFoundError(f"Run {run_id!r} does not exist.", details={"run_id": run_id})
        raw = row["trace"]
        return RunTrace.model_validate_json(raw) if raw else None

    @staticmethod
    def _to_run(row: sqlite3.Row) -> ResearchRun:
        return ResearchRun(
            id=row["id"],
            status=RunStatus(row["status"]),
            request=ResearchRequest.model_validate_json(row["request"]),
            plan=ResearchPlan.model_validate_json(row["plan"]) if row["plan"] else None,
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def close(self) -> None:
        self._conn.close()
