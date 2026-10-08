"""SQLite connection handling.

One file, plain SQL, no ORM. The schema is small and the queries are simple;
an ORM would be a dependency and an abstraction for no benefit here.

`sqlite3` is used in a thread executor by the repository layer because the
stdlib driver is synchronous. At this scale (one writer, short transactions)
that is correct and uncomplicated.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    request     TEXT NOT NULL,        -- JSON
    plan        TEXT,                 -- JSON, null until stage 1 completes
    trace       TEXT,                 -- JSON RunTrace
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_status     ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
"""


def connect(path: Path) -> sqlite3.Connection:
    """Open a connection with the pragmas this application depends on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL lets a reader (GET /research/{id}) proceed while the background task
    # writes stage progress — the whole point of returning an id immediately.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if absent. Idempotent, so it is safe on every startup."""
    conn.executescript(SCHEMA)
    conn.commit()
