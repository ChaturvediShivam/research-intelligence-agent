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
    report      TEXT,                 -- JSON ResearchReport, null until stage 8
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


# Columns added after the first schema shipped. `CREATE TABLE IF NOT EXISTS`
# is a no-op on a database that already has the table, so a new column in
# SCHEMA above would never reach a deployed database — the production file on
# the mounted disk keeps the shape it was created with. These are applied as
# additive `ALTER TABLE`s instead.
#
# Additive only, by rule: no DROP, no table rebuild, no rewrite of existing
# rows. A migration that loses a research run costs real money to reproduce.
_ADDITIVE_COLUMNS: tuple[tuple[str, str], ...] = (("report", "report TEXT"),)


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Column names of `table`. Indexed by position, so the caller's
    `row_factory` cannot change the answer."""
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Add any missing additive columns. Returns the ones added.

    Idempotent: a column already present is left alone, so this is safe to
    run on every startup and safe to run twice.
    """
    present = _existing_columns(conn, "runs")
    added: list[str] = []
    for column, ddl in _ADDITIVE_COLUMNS:
        if column not in present:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {ddl}")  # noqa: S608 - fixed literals
            added.append(column)
    if added:
        conn.commit()
    return added


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if absent, then apply additive migrations.

    Idempotent, so it is safe on every startup.
    """
    conn.executescript(SCHEMA)
    conn.commit()
    migrate(conn)
