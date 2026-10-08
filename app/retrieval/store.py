"""Chunk storage and retrieval: dense vectors plus lexical BM25.

One store instance holds **one corpus**. Per ADR-004 the corpus is assembled
per research question and discarded, so there is no run_id column and no
cross-run isolation to get wrong — a run gets its own store, at its own path
or in memory.

Both indexes live in the same SQLite file:

- `chunks` — the canonical rows, carrying `source_url`, `chunk_index`,
  `start_char`, `end_char`, `section` and the source's `content_hash`. Offsets
  and provenance survive the round trip, because deterministic citation
  verification later slices the *source* text at exactly these offsets.
- `chunk_vec` — a `vec0` virtual table, rowid-aligned to `chunks`.
- `chunk_fts` — an FTS5 table, rowid-aligned to `chunks`, for BM25.

Rowid alignment is what lets a dense hit and a lexical hit refer to the same
chunk without a join key of their own.
"""

from __future__ import annotations

import re
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import structlog

from app.schemas.source import Chunk

logger = structlog.get_logger(__name__)

# FTS5 treats many characters as query syntax. A research question contains
# apostrophes, hyphens and question marks, any of which turns MATCH into a
# syntax error. Queries are reduced to bare terms and re-quoted.
_TERM = re.compile(r"[A-Za-z0-9]+")
_MIN_TERM_LENGTH = 2


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A retrieved chunk with its score and provenance intact."""

    chunk_id: int
    chunk: Chunk
    score: float
    # Which retrievers contributed it, for debugging and eval breakdowns.
    retrievers: tuple[str, ...]
    rank: int


@dataclass(frozen=True, slots=True)
class Hit:
    """An internal retrieval hit: a rowid and a score."""

    chunk_id: int
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """Stores chunks and retrieves them densely and lexically."""

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def search_dense(self, query_vector: list[float], k: int) -> list[Hit]: ...

    def search_lexical(self, query: str, k: int) -> list[Hit]: ...

    def get_chunks(self, chunk_ids: list[int]) -> dict[int, Chunk]: ...

    def count(self) -> int: ...

    def close(self) -> None: ...


def _pack(vector: list[float]) -> bytes:
    """Serialise a float vector the way sqlite-vec expects."""
    return struct.pack(f"{len(vector)}f", *vector)


def to_fts_query(text: str) -> str:
    """Reduce free text to a safe FTS5 OR-query.

    Terms are extracted, lowercased, de-duplicated and double-quoted, then
    joined with OR. Quoting each term means an FTS5 keyword appearing in the
    question — `OR`, `NEAR`, `AND` — is treated as a term rather than as
    syntax. Returns `""` when nothing usable survives, which callers treat as
    "no lexical results" rather than running a malformed query.
    """
    seen: dict[str, None] = {}
    for match in _TERM.finditer(text.lower()):
        term = match.group(0)
        if len(term) >= _MIN_TERM_LENGTH:
            seen.setdefault(term, None)
    if not seen:
        return ""
    return " OR ".join(f'"{term}"' for term in seen)


class SqliteVecStore:
    """sqlite-vec + FTS5 over one corpus."""

    def __init__(self, path: Path | str = ":memory:", *, dimension: int = 384) -> None:
        self._dimension = dimension
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._load_extension()
        self._create_schema()

    def _load_extension(self) -> None:
        import sqlite_vec

        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

    def _create_schema(self) -> None:
        self._conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id           INTEGER PRIMARY KEY,
                source_url   TEXT    NOT NULL,
                chunk_index  INTEGER NOT NULL,
                text         TEXT    NOT NULL,
                start_char   INTEGER NOT NULL,
                end_char     INTEGER NOT NULL,
                section      TEXT,
                content_hash TEXT,
                UNIQUE(source_url, chunk_index)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec
                USING vec0(embedding float[{self._dimension}]);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts
                USING fts5(text, tokenize='porter unicode61');
            """
        )
        self._conn.commit()

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        *,
        content_hash: str | None = None,
    ) -> None:
        """Index chunks and their vectors, keeping all three tables aligned."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"{len(chunks)} chunks but {len(embeddings)} embeddings; "
                "they must correspond one to one."
            )
        for vector in embeddings:
            if len(vector) != self._dimension:
                raise ValueError(
                    f"embedding has {len(vector)} dimensions, index expects {self._dimension}"
                )

        with self._conn:
            for chunk, vector in zip(chunks, embeddings, strict=True):
                cursor = self._conn.execute(
                    "INSERT OR REPLACE INTO chunks "
                    "(source_url, chunk_index, text, start_char, end_char, "
                    " section, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(chunk.source_url),
                        chunk.index,
                        chunk.text,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.section,
                        content_hash,
                    ),
                )
                chunk_id = cursor.lastrowid
                # Rowid alignment across all three tables is the join key.
                self._conn.execute(
                    "INSERT OR REPLACE INTO chunk_vec(rowid, embedding) VALUES (?, ?)",
                    (chunk_id, _pack(vector)),
                )
                self._conn.execute("DELETE FROM chunk_fts WHERE rowid = ?", (chunk_id,))
                self._conn.execute(
                    "INSERT INTO chunk_fts(rowid, text) VALUES (?, ?)",
                    (chunk_id, chunk.text),
                )
        logger.info("chunks_indexed", count=len(chunks), total=self.count())

    def search_dense(self, query_vector: list[float], k: int) -> list[Hit]:
        """k-nearest neighbours by vector distance (smaller is closer)."""
        if k <= 0 or self.count() == 0:
            return []
        rows = self._conn.execute(
            "SELECT rowid, distance FROM chunk_vec "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (_pack(query_vector), k),
        ).fetchall()
        # Distance is converted to a similarity so every retriever reports
        # "higher is better" and fusion does not need per-retriever sign rules.
        return [Hit(chunk_id=row["rowid"], score=1.0 / (1.0 + row["distance"])) for row in rows]

    def search_lexical(self, query: str, k: int) -> list[Hit]:
        """BM25 over the FTS5 index (higher returned score is better)."""
        if k <= 0:
            return []
        match_query = to_fts_query(query)
        if not match_query:
            return []
        try:
            rows = self._conn.execute(
                "SELECT rowid, bm25(chunk_fts) AS score FROM chunk_fts "
                "WHERE chunk_fts MATCH ? ORDER BY score LIMIT ?",
                (match_query, k),
            ).fetchall()
        except sqlite3.OperationalError as exc:  # pragma: no cover - defensive
            # to_fts_query should make this unreachable; if it ever fires, an
            # empty lexical result is better than failing the whole retrieval.
            logger.warning("fts_query_failed", error=str(exc)[:200])
            return []
        # SQLite's bm25() is negative, with more negative meaning a better
        # match. Negate so the convention matches search_dense.
        return [Hit(chunk_id=row["rowid"], score=-row["score"]) for row in rows]

    def get_chunks(self, chunk_ids: list[int]) -> dict[int, Chunk]:
        """Rehydrate chunks by id, offsets and provenance intact."""
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})",  # noqa: S608
            chunk_ids,
        ).fetchall()
        return {
            row["id"]: Chunk(
                source_url=row["source_url"],
                index=row["chunk_index"],
                text=row["text"],
                start_char=row["start_char"],
                end_char=row["end_char"],
                section=row["section"],
            )
            for row in rows
        }

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SqliteVecStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
