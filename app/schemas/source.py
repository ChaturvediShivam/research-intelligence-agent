"""Source and chunk schemas.

The load-bearing invariant in this module: a `Chunk`'s `start_char` and
`end_char` index into `FetchedSource.text` exactly, such that

    source.text[chunk.start_char : chunk.end_char] == chunk.text

Deterministic citation verification (ADR-002) depends on that holding. Text is
normalised **once**, at fetch time, and the normalised form is what gets
stored, chunked, embedded and cited against. Any later re-normalisation would
shift offsets and silently break every citation.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from pydantic import BaseModel, Field, HttpUrl, field_validator

from app.schemas.research import SourceType


class SourceCandidate(BaseModel):
    """A search result, before it has been fetched.

    Title and snippet come from a search index and are therefore untrusted
    text; they are never treated as instruction.
    """

    url: HttpUrl
    title: str = Field(max_length=500)
    snippet: str = Field(default="", max_length=2000)
    # Which sub-question this candidate was found for, so coverage can be
    # attributed per sub-question rather than only per run.
    sub_question_id: str | None = None
    # The planner's expectation, not an assessment of the fetched source.
    expected_source_type: SourceType | None = None

    @property
    def domain(self) -> str:
        return (self.url.host or "").lower().removeprefix("www.")


class FetchedSource(BaseModel):
    """A fetched, extracted and normalised source document.

    `text` is canonical: offsets, chunks, embeddings and citations all refer
    to this exact string.
    """

    url: HttpUrl
    final_url: HttpUrl
    title: str = Field(default="", max_length=500)
    text: str
    # sha256 of `text`. Lets a later run detect that a source changed under it,
    # and lets citation verification confirm it is checking the same bytes the
    # citation was produced against.
    content_hash: str
    content_type: str = ""
    status_code: int
    byte_length: int
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Redirect chain, recorded because each hop was independently validated
    # and because a redirect to a different domain affects corroboration
    # independence (ADR-010).
    redirect_chain: list[str] = Field(default_factory=list)

    @property
    def domain(self) -> str:
        return (self.final_url.host or "").lower().removeprefix("www.")

    @field_validator("content_hash")
    @classmethod
    def _hash_shape(cls, value: str) -> str:
        if len(value) != 64 or not all(c in "0123456789abcdef" for c in value):
            raise ValueError("content_hash must be a lowercase sha256 hex digest")
        return value

    def verify_hash(self) -> bool:
        """Whether `content_hash` still matches `text`."""
        return content_hash(self.text) == self.content_hash


class Chunk(BaseModel):
    """A retrievable span of a source, with offsets back into its text."""

    source_url: HttpUrl
    index: int = Field(ge=0)
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    # Nearest preceding heading, when the document had one. Carried because a
    # chunk read without its section is often ambiguous to a human reviewer.
    section: str | None = Field(default=None, max_length=300)

    @field_validator("end_char")
    @classmethod
    def _end_after_start(cls, value: int, info: object) -> int:
        # Pydantic v2 passes a ValidationInfo with the already-validated data.
        data = getattr(info, "data", {}) or {}
        start = data.get("start_char")
        if start is not None and value <= start:
            raise ValueError("end_char must be greater than start_char")
        return value

    @property
    def char_length(self) -> int:
        return self.end_char - self.start_char

    def verify_against(self, source_text: str) -> bool:
        """Whether this chunk's offsets still select its own text.

        The check that makes the module invariant testable rather than
        aspirational.
        """
        return source_text[self.start_char : self.end_char] == self.text


def content_hash(text: str) -> str:
    """sha256 of the canonical text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
