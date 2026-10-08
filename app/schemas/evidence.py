"""Evidence, citations and claims.

The chain this module defines is the product's central guarantee:

    FetchedSource.text  ──slice──►  Quote  ──cited by──►  Claim

A `Quote` is the only verifiable unit. It carries a `source_id`, character
offsets, and the text those offsets select. Verification re-slices the stored
source and compares — so a quote is either provably present in a named source
or it is rejected. Nothing in this module trusts a model's assertion that a
citation is real.

`Claim.confidence` is **derived**, never asserted by a model: it falls out of
corroboration × credibility × verification outcome (ADR-010). It is ordinal
rather than a percentage, because a number like "73% confident" is precision
the system does not have.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl, field_validator

from app.schemas.research import SourceType


class EvidenceGrade(StrEnum):
    """Talk < Behavior < Money.

    The ordering is the product's core epistemic claim: what someone *said* is
    weaker than what they *did*, which is weaker than what they *paid*. Only
    MONEY settles a willingness-to-pay question.
    """

    TALK = "talk"
    BEHAVIOR = "behavior"
    MONEY = "money"

    @property
    def rank(self) -> int:
        return {"talk": 1, "behavior": 2, "money": 3}[self.value]


class CredibilityTier(StrEnum):
    """Source provenance tier, assigned by rule rather than by model.

    A heuristic about provenance, not a fact-check: a regulatory filing can
    still be wrong, and a forum post can still be right.
    """

    PRIMARY = "primary"
    ESTABLISHED_SECONDARY = "established_secondary"
    UNVETTED = "unvetted"

    @property
    def rank(self) -> int:
        return {"primary": 3, "established_secondary": 2, "unvetted": 1}[self.value]


class Confidence(StrEnum):
    """Derived confidence. UNKNOWN is a real answer, not a failure."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNKNOWN = "unknown"


class FailureCode(StrEnum):
    """Why a citation or claim failed verification.

    Codes rather than messages so tests and the eval can assert on the
    specific failure mode, and so a fabricated citation is distinguishable
    from a merely misaligned one.
    """

    UNKNOWN_SOURCE = "unknown_source"
    OFFSET_OUT_OF_RANGE = "offset_out_of_range"
    OFFSET_INVALID = "offset_invalid"
    QUOTE_MISMATCH = "quote_mismatch"
    WRONG_SOURCE = "wrong_source"
    UNGROUNDED_FIGURE = "ungrounded_figure"
    NO_CITATION = "no_citation"
    CONTENT_CHANGED = "content_changed"


def source_id_for(url: str) -> str:
    """Stable, canonical id for a source, derived from its final URL.

    Deterministic so the same source gets the same id across runs and
    processes, which is what lets a citation be checked against a source
    fetched earlier.
    """
    return "src_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]


class SourceRef(BaseModel):
    """Canonical identity and provenance of a source.

    Carries `content_hash` so verification can detect that the stored text has
    changed since the citation was produced — a citation is only meaningful
    against the bytes it was made from.
    """

    source_id: str
    url: HttpUrl
    title: str = Field(default="", max_length=500)
    domain: str
    content_hash: str
    credibility: CredibilityTier
    source_type: SourceType | None = None
    # Publisher, where it can be determined. Used for independence: two
    # distinct domains owned by one publisher are not independent sources.
    publisher: str | None = Field(default=None, max_length=200)

    @classmethod
    def from_fetched(
        cls,
        *,
        url: str,
        title: str,
        domain: str,
        content_hash: str,
        credibility: CredibilityTier,
        source_type: SourceType | None = None,
        publisher: str | None = None,
    ) -> SourceRef:
        return cls(
            source_id=source_id_for(url),
            url=url,
            title=title[:500],
            domain=domain,
            content_hash=content_hash,
            credibility=credibility,
            source_type=source_type,
            publisher=publisher,
        )


class Quote(BaseModel):
    """A span of a source, identified by offsets and carrying its own text.

    The invariant verification enforces:

        source.text[start_char:end_char] == text   (after normalisation)
    """

    source_id: str
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    text: str = Field(min_length=1)

    @field_validator("end_char")
    @classmethod
    def _end_after_start(cls, value: int, info: object) -> int:
        data = getattr(info, "data", {}) or {}
        start = data.get("start_char")
        if start is not None and value <= start:
            raise ValueError("end_char must be greater than start_char")
        return value

    @property
    def length(self) -> int:
        return self.end_char - self.start_char


class EvidenceItem(BaseModel):
    """One piece of evidence extracted from one chunk.

    `statement` is what the evidence establishes, in the extractor's words.
    `quote` is the verbatim support. The two are kept separate so the
    statement can be read while the quote remains checkable — a paraphrase
    that has swallowed its own source is unverifiable.
    """

    evidence_id: str
    sub_question_id: str
    statement: str = Field(min_length=4, max_length=1000)
    quote: Quote
    grade: EvidenceGrade
    # Provenance back to the retrieved chunk that produced this item.
    chunk_index: int = Field(ge=0)

    @property
    def source_id(self) -> str:
        return self.quote.source_id


class Citation(BaseModel):
    """A claim's pointer at a span of a source.

    Produced either by the synthesis model's native citations or by linking an
    `EvidenceItem`. Either way it is verified the same way: by re-slicing the
    stored source.
    """

    source_id: str
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    cited_text: str = Field(min_length=1)
    evidence_id: str | None = None

    def as_quote(self) -> Quote:
        return Quote(
            source_id=self.source_id,
            start_char=self.start_char,
            end_char=self.end_char,
            text=self.cited_text,
        )


class ClaimStatus(StrEnum):
    """Whether a claim survived verification."""

    SUPPORTED = "supported"
    UNKNOWN = "unknown"


class Claim(BaseModel):
    """A factual assertion in the report, with its citations and verdict.

    A claim whose citations fail verification becomes UNKNOWN rather than
    being deleted: a removed claim is invisible, a marked one is auditable.
    """

    claim_id: str
    sub_question_id: str | None = None
    text: str = Field(min_length=4)
    citations: list[Citation] = Field(default_factory=list)

    # --- derived by the verifier; never supplied by a model ---------------
    status: ClaimStatus = ClaimStatus.UNKNOWN
    confidence: Confidence = Confidence.UNKNOWN
    # Count of independent supporting sources (distinct domain and publisher).
    corroboration: int = 0
    verified_citations: int = 0
    failures: list[FailureCode] = Field(default_factory=list)
    # Human-readable basis for the derived confidence, so a reader can argue
    # with the rule rather than with a number.
    confidence_basis: str = ""

    @property
    def is_supported(self) -> bool:
        return self.status is ClaimStatus.SUPPORTED
