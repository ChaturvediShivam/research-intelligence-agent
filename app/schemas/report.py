"""The final research report.

This module adds the layer that makes the system's ignorance legible. Every
field here is **derived from verified pipeline output** — there is no model
call in stage 8, and nothing in this module can turn a missing fact into an
inferred one.

Three rules shape the schema:

1. **UNKNOWN is a status, not an absence.** A sub-question nobody could
   answer gets an explicit record saying so, with the reason and what would
   resolve it. Silence would read as "nothing to report".
2. **Status is driven by evidence, never by prose.** A sub-question is not
   ANSWERED because the model wrote fluently about it; it is ANSWERED because
   a claim survived citation verification at usable confidence.
3. **Confidence reuses `app.schemas.evidence.Confidence`.** No second scoring
   model, no numbers. The verifier is the authority on confidence and this
   layer only reports what it derived.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from app.schemas.evidence import Claim, Confidence, CredibilityTier


class SubQuestionStatus(StrEnum):
    """How well the run answered one sub-question.

    PARTIAL exists because "answered, but only by a single unvetted source"
    and "answered by corroborated primary sources" are different findings, and
    collapsing them would be the exact confidence inflation this milestone
    exists to prevent.
    """

    ANSWERED = "answered"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class GapCause(StrEnum):
    """Why a question could not be answered.

    Drawn from the failure vocabulary the pipeline already produces, so every
    value here can be traced to a real pipeline outcome rather than guessed.
    """

    NO_SOURCES_DISCOVERED = "no_sources_discovered"
    SOURCE_FETCH_FAILED = "source_fetch_failed"
    ACCESS_RESTRICTED = "access_restricted"
    SOURCE_BLOCKED = "source_blocked"
    NOTHING_RETRIEVED = "nothing_retrieved"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CITATIONS_REJECTED = "citations_rejected"
    WEAK_SOURCE_QUALITY = "weak_source_quality"
    EXTRACTION_FAILED = "extraction_failed"


class InformationGap(BaseModel):
    """Something the run could not establish, and why.

    `what_would_resolve` comes from the plan's own `expected_source_types` and
    `answerable_if`, both written before any search ran. It is therefore a
    restatement of the plan's criteria, not a prediction — the schema will not
    carry a claim that a named source *would* settle the question, because
    nothing in the run's output could support that.
    """

    sub_question_id: str
    question: str
    cause: GapCause
    # Stated in terms of counts the pipeline measured, so it can be checked.
    why_insufficient: str
    # Phrased as what the plan said would answer it. Empty when the plan gave
    # no criterion to restate.
    what_would_resolve: str = ""
    # Measured context, so a reader can see how far the run got.
    sources_considered: int = 0
    chunks_retrieved: int = 0
    evidence_items: int = 0
    verified_citations: int = 0


class SubQuestionAssessment(BaseModel):
    """One sub-question's outcome, with its supporting evidence."""

    sub_question_id: str
    question: str
    rank: int
    status: SubQuestionStatus
    confidence: Confidence
    # The supported claims, in full, so provenance reaches the report.
    supporting_claims: list[Claim] = Field(default_factory=list)
    # Claims that did not survive verification, kept rather than dropped: a
    # removed claim is invisible, a marked one is auditable.
    unknown_claims: list[Claim] = Field(default_factory=list)
    verified_evidence_count: int = 0
    verified_citation_count: int = 0
    rejected_citation_count: int = 0
    # Set only when status is UNKNOWN or PARTIAL.
    unknown_reason: str = ""
    information_gap: InformationGap | None = None

    @property
    def answer(self) -> str:
        """The supported claims as prose, or an explicit statement of absence.

        Never an inference. When nothing was supported, this says so rather
        than returning an empty string a caller might render as a blank
        answer.
        """
        if not self.supporting_claims:
            return "UNKNOWN — the evidence retrieved by this run does not support an answer."
        return " ".join(claim.text.strip() for claim in self.supporting_claims)


class SourceCoverage(BaseModel):
    """What the run actually read, and what it could not."""

    discovered: int = 0
    fetched: int = 0
    failed: int = 0
    # Fetched sources by credibility tier, so weak coverage is visible rather
    # than averaged away.
    by_credibility: dict[str, int] = Field(default_factory=dict)
    failed_sources: list[dict[str, str]] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)

    @property
    def has_primary_source(self) -> bool:
        return self.by_credibility.get(CredibilityTier.PRIMARY.value, 0) > 0


class ResearchReport(BaseModel):
    """The final deliverable.

    Structured enough to be machine-readable, and explicit about its own
    limits. Field order follows how a researcher reads it: what was asked,
    what is known, what is not, and why.
    """

    run_id: str
    question: str
    restated_question: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # What the run established.
    executive_summary: str
    sub_questions: list[SubQuestionAssessment] = Field(default_factory=list)

    # What it did not.
    unknowns: list[str] = Field(default_factory=list)
    information_gaps: list[InformationGap] = Field(default_factory=list)

    # How it got there.
    source_coverage: SourceCoverage = Field(default_factory=SourceCoverage)
    limitations: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)

    # Measured totals, carried so the report is self-contained evidence.
    total_claims: int = 0
    supported_claims: int = 0
    unknown_claims: int = 0
    verified_citations: int = 0
    rejected_citations: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0

    @property
    def answered(self) -> list[SubQuestionAssessment]:
        return [s for s in self.sub_questions if s.status is SubQuestionStatus.ANSWERED]

    @property
    def partial(self) -> list[SubQuestionAssessment]:
        return [s for s in self.sub_questions if s.status is SubQuestionStatus.PARTIAL]

    @property
    def unanswered(self) -> list[SubQuestionAssessment]:
        return [s for s in self.sub_questions if s.status is SubQuestionStatus.UNKNOWN]

    @property
    def coverage_ratio(self) -> float:
        """Share of sub-questions with any supported evidence."""
        if not self.sub_questions:
            return 0.0
        return (len(self.answered) + len(self.partial)) / len(self.sub_questions)
