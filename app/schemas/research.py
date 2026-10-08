"""Research request, plan, and run schemas.

These are the contract between the API and the pipeline. Stage 1 (PLAN)
produces a `ResearchPlan` via structured output, so this module's models are
also the JSON schema the model is constrained to.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class SourceType(StrEnum):
    """Kinds of source a sub-question can be answered from.

    Used by the planner to say what *would* answer a question, which then
    guides discovery. Ordered loosely by provenance strength — see
    `app.pipeline.validate` for how credibility is actually assigned.
    """

    REGULATORY_FILING = "regulatory_filing"
    OFFICIAL_STATISTICS = "official_statistics"
    COMPANY_PRIMARY = "company_primary"
    ACADEMIC = "academic"
    INDUSTRY_REPORT = "industry_report"
    NEWS = "news"
    EXPERT_COMMENTARY = "expert_commentary"
    COMMUNITY = "community"


class RunStatus(StrEnum):
    """Lifecycle of a research run."""

    PENDING = "pending"
    PLANNING = "planning"
    DISCOVERING = "discovering"
    PROCESSING = "processing"
    RETRIEVING = "retrieving"
    EXTRACTING = "extracting"
    SYNTHESISING = "synthesising"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {RunStatus.COMPLETED, RunStatus.FAILED}


class ResearchRequest(BaseModel):
    """What a caller submits.

    Validation is deliberately strict at this boundary: the question becomes
    part of a model prompt, and an unbounded string is both a cost and an
    injection concern.
    """

    question: Annotated[str, Field(min_length=12, max_length=2000)]
    # Optional analyst framing. Kept separate from `question` so the prompt can
    # treat it as caller-supplied context rather than as the question itself.
    context: Annotated[str | None, Field(default=None, max_length=4000)] = None
    max_sources: Annotated[int, Field(default=8, ge=1, le=25)] = 8

    @field_validator("question", "context")
    @classmethod
    def _reject_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank or whitespace only")
        return stripped

    @field_validator("question")
    @classmethod
    def _require_substance(cls, value: str) -> str:
        """A question needs enough words to decompose; one token cannot be planned."""
        if len(value.split()) < 3:
            raise ValueError("question must contain at least 3 words")
        return value


class SubQuestion(BaseModel):
    """One decomposed, independently answerable part of the research question.

    `rank` exists because the planner must commit to an order — the pipeline
    spends a bounded source budget, and which sub-question gets researched
    first is a real decision, not a presentation detail.
    """

    id: Annotated[str, Field(min_length=1, max_length=16)]
    question: Annotated[str, Field(min_length=8, max_length=500)]
    rationale: Annotated[str, Field(min_length=8, max_length=500)]
    rank: Annotated[int, Field(ge=1, le=20)]
    expected_source_types: Annotated[list[SourceType], Field(min_length=1, max_length=4)]
    # What would make this sub-question answered. Written before any search, so
    # it cannot be retrofitted to whatever the sources happened to say.
    answerable_if: Annotated[str, Field(min_length=8, max_length=500)]


class ResearchPlan(BaseModel):
    """Stage 1 output. This is the schema the model is constrained to.

    Field descriptions are load-bearing: they are what the model sees in the
    JSON schema, so they double as instruction.
    """

    restated_question: Annotated[
        str,
        Field(
            min_length=8,
            max_length=600,
            description=(
                "The research question restated precisely enough to be "
                "testable. Name the entity, scope, and time period explicitly."
            ),
        ),
    ]
    sub_questions: Annotated[
        list[SubQuestion],
        Field(
            min_length=1,
            max_length=8,
            description=(
                "Decomposition into independently answerable parts, ranked "
                "with the most decision-relevant first."
            ),
        ),
    ]
    out_of_scope: Annotated[
        list[str],
        Field(
            default_factory=list,
            max_length=6,
            description=(
                "Adjacent questions deliberately excluded, so the report's "
                "boundaries are explicit rather than accidental."
            ),
        ),
    ]
    assumptions: Annotated[
        list[str],
        Field(
            default_factory=list,
            max_length=6,
            description=(
                "Interpretation choices made while restating the question. "
                "Each one is a thing the caller could disagree with."
            ),
        ),
    ]

    @field_validator("sub_questions")
    @classmethod
    def _ranks_must_be_unique(cls, value: list[SubQuestion]) -> list[SubQuestion]:
        ranks = [sq.rank for sq in value]
        if len(set(ranks)) != len(ranks):
            raise ValueError("sub_question ranks must be unique")
        return value

    @field_validator("sub_questions")
    @classmethod
    def _ids_must_be_unique(cls, value: list[SubQuestion]) -> list[SubQuestion]:
        ids = [sq.id for sq in value]
        if len(set(ids)) != len(ids):
            raise ValueError("sub_question ids must be unique")
        return value

    def ordered(self) -> list[SubQuestion]:
        """Sub-questions in the planner's intended order of attack."""
        return sorted(self.sub_questions, key=lambda sq: sq.rank)


def new_run_id() -> str:
    """Opaque run identifier. Not sequential — run ids appear in URLs."""
    return f"run_{uuid.uuid4().hex[:16]}"


class ResearchRun(BaseModel):
    """A research run and whatever has been produced for it so far."""

    id: str = Field(default_factory=new_run_id)
    status: RunStatus = RunStatus.PENDING
    request: ResearchRequest
    plan: ResearchPlan | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
