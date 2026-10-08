"""Request validation and plan invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.research import (
    ResearchPlan,
    ResearchRequest,
    RunStatus,
    SourceType,
    SubQuestion,
    new_run_id,
)


def _sub(sid: str, rank: int) -> SubQuestion:
    return SubQuestion(
        id=sid,
        question="What was the reported revenue for the period?",
        rationale="Revenue anchors every downstream comparison.",
        rank=rank,
        expected_source_types=[SourceType.REGULATORY_FILING],
        answerable_if="A filing states a revenue figure for the stated period.",
    )


class TestResearchRequest:
    def test_accepts_a_substantive_question(self) -> None:
        r = ResearchRequest(question="How large is the UK pet insurance market?")
        assert r.max_sources == 8

    @pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
    def test_rejects_blank(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            ResearchRequest(question=bad)

    def test_rejects_too_short(self) -> None:
        with pytest.raises(ValidationError):
            ResearchRequest(question="why?")

    def test_rejects_fewer_than_three_words(self) -> None:
        """A single long token cannot be decomposed into sub-questions."""
        with pytest.raises(ValidationError, match="at least 3 words"):
            ResearchRequest(question="aaaaaaaaaaaaaaaaaaaa")

    def test_rejects_unbounded_question(self) -> None:
        """An unbounded prompt input is both a cost and an injection concern."""
        with pytest.raises(ValidationError):
            ResearchRequest(question="x " * 2000)

    def test_strips_surrounding_whitespace(self) -> None:
        r = ResearchRequest(question="  How big is the market for X?  ")
        assert r.question == "How big is the market for X?"

    @pytest.mark.parametrize("bad", [0, -1, 26, 100])
    def test_max_sources_bounded(self, bad: int) -> None:
        with pytest.raises(ValidationError):
            ResearchRequest(question="How large is this market?", max_sources=bad)

    def test_context_is_optional_and_stripped(self) -> None:
        r = ResearchRequest(question="How large is this market?", context="  note  ")
        assert r.context == "note"
        assert ResearchRequest(question="How large is this market?").context is None


class TestResearchPlan:
    def test_valid_plan(self) -> None:
        plan = ResearchPlan(
            restated_question="UK pet insurance gross written premium, 2024-2026.",
            sub_questions=[_sub("SQ1", 1), _sub("SQ2", 2)],
        )
        assert len(plan.sub_questions) == 2
        assert plan.out_of_scope == []

    def test_duplicate_ranks_rejected(self) -> None:
        """Rank is an allocation decision; a tie makes it meaningless."""
        with pytest.raises(ValidationError, match="ranks must be unique"):
            ResearchPlan(
                restated_question="A precise restatement of the question.",
                sub_questions=[_sub("SQ1", 1), _sub("SQ2", 1)],
            )

    def test_duplicate_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match="ids must be unique"):
            ResearchPlan(
                restated_question="A precise restatement of the question.",
                sub_questions=[_sub("SQ1", 1), _sub("SQ1", 2)],
            )

    def test_empty_decomposition_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResearchPlan(
                restated_question="A precise restatement of the question.",
                sub_questions=[],
            )

    def test_ordered_sorts_by_rank(self) -> None:
        plan = ResearchPlan(
            restated_question="A precise restatement of the question.",
            sub_questions=[_sub("SQ3", 3), _sub("SQ1", 1), _sub("SQ2", 2)],
        )
        assert [sq.id for sq in plan.ordered()] == ["SQ1", "SQ2", "SQ3"]

    def test_sub_question_requires_at_least_one_source_type(self) -> None:
        with pytest.raises(ValidationError):
            SubQuestion(
                id="SQ1",
                question="What was the reported revenue?",
                rationale="Anchors the comparison.",
                rank=1,
                expected_source_types=[],
                answerable_if="A filing states the figure.",
            )


class TestRunIdentity:
    def test_run_ids_are_opaque_and_unique(self) -> None:
        ids = {new_run_id() for _ in range(200)}
        assert len(ids) == 200
        assert all(i.startswith("run_") for i in ids)


class TestRunStatus:
    def test_terminal_states(self) -> None:
        assert RunStatus.COMPLETED.is_terminal
        assert RunStatus.FAILED.is_terminal

    @pytest.mark.parametrize(
        "status", [RunStatus.PENDING, RunStatus.PLANNING, RunStatus.RETRIEVING]
    )
    def test_non_terminal_states(self, status: RunStatus) -> None:
        assert not status.is_terminal
