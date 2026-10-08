"""Retrieval metrics, checked against hand-computed values.

Every expected number here is worked out by hand in the test, so the tests
verify the implementation rather than echo it.
"""

from __future__ import annotations

import math

import pytest

from app.evaluation.metrics import (
    dcg_at_k,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Two relevant documents: d1 (grade 2) and d3 (grade 1).
JUDGMENTS = {"d1": 2, "d3": 1}


class TestRecallAtK:
    def test_finds_one_of_two_relevant(self) -> None:
        assert recall_at_k(["d1", "d2"], JUDGMENTS, 2) == pytest.approx(0.5)

    def test_finds_both(self) -> None:
        assert recall_at_k(["d1", "d3"], JUDGMENTS, 2) == pytest.approx(1.0)

    def test_cutoff_excludes_a_later_relevant_hit(self) -> None:
        assert recall_at_k(["d2", "d1"], JUDGMENTS, 1) == pytest.approx(0.0)

    def test_no_relevant_documents_scores_zero_not_one(self) -> None:
        """A degenerate query must not inflate the dataset mean."""
        assert recall_at_k(["d1"], {}, 5) == 0.0
        assert recall_at_k(["d1"], {"d9": 0}, 5) == 0.0

    def test_duplicates_do_not_earn_double_credit(self) -> None:
        assert recall_at_k(["d1", "d1", "d1"], JUDGMENTS, 3) == pytest.approx(0.5)

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k(self, k: int) -> None:
        assert recall_at_k(["d1"], JUDGMENTS, k) == 0.0

    def test_grade_zero_is_not_relevant(self) -> None:
        assert recall_at_k(["d9"], {"d1": 2, "d9": 0}, 5) == pytest.approx(0.0)


class TestPrecisionAtK:
    def test_half_the_top_two_are_relevant(self) -> None:
        assert precision_at_k(["d1", "d2"], JUDGMENTS, 2) == pytest.approx(0.5)

    def test_all_relevant(self) -> None:
        assert precision_at_k(["d1", "d3"], JUDGMENTS, 2) == pytest.approx(1.0)

    def test_fewer_results_than_k_divides_by_results(self) -> None:
        assert precision_at_k(["d1"], JUDGMENTS, 5) == pytest.approx(1.0)

    def test_empty_retrieval(self) -> None:
        assert precision_at_k([], JUDGMENTS, 5) == 0.0


class TestReciprocalRank:
    @pytest.mark.parametrize(
        ("retrieved", "expected"),
        [
            (["d1", "d2", "d3"], 1.0),
            (["d2", "d1", "d3"], 0.5),
            (["d2", "d9", "d3"], 1 / 3),
            (["d2", "d9"], 0.0),
            ([], 0.0),
        ],
    )
    def test_rank_of_first_relevant(self, retrieved: list[str], expected: float) -> None:
        assert reciprocal_rank(retrieved, JUDGMENTS) == pytest.approx(expected)

    def test_duplicates_before_the_hit_do_not_shift_the_rank(self) -> None:
        assert reciprocal_rank(["d2", "d2", "d1"], JUDGMENTS) == pytest.approx(0.5)


class TestDCG:
    def test_single_grade_two_at_rank_one(self) -> None:
        # gain = 2**2 - 1 = 3, discount = log2(2) = 1  ->  3.0
        assert dcg_at_k(["d1"], JUDGMENTS, 1) == pytest.approx(3.0)

    def test_single_grade_one_at_rank_one(self) -> None:
        # gain = 2**1 - 1 = 1, discount = 1  ->  1.0
        assert dcg_at_k(["d3"], JUDGMENTS, 1) == pytest.approx(1.0)

    def test_exponential_gain_favours_the_higher_grade(self) -> None:
        """A grade-2 document is worth three times a grade-1 one, not twice."""
        assert dcg_at_k(["d1"], JUDGMENTS, 1) == pytest.approx(3 * dcg_at_k(["d3"], JUDGMENTS, 1))

    def test_discount_applies_by_position(self) -> None:
        # d3 at rank 2: 1 / log2(3)
        assert dcg_at_k(["d9", "d3"], JUDGMENTS, 2) == pytest.approx(1 / math.log2(3))

    def test_irrelevant_documents_contribute_nothing(self) -> None:
        assert dcg_at_k(["d9", "d8"], JUDGMENTS, 2) == 0.0


class TestNDCG:
    def test_ideal_ordering_scores_one(self) -> None:
        assert ndcg_at_k(["d1", "d3"], JUDGMENTS, 2) == pytest.approx(1.0)

    def test_inverted_ordering_scores_below_one(self) -> None:
        """Putting the grade-1 document first is penalised."""
        score = ndcg_at_k(["d3", "d1"], JUDGMENTS, 2)
        assert 0.0 < score < 1.0
        # DCG = 1 + 3/log2(3); ideal = 3 + 1/log2(3)
        expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
        assert score == pytest.approx(expected)

    def test_no_relevant_documents_scores_zero(self) -> None:
        assert ndcg_at_k(["d1"], {}, 5) == 0.0

    def test_bounded_by_one(self) -> None:
        for retrieved in (["d1"], ["d1", "d3"], ["d3", "d1"], ["d9", "d1", "d3"]):
            assert 0.0 <= ndcg_at_k(retrieved, JUDGMENTS, 10) <= 1.0

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k(self, k: int) -> None:
        assert ndcg_at_k(["d1"], JUDGMENTS, k) == 0.0


class TestMean:
    def test_arithmetic_mean(self) -> None:
        assert mean([1.0, 0.0, 0.5]) == pytest.approx(0.5)

    def test_empty_is_zero(self) -> None:
        assert mean([]) == 0.0
