"""The evaluation harness: golden set, the seven metrics, baseline, determinism.

Offline. The judge-backed metrics (6 and 7) are not exercised here — they need
the model the architecture specifies, and their values live in the committed
artifacts. What is tested here is everything that must be deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.golden import (
    GOLDEN_SET_VERSION,
    GoldenCase,
    GoldenSetError,
    load_golden_set,
)
from app.evaluation.judge import (
    ANSWER_RELEVANCE_RUBRIC,
    JUDGE_RUBRIC_VERSION,
    REPORT_QUALITY_RUBRIC,
    JudgeScore,
)
from app.evaluation.research_eval import (
    EVALUATION_VERSION,
    RETRIEVAL_K,
    EvaluationReport,
    compare,
    score_case,
)
from app.retrieval.embeddings import FastEmbedEmbedder
from tests.fixtures.eval_harness import doc_url_map, load_corpus, run_case

GOLDEN_DIR = Path("evals/datasets/golden_v1")
RESULTS = Path("evals/results")

EXPECTED_METRICS = [
    "retrieval_relevance",
    "source_coverage",
    "citation_correctness",
    "unsupported_claim_rate",
    "tool_selection",
    "answer_relevance",
    "report_quality",
]


@pytest.fixture(scope="module")
def corpus() -> dict:
    return load_corpus()


@pytest.fixture(scope="module")
def golden(corpus: dict):
    return load_golden_set(GOLDEN_DIR, corpus_doc_ids=set(corpus))


@pytest.fixture(scope="module")
def embedder() -> FastEmbedEmbedder:
    return FastEmbedEmbedder()


# ==========================================================================
# 1-2 · Golden set loading and schema validation
# ==========================================================================


class TestGoldenSetLoading:
    def test_loads_and_is_within_the_approved_size(self, golden) -> None:
        """Architecture §9 specifies 20-30 questions."""
        assert 20 <= len(golden.cases) <= 30
        assert golden.version == GOLDEN_SET_VERSION

    def test_has_a_holdout_split(self, golden) -> None:
        """§9: a held-out split guards against overfitting to the measured set."""
        assert golden.train
        assert golden.holdout
        assert len(golden.train) + len(golden.holdout) == len(golden.cases)

    def test_case_ids_are_unique(self, golden) -> None:
        ids = [c.case_id for c in golden.cases]
        assert len(set(ids)) == len(ids)

    def test_covers_the_documented_failure_modes(self, golden) -> None:
        conditions = {c.source_condition for c in golden.cases}
        assert {"healthy", "all_fetch_fail", "access_restricted", "one_fetch_fail"} <= conditions
        assert any(c.expect_unknown for c in golden.cases), "needs unanswerable cases"
        assert any(not c.expect_unknown for c in golden.cases)

    def test_every_judgment_names_an_available_document(self, golden) -> None:
        for case in golden.cases:
            assert set(case.judgments) <= set(case.available_docs), case.case_id

    def test_available_docs_fit_inside_the_discovery_cap(self, golden) -> None:
        """F-012: a case larger than the cap measures naming, not retrieval."""
        for case in golden.cases:
            assert len(case.available_docs) <= 8, case.case_id


class TestGoldenSetValidation:
    def _write(self, tmp_path: Path, case: dict) -> Path:
        (tmp_path / "cases.jsonl").write_text(json.dumps(case) + "\n")
        return tmp_path

    def _base(self, **overrides: object) -> dict:
        case = {
            "case_id": "X1",
            "split": "train",
            "question": "A question of sufficient length?",
            "intent": "testing validation",
            "available_docs": ["doc-a"],
            "judgments": {"doc-a": 2},
            "expect_unknown": False,
            "expected_tools": ["search_sources"],
            "source_condition": "healthy",
        }
        case.update(overrides)
        return case

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="missing"):
            load_golden_set(tmp_path)

    def test_judging_an_unavailable_document_is_rejected(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, self._base(judgments={"not-available": 2}))
        with pytest.raises(GoldenSetError, match="does not make available"):
            load_golden_set(path)

    def test_document_absent_from_the_corpus_is_rejected(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, self._base())
        with pytest.raises(GoldenSetError, match="absent from the corpus"):
            load_golden_set(path, corpus_doc_ids={"other"})

    def test_healthy_case_cannot_expect_unknown_while_judging_relevance(
        self, tmp_path: Path
    ) -> None:
        path = self._write(tmp_path, self._base(expect_unknown=True))
        with pytest.raises(GoldenSetError, match="cannot expect UNKNOWN"):
            load_golden_set(path)

    def test_unreadable_sources_must_expect_unknown(self, tmp_path: Path) -> None:
        """An answer is impossible when no source can be read."""
        path = self._write(
            tmp_path, self._base(source_condition="all_fetch_fail", expect_unknown=False)
        )
        with pytest.raises(GoldenSetError, match="does not expect UNKNOWN"):
            load_golden_set(path)

    def test_a_fetch_failure_case_may_judge_relevance_and_expect_unknown(
        self, tmp_path: Path
    ) -> None:
        """F-012b: the relevant source existing but being unreachable is the case."""
        path = self._write(
            tmp_path, self._base(source_condition="access_restricted", expect_unknown=True)
        )
        loaded = load_golden_set(path)
        assert loaded.cases[0].relevant_docs == {"doc-a"}

    def test_duplicate_case_ids_are_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "cases.jsonl").write_text(
            json.dumps(self._base()) + "\n" + json.dumps(self._base()) + "\n"
        )
        with pytest.raises(GoldenSetError, match="duplicate"):
            load_golden_set(tmp_path)

    def test_malformed_line_is_rejected_with_its_line_number(self, tmp_path: Path) -> None:
        (tmp_path / "cases.jsonl").write_text("{not json}\n")
        with pytest.raises(GoldenSetError, match=":1 is not a valid case"):
            load_golden_set(tmp_path)

    def test_out_of_scale_grade_is_rejected(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, self._base(judgments={"doc-a": 5}))
        with pytest.raises(GoldenSetError, match="must be 0, 1 or 2"):
            load_golden_set(path)

    def test_empty_file_is_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "cases.jsonl").write_text("\n")
        with pytest.raises(GoldenSetError, match="no cases"):
            load_golden_set(tmp_path)


# ==========================================================================
# 12 · The golden set cannot be mutated
# ==========================================================================


class TestGoldenSetIsImmutable:
    def test_a_case_cannot_be_edited(self, golden) -> None:
        """Stops an evaluator quietly editing an expected outcome to match."""
        from pydantic import ValidationError

        case = golden.cases[0]
        with pytest.raises(ValidationError):
            case.expect_unknown = True  # type: ignore[misc]

    def test_the_set_cannot_be_edited(self, golden) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            golden.cases = ()  # type: ignore[misc]

    def test_cases_are_a_tuple_not_a_list(self, golden) -> None:
        assert isinstance(golden.cases, tuple)


# ==========================================================================
# 3-5 · The seven metrics
# ==========================================================================


class TestSevenMetrics:
    def test_exactly_the_seven_approved_metrics_are_reported(self) -> None:
        """Architecture §9. Not six, not eight, and not renamed."""
        report = EvaluationReport(
            evaluation_version=EVALUATION_VERSION,
            golden_set_version="golden_v1",
            git_sha="abc",
            retrieval_k=RETRIEVAL_K,
        )
        assert [m.name for m in report.metrics()] == EXPECTED_METRICS

    def test_determinism_flags_match_the_architecture(self) -> None:
        """Five deterministic, two judged."""
        report = EvaluationReport(
            evaluation_version=EVALUATION_VERSION,
            golden_set_version="golden_v1",
            git_sha="abc",
            retrieval_k=RETRIEVAL_K,
        )
        flags = {m.name: m.deterministic for m in report.metrics()}
        assert sum(flags.values()) == 5
        assert flags["answer_relevance"] is False
        assert flags["report_quality"] is False

    def test_empty_report_reports_none_not_zero(self) -> None:
        """An unmeasured metric must not read as a score of 0."""
        report = EvaluationReport(
            evaluation_version=EVALUATION_VERSION,
            golden_set_version="golden_v1",
            git_sha="abc",
            retrieval_k=RETRIEVAL_K,
        )
        assert all(m.value is None for m in report.metrics())

    async def test_scoring_a_real_case_populates_metrics_1_to_5(
        self, golden, corpus, embedder
    ) -> None:
        case = golden.by_id("G15")
        result = await run_case(case, corpus, embedder)
        scored = score_case(case, result, doc_url_map(corpus))

        assert scored.recall_at_k is not None
        assert scored.source_coverage is not None
        assert scored.citations_total > 0
        assert scored.citation_correctness is not None
        assert scored.claims_total > 0
        assert scored.tool_sequence_correct is True
        # Metrics 6-7 are only populated by the judge.
        assert scored.answer_relevance is None
        assert scored.report_quality is None

    async def test_citation_correctness_is_recomputed_not_read_back(
        self, golden, corpus, embedder
    ) -> None:
        """Metric 3 must not trust Claim.verified_citations.

        Reading the pipeline's own field would make the metric a tautology, so
        the evaluator re-slices the stored source itself. Corrupting the field
        must not change the metric.
        """
        case = golden.by_id("G15")
        result = await run_case(case, corpus, embedder)
        assert result.claim_validation is not None

        honest = score_case(case, result, doc_url_map(corpus))
        assert honest.citation_correctness == 1.0

        # Lie about verification; the metric must be unmoved.
        for claim in result.claim_validation.claims:
            object.__setattr__(claim, "verified_citations", 0)
        for assessment in result.report.sub_questions if result.report else []:
            for claim in assessment.supporting_claims:
                object.__setattr__(claim, "verified_citations", 0)

        after = score_case(case, result, doc_url_map(corpus))
        assert after.citation_correctness == honest.citation_correctness

    async def test_an_unanswerable_case_does_not_score_retrieval_as_zero(
        self, golden, corpus, embedder
    ) -> None:
        """A case with nothing relevant is not a retrieval failure."""
        case = golden.by_id("G19")
        result = await run_case(case, corpus, embedder)
        scored = score_case(case, result, doc_url_map(corpus))
        assert scored.recall_at_k is None
        assert any("not scored" in note for note in scored.notes)

    def test_unsupported_claim_rate_direction_is_lower_is_better(self) -> None:
        baseline = EvaluationReport(
            evaluation_version="v", golden_set_version="g", git_sha="a", retrieval_k=5
        )
        current = EvaluationReport(
            evaluation_version="v", golden_set_version="g", git_sha="b", retrieval_k=5
        )
        from app.evaluation.research_eval import CaseResult

        baseline.cases.append(
            CaseResult(
                case_id="A", split="train", question="q", claims_total=4, claims_unsupported=2
            )
        )
        current.cases.append(
            CaseResult(
                case_id="A", split="train", question="q", claims_total=4, claims_unsupported=1
            )
        )
        rows = {name: verdict for name, _, _, verdict in compare(baseline, current)}
        # Fewer unsupported claims is an improvement, not a regression.
        assert rows["unsupported_claim_rate"] == "improved"


# ==========================================================================
# 6-7 · Baseline generation and reproducibility
# ==========================================================================


class TestBaselineArtifact:
    def test_committed_baseline_exists_and_names_its_inputs(self) -> None:
        candidates = sorted(RESULTS.glob("evaluation_baseline_*.json"))
        assert candidates, "no committed baseline artifact"
        payload = json.loads(candidates[-1].read_text())
        assert payload["golden_set_version"] == GOLDEN_SET_VERSION
        assert payload["evaluation_version"] == EVALUATION_VERSION
        assert payload["git_sha"]
        assert payload["retrieval_k"] == RETRIEVAL_K
        assert payload["reproducibility"]["command"]

    def test_committed_baseline_reports_all_seven_metrics(self) -> None:
        candidates = sorted(RESULTS.glob("evaluation_baseline_*.json"))
        payload = json.loads(candidates[-1].read_text())
        assert sorted(payload["metrics"]) == sorted(EXPECTED_METRICS)

    def test_baseline_records_which_metrics_are_nondeterministic(self) -> None:
        candidates = sorted(RESULTS.glob("evaluation_baseline_*.json"))
        payload = json.loads(candidates[-1].read_text())
        nondet = payload["reproducibility"]["nondeterministic_metrics"]
        assert set(nondet) == {"answer_relevance", "report_quality"}


class TestDeterminism:
    async def test_scoring_the_same_run_twice_is_identical(self, golden, corpus, embedder) -> None:
        case = golden.by_id("G16")
        result = await run_case(case, corpus, embedder)
        urls = doc_url_map(corpus)
        first = score_case(case, result, urls)
        second = score_case(case, result, urls)
        assert first == second

    async def test_running_the_same_case_twice_gives_the_same_metrics(
        self, golden, corpus, embedder
    ) -> None:
        """The deterministic metrics must reproduce across pipeline runs."""
        case = golden.by_id("G15")
        urls = doc_url_map(corpus)
        a = score_case(case, await run_case(case, corpus, embedder), urls)
        b = score_case(case, await run_case(case, corpus, embedder), urls)
        assert (a.recall_at_k, a.mrr, a.ndcg_at_k) == (b.recall_at_k, b.mrr, b.ndcg_at_k)
        assert a.source_coverage == b.source_coverage
        assert a.citation_correctness == b.citation_correctness
        assert a.unsupported_claim_rate == b.unsupported_claim_rate
        assert a.tool_sequence_correct == b.tool_sequence_correct


# ==========================================================================
# 8-9 · Comparison and regression detection
# ==========================================================================


class TestComparison:
    def _report(self, label: str, **values: float) -> EvaluationReport:
        from app.evaluation.research_eval import CaseResult

        report = EvaluationReport(
            evaluation_version="v",
            golden_set_version="g",
            git_sha="sha",
            retrieval_k=5,
            label=label,
        )
        report.cases.append(
            CaseResult(
                case_id="A",
                split="train",
                question="q",
                recall_at_k=values.get("recall"),
                mrr=values.get("recall"),
                ndcg_at_k=values.get("recall"),
            )
        )
        return report

    def test_detects_improvement(self) -> None:
        rows = {
            name: verdict
            for name, _, _, verdict in compare(
                self._report("base", recall=0.5), self._report("new", recall=0.8)
            )
        }
        assert rows["retrieval_relevance"] == "improved"

    def test_detects_regression(self) -> None:
        rows = {
            name: verdict
            for name, _, _, verdict in compare(
                self._report("base", recall=0.8), self._report("new", recall=0.5)
            )
        }
        assert rows["retrieval_relevance"] == "regressed"

    def test_detects_unchanged(self) -> None:
        rows = {
            name: verdict
            for name, _, _, verdict in compare(
                self._report("base", recall=0.8), self._report("new", recall=0.8)
            )
        }
        assert rows["retrieval_relevance"] == "unchanged"

    def test_reports_every_metric_not_only_the_changed_one(self) -> None:
        """Cherry-picking one metric is the failure mode this prevents."""
        rows = compare(self._report("base", recall=0.5), self._report("new", recall=0.8))
        assert [name for name, _, _, _ in rows] == EXPECTED_METRICS

    def test_unmeasured_metrics_are_labelled_not_scored(self) -> None:
        rows = {
            name: verdict
            for name, _, _, verdict in compare(
                self._report("base", recall=0.5), self._report("new", recall=0.8)
            )
        }
        assert rows["answer_relevance"] == "not measured"


# ==========================================================================
# The judge's rubric is fixed, not improvised
# ==========================================================================


class TestJudgeRubric:
    def test_rubric_is_versioned(self) -> None:
        assert JUDGE_RUBRIC_VERSION == "rubric_v1"

    def test_rubrics_state_a_bounded_scale(self) -> None:
        for rubric in (ANSWER_RELEVANCE_RUBRIC, REPORT_QUALITY_RUBRIC):
            assert "0.0" in rubric and "1.0" in rubric

    def test_answer_rubric_credits_an_honest_refusal(self) -> None:
        """Declining to answer an unanswerable question must not be punished."""
        assert "Declining to answer is" in ANSWER_RELEVANCE_RUBRIC

    def test_report_rubric_penalises_unsupported_confidence(self) -> None:
        assert "confident" in REPORT_QUALITY_RUBRIC

    def test_score_is_bounded(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            JudgeScore(score=1.5, reason="out of range entirely")
        with pytest.raises(ValidationError):
            JudgeScore(score=-0.1, reason="out of range entirely")
        assert JudgeScore(score=0.5, reason="a valid reason string").score == 0.5


# ==========================================================================
# 13 · M0-M6 behaviour preserved
# ==========================================================================


class TestNoRegressionInGuarantees:
    async def test_citation_verification_still_rejects_nothing_valid(
        self, golden, corpus, embedder
    ) -> None:
        """Every citation produced across the golden set must verify."""
        urls = doc_url_map(corpus)
        total = correct = 0
        for case_id in ("G02", "G15", "G16"):
            case = golden.by_id(case_id)
            scored = score_case(case, await run_case(case, corpus, embedder), urls)
            total += scored.citations_total
            correct += scored.citations_correct
        assert total > 0
        assert correct == total

    async def test_unknown_semantics_survive(self, golden, corpus, embedder) -> None:
        """A case whose sources all fail must produce no supported claim."""
        case = golden.by_id("G12")
        result = await run_case(case, corpus, embedder)
        assert result.report is not None
        assert result.report.supported_claims == 0
        assert result.report.information_gaps

    async def test_provenance_survives_into_scored_output(self, golden, corpus, embedder) -> None:
        case = golden.by_id("G15")
        result = await run_case(case, corpus, embedder)
        assert result.report is not None
        for assessment in result.report.sub_questions:
            for claim in assessment.supporting_claims:
                for citation in claim.citations:
                    stored = result.source_texts[citation.source_id]
                    assert stored[citation.start_char : citation.end_char] == citation.cited_text

    def test_golden_case_type_is_frozen(self) -> None:
        assert GoldenCase.model_config.get("frozen") is True
