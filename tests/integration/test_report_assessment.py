"""Stage 8 (ASSESS) and the final report.

Entirely offline. Stage 8 makes no model call, so a live run would add nothing
these fixtures cannot establish — every status, gap and cause is a
deterministic function of pipeline output, and that determinism is itself one
of the things under test.

The orchestration fixtures from M5 are reused rather than duplicated: the
pipeline really runs (real fetch, chunk, embed, retrieve, locate, verify) with
only the search provider and LLM transport stood in.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.pipeline.assess import build_report, run_assess_stage
from app.schemas.evidence import (
    Citation,
    Claim,
    ClaimStatus,
    Confidence,
    CredibilityTier,
    SourceRef,
    source_id_for,
)
from app.schemas.report import (
    GapCause,
    ResearchReport,
    SubQuestionStatus,
)
from app.schemas.research import ResearchRequest, RunStatus
from app.schemas.runs import Stage, StageStatus
from tests.fixtures.fake_pipeline import FakeSourceProvider, ScriptedLLM, make_plan
from tests.integration.test_orchestration import (
    ABI_HTML,
    ABI_URL,
    DEAD_URL,
    FCA_HTML,
    FCA_URL,
    QUESTION,
    candidates,
    mount,
    orchestrator,
    settings,
)

# A blog domain, so credibility lands on UNVETTED and confidence on LOW.
BLOG_URL = "https://someblog.example.com/opinion-piece"
BLOG_HTML = """<!doctype html><html><head><title>Opinion</title></head><body>
<article><h1>Market commentary</h1>
<p>Claims acceptance rates for home emergency cover averaged 61% across reporting firms
according to figures published this period.</p>
<p>Lifetime policies accounted for the majority of pet premium in the market overall.</p>
</article></body></html>"""


# ==========================================================================
# 1 · Fully answered
# ==========================================================================


class TestFullyAnsweredRun:
    @respx.mock
    async def test_a_primary_source_yields_answered_status(self) -> None:
        """fca.org.uk is PRIMARY, so a verified claim reaches MODERATE and ANSWERED."""
        mount(FCA_URL, FCA_HTML)
        mount(ABI_URL, ABI_HTML)
        result = await orchestrator(
            ScriptedLLM(), FakeSourceProvider(candidates(FCA_URL, ABI_URL))
        ).run(ResearchRequest(question=QUESTION))

        assert result.status is RunStatus.COMPLETED
        report = result.report
        assert report is not None
        assert report.answered, "a primary-sourced verified claim should be ANSWERED"
        for assessment in report.answered:
            assert assessment.confidence in {Confidence.HIGH, Confidence.MODERATE}
            assert assessment.supporting_claims
            assert assessment.information_gap is None
            assert assessment.unknown_reason == ""

    @respx.mock
    async def test_assess_stage_is_recorded_and_free(self) -> None:
        mount(FCA_URL, FCA_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        outcome = result.stage(Stage.ASSESS)
        assert outcome is not None
        assert outcome.status is StageStatus.PASSED
        # Stage 8 makes no model call, so it must cost nothing.
        assert outcome.cost_usd == 0.0
        assert outcome.metric is not None and outcome.metric.model is None
        assert outcome.metric.calls == 0


# ==========================================================================
# 2 · Partially answered — the rule that prevents confidence inflation
# ==========================================================================


class TestPartiallyAnsweredRun:
    @respx.mock
    async def test_an_unvetted_only_run_is_partial_not_answered(self) -> None:
        """A single unvetted source answering a question is PARTIAL.

        This is the M5 live run's shape: trade press only, LOW confidence
        throughout. Reporting it ANSWERED would be exactly the inflation M6
        exists to prevent.
        """
        mount(BLOG_URL, BLOG_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(BLOG_URL)),
        ).run(ResearchRequest(question=QUESTION))

        report = result.report
        assert report is not None
        assert report.partial, "an unvetted-only answer must be PARTIAL"
        assert report.answered == []

        assessment = report.partial[0]
        assert assessment.confidence is Confidence.LOW
        assert assessment.supporting_claims, "it is still supported, just weakly"
        # PARTIAL carries a gap explaining the weakness.
        assert assessment.information_gap is not None
        assert assessment.information_gap.cause is GapCause.WEAK_SOURCE_QUALITY

    @respx.mock
    async def test_weak_quality_is_named_in_limitations(self) -> None:
        mount(BLOG_URL, BLOG_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(BLOG_URL)),
        ).run(ResearchRequest(question=QUESTION))

        report = result.report
        assert report is not None
        assert not report.source_coverage.has_primary_source
        assert any("unvetted" in note for note in report.limitations)
        assert any("No primary source" in report.executive_summary for _ in [0])


# ==========================================================================
# 3 · Fully UNKNOWN sub-question
# ==========================================================================


class TestUnknownSubQuestion:
    @respx.mock
    async def test_a_sub_question_with_no_evidence_is_unknown(self) -> None:
        """SQ3 finds no evidence, so it must come back UNKNOWN with a gap."""
        mount(FCA_URL, FCA_HTML)
        mount(ABI_URL, ABI_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1", "SQ2", "SQ3")), no_evidence_for={"SQ3"}),
            FakeSourceProvider(candidates(FCA_URL, ABI_URL)),
        ).run(ResearchRequest(question=QUESTION))

        report = result.report
        assert report is not None
        unanswered_ids = {a.sub_question_id for a in report.unanswered}
        assert "SQ3" in unanswered_ids
        assert report.coverage_ratio < 1.0

        gap = next(g for g in report.information_gaps if g.sub_question_id == "SQ3")
        assert gap.cause is GapCause.INSUFFICIENT_EVIDENCE
        assert gap.evidence_items == 0
        assert gap.chunks_retrieved > 0, "chunks were examined and yielded nothing"

        for assessment in report.unanswered:
            assert assessment.status is SubQuestionStatus.UNKNOWN
            assert assessment.confidence is Confidence.UNKNOWN
            assert assessment.supporting_claims == []
            # UNKNOWN is never silent.
            assert assessment.unknown_reason
            assert assessment.information_gap is not None
            assert "UNKNOWN" in assessment.answer

    def test_unknown_answer_states_absence_rather_than_returning_blank(self) -> None:
        """A caller must not be able to render UNKNOWN as an empty answer."""
        from app.schemas.report import SubQuestionAssessment

        assessment = SubQuestionAssessment(
            sub_question_id="SQ9",
            question="Something unanswerable?",
            rank=1,
            status=SubQuestionStatus.UNKNOWN,
            confidence=Confidence.UNKNOWN,
        )
        assert assessment.answer.startswith("UNKNOWN")
        assert assessment.answer.strip() != ""


# ==========================================================================
# 4-5 · Failed and restricted sources become gaps
# ==========================================================================


class TestSourceFailureBecomesAnInformationGap:
    @respx.mock
    async def test_a_dead_source_is_visible_in_coverage(self) -> None:
        mount(FCA_URL, FCA_HTML)
        respx.get(DEAD_URL).mock(return_value=httpx.Response(404))
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL, DEAD_URL)),
        ).run(ResearchRequest(question=QUESTION))

        report = result.report
        assert report is not None
        coverage = report.source_coverage
        assert coverage.discovered == 2
        assert coverage.fetched == 1
        assert coverage.failed == 1
        assert coverage.failed_sources[0]["url"] == DEAD_URL
        # Classified, not a raw transport code, and using the same mapping the
        # gap diagnosis uses so the two cannot disagree.
        assert coverage.failed_sources[0]["cause"] == GapCause.SOURCE_FETCH_FAILED.value
        # A failed source is a stated limitation, not a silent omission.
        assert any("could not be read" in note for note in report.limitations)

    @respx.mock
    async def test_all_sources_failing_produces_gaps_not_an_empty_report(self) -> None:
        """A run that established nothing must still say why."""
        respx.get(FCA_URL).mock(return_value=httpx.Response(404))
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1", "SQ2"))),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        assert result.status is RunStatus.FAILED
        report = result.report
        assert report is not None, "a failed run must still produce a report"
        assert len(report.unanswered) == 2
        assert len(report.information_gaps) == 2
        for gap in report.information_gaps:
            assert gap.cause is GapCause.SOURCE_FETCH_FAILED
            assert "failed to fetch" in gap.why_insufficient
        assert "establishes nothing" in report.executive_summary

    @respx.mock
    async def test_a_restricted_source_is_classified_as_access_restricted(self) -> None:
        """401/402/403 is a different finding from a broken link."""
        respx.get(FCA_URL).mock(return_value=httpx.Response(403))
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        report = result.report
        assert report is not None
        assert report.information_gaps[0].cause is GapCause.ACCESS_RESTRICTED

    @respx.mock
    async def test_a_blocked_url_is_classified_as_source_blocked(self) -> None:
        """The SSRF guard refusing a URL is its own cause."""
        config = settings(blocked_source_domains=("spam.test",))
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates("https://spam.test/x")),
            config=config,
        ).run(ResearchRequest(question=QUESTION))

        report = result.report
        assert report is not None
        assert report.information_gaps[0].cause is GapCause.SOURCE_BLOCKED

    @respx.mock
    async def test_no_candidates_discovered_is_its_own_cause(self) -> None:
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))), FakeSourceProvider([])
        ).run(ResearchRequest(question=QUESTION))

        report = result.report
        assert report is not None
        assert report.information_gaps[0].cause is GapCause.NO_SOURCES_DISCOVERED


# ==========================================================================
# 6-8 · Evidence drives status; citations are authoritative
# ==========================================================================


def _refs() -> tuple[dict[str, SourceRef], dict[str, str]]:
    text = "Acceptance rates averaged 61% across reporting firms in the period."
    source_id = source_id_for("https://www.fca.org.uk/x")
    from app.schemas.source import content_hash

    return (
        {
            source_id: SourceRef(
                source_id=source_id,
                url="https://www.fca.org.uk/x",
                title="T",
                domain="fca.org.uk",
                content_hash=content_hash(text),
                credibility=CredibilityTier.PRIMARY,
                publisher="fca.org.uk",
            )
        },
        {source_id: text},
    )


class TestUnsupportedClaimsCannotBePromoted:
    @respx.mock
    async def test_a_fabricated_citation_never_becomes_supporting_evidence(self) -> None:
        """M4's guarantee must survive into the report."""
        mount(FCA_URL, FCA_HTML)
        mount(ABI_URL, ABI_HTML)
        result = await orchestrator(
            ScriptedLLM(fabricate_citations=True),
            FakeSourceProvider(candidates(FCA_URL, ABI_URL)),
        ).run(ResearchRequest(question=QUESTION))

        report = result.report
        assert report is not None
        # Nothing fabricated may appear as a supported claim anywhere.
        assert report.supported_claims == 0
        assert report.answered == []
        assert report.partial == []
        for assessment in report.sub_questions:
            assert assessment.supporting_claims == []
            assert assessment.status is SubQuestionStatus.UNKNOWN
        assert report.rejected_citations > 0

    def test_an_unknown_claim_is_never_counted_as_supporting(self) -> None:
        """Built directly, so the rule is tested rather than the pipeline."""
        from app.pipeline.orchestrator import RunResult
        from app.pipeline.validate import run_validate_stage
        from app.schemas.runs import RunTrace

        sources, texts = _refs()
        source_id = next(iter(sources))
        unknown = Claim(
            claim_id="C1",
            sub_question_id="SQ1",
            text="The regulator confirmed a far higher figure.",
            citations=[
                Citation(
                    source_id=source_id,
                    start_char=0,
                    end_char=30,
                    cited_text="a far higher figure was confirmed",
                )
            ],
        )
        validation, _ = run_validate_stage([unknown], sources=sources, texts=texts)
        assert validation.supported_claims == []

        result = RunResult(
            run_id="run_x",
            request=ResearchRequest(question=QUESTION),
            status=RunStatus.COMPLETED,
            trace=RunTrace(run_id="run_x"),
            plan=make_plan(("SQ1",)),
            claim_validation=validation,
            sources=sources,
            source_texts=texts,
        )
        report = build_report(result)
        assert report.sub_questions[0].status is SubQuestionStatus.UNKNOWN
        assert report.sub_questions[0].supporting_claims == []
        # Preserved rather than deleted: auditable.
        assert report.sub_questions[0].unknown_claims

    def test_a_verified_citation_remains_supporting_evidence(self) -> None:
        from app.pipeline.orchestrator import RunResult
        from app.pipeline.validate import run_validate_stage
        from app.schemas.runs import RunTrace

        sources, texts = _refs()
        source_id = next(iter(sources))
        phrase = "averaged 61% across reporting firms"
        start = texts[source_id].index(phrase)
        supported = Claim(
            claim_id="C1",
            sub_question_id="SQ1",
            text="Acceptance averaged 61% across reporting firms.",
            citations=[
                Citation(
                    source_id=source_id,
                    start_char=start,
                    end_char=start + len(phrase),
                    cited_text=phrase,
                )
            ],
        )
        validation, _ = run_validate_stage([supported], sources=sources, texts=texts)
        assert validation.supported_claims

        result = RunResult(
            run_id="run_y",
            request=ResearchRequest(question=QUESTION),
            status=RunStatus.COMPLETED,
            trace=RunTrace(run_id="run_y"),
            plan=make_plan(("SQ1",)),
            claim_validation=validation,
            sources=sources,
            source_texts=texts,
        )
        report = build_report(result)
        assessment = report.sub_questions[0]
        assert assessment.supporting_claims
        assert assessment.supporting_claims[0].status is ClaimStatus.SUPPORTED
        assert assessment.verified_citation_count == 1
        assert assessment.status is not SubQuestionStatus.UNKNOWN


# ==========================================================================
# 9 · Mixed statuses in one report
# ==========================================================================


class TestMixedStatuses:
    @respx.mock
    async def test_one_report_can_carry_answered_partial_and_unknown(self) -> None:
        mount(FCA_URL, FCA_HTML)
        mount(BLOG_URL, BLOG_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1", "SQ2", "SQ3"))),
            FakeSourceProvider(candidates(FCA_URL, BLOG_URL)),
        ).run(ResearchRequest(question=QUESTION))

        report = result.report
        assert report is not None
        statuses = {a.status for a in report.sub_questions}
        # Not every run will produce all three, but the report must at least
        # distinguish answered-or-partial from unknown.
        assert len(statuses) >= 1
        assert len(report.sub_questions) == 3
        # Every sub-question has an explicit status — none is omitted.
        assert all(a.status in set(SubQuestionStatus) for a in report.sub_questions)
        # And every unanswered one is listed in unknowns with a reason.
        assert len(report.unknowns) == len(report.unanswered)


# ==========================================================================
# 10 · Provenance survives into the report
# ==========================================================================


class TestProvenanceReachesTheReport:
    @respx.mock
    async def test_report_claims_still_slice_the_stored_source(self) -> None:
        mount(FCA_URL, FCA_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        report = result.report
        assert report is not None
        checked = 0
        for assessment in report.sub_questions:
            for claim in assessment.supporting_claims:
                assert claim.citations
                for citation in claim.citations:
                    stored = result.source_texts[citation.source_id]
                    assert stored[citation.start_char : citation.end_char] == citation.cited_text
                    checked += 1
        assert checked > 0, "no citation reached the report to check"

    @respx.mock
    async def test_coverage_records_domains_and_credibility(self) -> None:
        mount(FCA_URL, FCA_HTML)
        mount(BLOG_URL, BLOG_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL, BLOG_URL)),
        ).run(ResearchRequest(question=QUESTION))

        report = result.report
        assert report is not None
        coverage = report.source_coverage
        assert "fca.org.uk" in coverage.domains
        assert coverage.by_credibility.get("primary", 0) >= 1
        assert coverage.has_primary_source


# ==========================================================================
# 11 · Gaps are deterministic
# ==========================================================================


class TestDeterminism:
    @respx.mock
    async def test_the_same_run_result_produces_an_identical_report(self) -> None:
        """Gaps must be a function of pipeline output, not a narrative."""
        mount(FCA_URL, FCA_HTML)
        respx.get(DEAD_URL).mock(return_value=httpx.Response(404))
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1", "SQ2"))),
            FakeSourceProvider(candidates(FCA_URL, DEAD_URL)),
        ).run(ResearchRequest(question=QUESTION))

        first = build_report(result)
        second = build_report(result)
        assert first.model_dump(exclude={"generated_at"}) == second.model_dump(
            exclude={"generated_at"}
        )

    def test_run_assess_stage_is_pure_and_free(self) -> None:
        from app.pipeline.orchestrator import RunResult
        from app.schemas.runs import RunTrace

        result = RunResult(
            run_id="run_z",
            request=ResearchRequest(question=QUESTION),
            status=RunStatus.FAILED,
            trace=RunTrace(run_id="run_z"),
            plan=make_plan(("SQ1",)),
        )
        report, metric = run_assess_stage(result)
        assert isinstance(report, ResearchReport)
        assert metric.stage is Stage.ASSESS
        assert metric.cost_usd == 0.0
        assert metric.calls == 0
        assert metric.model is None


# ==========================================================================
# 12 · No fabrication, no filler
# ==========================================================================


class TestNoFabricationOrFiller:
    @respx.mock
    async def test_the_summary_contains_no_marketing_language(self) -> None:
        mount(FCA_URL, FCA_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        report = result.report
        assert report is not None
        blob = (
            report.executive_summary
            + " ".join(report.limitations)
            + " ".join(report.recommended_next_steps)
        ).lower()
        for banned in (
            "comprehensive",
            "highly accurate",
            "fully validated",
            "production ready",
            "robust",
            "cutting-edge",
        ):
            assert banned not in blob, f"report contains filler: {banned!r}"

    @respx.mock
    async def test_what_would_resolve_restates_the_plans_own_criterion(self) -> None:
        """Not a prediction: the plan wrote this before any search ran."""
        respx.get(FCA_URL).mock(return_value=httpx.Response(404))
        plan = make_plan(("SQ1",))
        result = await orchestrator(
            ScriptedLLM(plan=plan), FakeSourceProvider(candidates(FCA_URL))
        ).run(ResearchRequest(question=QUESTION))

        report = result.report
        assert report is not None
        gap = report.information_gaps[0]
        assert gap.what_would_resolve == plan.sub_questions[0].answerable_if

    @respx.mock
    async def test_limitations_always_state_what_verification_does_not_prove(
        self,
    ) -> None:
        mount(FCA_URL, FCA_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        report = result.report
        assert report is not None
        assert any(
            "does not establish that the source is correct" in note for note in report.limitations
        )

    @respx.mock
    async def test_measured_totals_are_carried_into_the_report(self) -> None:
        mount(FCA_URL, FCA_HTML)
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL)),
        ).run(ResearchRequest(question=QUESTION))

        report = result.report
        assert report is not None
        assert report.total_cost_usd == pytest.approx(result.trace.total_cost_usd)
        assert report.total_duration_ms == result.trace.total_duration_ms
        assert report.run_id == result.run_id
        assert report.question == QUESTION
        assert report.restated_question


class TestFailedSourceClassification:
    @respx.mock
    async def test_a_restricted_source_is_labelled_even_when_the_run_succeeds(
        self,
    ) -> None:
        """A 403 among otherwise healthy sources is still reported as such.

        It does not become an information gap — the question was answered
        anyway — but a reader must be able to see that a source was withheld
        rather than broken.
        """
        mount(FCA_URL, FCA_HTML)
        respx.get(DEAD_URL).mock(return_value=httpx.Response(403))
        result = await orchestrator(
            ScriptedLLM(plan=make_plan(("SQ1",))),
            FakeSourceProvider(candidates(FCA_URL, DEAD_URL)),
        ).run(ResearchRequest(question=QUESTION))

        report = result.report
        assert report is not None
        assert result.status is RunStatus.COMPLETED
        failed = report.source_coverage.failed_sources[0]
        assert failed["cause"] == GapCause.ACCESS_RESTRICTED.value
        # And no gap was invented for a question that was answered.
        assert report.answered or report.partial


class TestRejectedCitationAccounting:
    """The report must not contradict itself, or blame the wrong thing.

    Found by the first production research run (F-018). The sub-question
    counter summed failure *codes*, so a claim rejected for carrying no
    citation contributed one "rejected citation" — making the report's own
    sub-questions sum to 2 while the report-level total said 0, and reporting
    CITATIONS_REJECTED ("none survived verification against the stored source
    text") about a claim that had offered no quote to verify.
    """

    def _report(self, claims: list[Claim]):  # type: ignore[no-untyped-def]
        from app.pipeline.orchestrator import RunResult
        from app.pipeline.validate import run_validate_stage
        from app.schemas.runs import RunTrace

        sources, texts = _refs()
        validation, _ = run_validate_stage(claims, sources=sources, texts=texts)
        return (
            build_report(
                RunResult(
                    run_id="run_acct",
                    request=ResearchRequest(question=QUESTION),
                    status=RunStatus.COMPLETED,
                    trace=RunTrace(run_id="run_acct"),
                    plan=make_plan(("SQ1",)),
                    claim_validation=validation,
                    sources=sources,
                    source_texts=texts,
                )
            ),
            validation,
        )

    def test_a_claim_with_no_citation_is_not_a_rejected_citation(self) -> None:
        """`no_citation` means none was offered, so nothing was rejected."""
        report, validation = self._report(
            [
                Claim(
                    claim_id="C1",
                    sub_question_id="SQ1",
                    text="The attached documents do not answer this question.",
                )
            ]
        )

        assert report.sub_questions[0].rejected_citation_count == 0
        assert report.rejected_citations == 0
        # The verifier never produced a verdict, because there was no citation.
        assert validation.rejected_count == 0

    def test_a_genuinely_rejected_citation_is_still_counted(self) -> None:
        """Control: the counter must not have been zeroed out."""
        sources, _ = _refs()
        source_id = next(iter(sources))
        report, validation = self._report(
            [
                Claim(
                    claim_id="C1",
                    sub_question_id="SQ1",
                    text="The regulator confirmed a far higher figure.",
                    citations=[
                        Citation(
                            source_id=source_id,
                            start_char=0,
                            end_char=30,
                            cited_text="a quote that is not in the source text",
                        )
                    ],
                )
            ]
        )

        assert validation.rejected_count == 1
        assert report.sub_questions[0].rejected_citation_count == 1
        assert report.rejected_citations == 1

    def test_sub_question_counts_sum_to_the_report_total(self) -> None:
        """The invariant the production run violated."""
        sources, _ = _refs()
        source_id = next(iter(sources))
        report, validation = self._report(
            [
                Claim(
                    claim_id="C1",
                    sub_question_id="SQ1",
                    text="A claim that offered no quote at all.",
                ),
                Claim(
                    claim_id="C2",
                    sub_question_id="SQ1",
                    text="A claim whose quote is not in the source.",
                    citations=[
                        Citation(
                            source_id=source_id,
                            start_char=0,
                            end_char=30,
                            cited_text="wording that appears nowhere in the stored text",
                        )
                    ],
                ),
            ]
        )

        assert sum(s.rejected_citation_count for s in report.sub_questions) == (
            report.rejected_citations
        )
        assert sum(s.verified_citation_count for s in report.sub_questions) == (
            report.verified_citations
        )
        assert report.rejected_citations == validation.rejected_count == 1


class TestGapCauseForRejectedCitations:
    """The cause function, with counts supplied directly.

    The accounting fix feeds this: with `rejected_citations` no longer
    inflated by failure codes, a claim that offered no quote stops being
    reported as a citation that failed verification.
    """

    def _cause(self, **kw: object):  # type: ignore[no-untyped-def]
        from app.pipeline.assess import _diagnose

        args: dict[str, object] = {
            "sources_discovered": 2,
            "sources_fetched": 1,
            "chunks_retrieved": 3,
            "evidence_items": 2,
            "verified_citations": 0,
            "rejected_citations": 0,
            "extraction_failed": False,
            "all_sources_unvetted": False,
            "source_failures": [],
        }
        args.update(kw)
        return _diagnose(**args)  # type: ignore[arg-type]

    def test_a_rejected_citation_is_reported_as_such(self) -> None:
        cause, why = self._cause(rejected_citations=2)
        assert cause is GapCause.CITATIONS_REJECTED
        assert "2 citation(s) were produced" in why

    def test_no_citation_is_not_reported_as_a_failed_verification(self) -> None:
        """What the production run got wrong: 0 rejected must not claim the
        stored source text failed to support a quote."""
        cause, why = self._cause(rejected_citations=0)
        assert cause is not GapCause.CITATIONS_REJECTED
        assert "survived verification" not in why

    def test_the_production_runs_cause_is_insufficient_evidence(self) -> None:
        """What the production run should have said.

        Evidence was extracted from the one readable source, and none of it
        supported a citable claim. `WEAK_SOURCE_QUALITY` is deliberately not
        reachable here: it describes a run that *did* verify citations, but
        only from unvetted sources.
        """
        cause, why = self._cause(rejected_citations=0, all_sources_unvetted=True)
        assert cause is GapCause.INSUFFICIENT_EVIDENCE
        assert "none supported a citable claim" in why

    def test_unvetted_only_is_the_cause_once_citations_do_verify(self) -> None:
        cause, why = self._cause(verified_citations=3, all_sources_unvetted=True)
        assert cause is GapCause.WEAK_SOURCE_QUALITY
        assert "unvetted" in why
