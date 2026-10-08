"""Stage 8 — ASSESS. The report, and what the run could not establish.

**No model call.** Like stages 7 and the report verification before it, this
stage is arithmetic over outputs the pipeline already produced and verified.
That is deliberate: a model asked to summarise its own run's shortcomings is
being asked to audit itself, and the whole point of this layer is that the
shortcomings are counted rather than characterised.

Status is derived from verified evidence, by rule:

| Condition | Status |
|---|---|
| no claim survived verification | **UNKNOWN** |
| a claim survived at HIGH or MODERATE confidence | **ANSWERED** |
| a claim survived, but only at LOW confidence | **PARTIAL** |

The PARTIAL rule is the one that matters. A sub-question answered solely by a
single unvetted source is not the same finding as one answered by
corroborated primary sources, and the verifier already distinguishes them via
`Confidence`. Reporting both as ANSWERED would be exactly the confidence
inflation this milestone exists to prevent — so the distinction is carried
through rather than flattened.

Nothing here invents a reason. Every `unknown_reason` and `why_insufficient`
string is composed from counts the pipeline measured, and
`what_would_resolve` restates the plan's own `answerable_if` criterion, which
was written before any source was seen.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog

from app.schemas.evidence import Claim, Confidence, CredibilityTier
from app.schemas.report import (
    GapCause,
    InformationGap,
    ResearchReport,
    SourceCoverage,
    SubQuestionAssessment,
    SubQuestionStatus,
)
from app.schemas.runs import Stage, StageMetric

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from app.pipeline.orchestrator import RunResult

logger = structlog.get_logger(__name__)

# Confidence levels that make a sub-question ANSWERED rather than PARTIAL.
_USABLE_CONFIDENCE = {Confidence.HIGH, Confidence.MODERATE}

# Fetch failures that indicate the source exists but is not readable, as
# distinct from one that is simply broken. Read off the recorded failure
# message, which carries the upstream status.
_ACCESS_RESTRICTED_MARKERS = ("401", "402", "403", "paywall", "subscription")


def _best_confidence(claims: list[Claim]) -> Confidence:
    """Highest confidence among supported claims, or UNKNOWN if none."""
    order = [Confidence.HIGH, Confidence.MODERATE, Confidence.LOW]
    present = {c.confidence for c in claims}
    for level in order:
        if level in present:
            return level
    return Confidence.UNKNOWN


def _classify_failure(code: str, message: str) -> GapCause:
    """Map a recorded source failure onto a gap cause."""
    if code == "unsafe_url":
        return GapCause.SOURCE_BLOCKED
    lowered = message.lower()
    if any(marker in lowered for marker in _ACCESS_RESTRICTED_MARKERS):
        return GapCause.ACCESS_RESTRICTED
    return GapCause.SOURCE_FETCH_FAILED


def _diagnose(
    *,
    sources_discovered: int,
    sources_fetched: int,
    chunks_retrieved: int,
    evidence_items: int,
    verified_citations: int,
    rejected_citations: int,
    extraction_failed: bool,
    all_sources_unvetted: bool,
    source_failures: list[tuple[str, str]],
) -> tuple[GapCause, str]:
    """Determine why a sub-question went unanswered, from measured counts.

    Ordered from the earliest point of failure outward, so the cause reported
    is the first thing that actually went wrong rather than its downstream
    symptom.
    """
    if sources_discovered == 0:
        return (
            GapCause.NO_SOURCES_DISCOVERED,
            "Discovery returned no candidate sources for this sub-question.",
        )
    if sources_fetched == 0:
        if source_failures:
            code, message = source_failures[0]
            cause = _classify_failure(code, message)
            return (
                cause,
                f"All {len(source_failures)} candidate source(s) failed to "
                f"fetch; first failure was {code}.",
            )
        return (
            GapCause.SOURCE_FETCH_FAILED,
            "No candidate source could be fetched.",
        )
    if chunks_retrieved == 0:
        return (
            GapCause.NOTHING_RETRIEVED,
            f"{sources_fetched} source(s) were read but retrieval surfaced no "
            "passage relevant to this sub-question.",
        )
    if extraction_failed:
        return (
            GapCause.EXTRACTION_FAILED,
            "Evidence extraction failed on the retrieved passages for this sub-question.",
        )
    if evidence_items == 0:
        return (
            GapCause.INSUFFICIENT_EVIDENCE,
            f"{chunks_retrieved} passage(s) were examined and none contained "
            "evidence bearing on this sub-question.",
        )
    if verified_citations == 0 and rejected_citations > 0:
        return (
            GapCause.CITATIONS_REJECTED,
            f"{rejected_citations} citation(s) were produced and none survived "
            "verification against the stored source text.",
        )
    if verified_citations == 0:
        return (
            GapCause.INSUFFICIENT_EVIDENCE,
            f"{evidence_items} evidence item(s) were extracted but none supported a citable claim.",
        )
    if all_sources_unvetted:
        return (
            GapCause.WEAK_SOURCE_QUALITY,
            "The supporting evidence comes only from sources classified "
            "unvetted; no primary or established secondary source was read.",
        )
    return (
        GapCause.INSUFFICIENT_EVIDENCE,
        f"{evidence_items} evidence item(s) were extracted but the verified "
        "evidence does not settle this sub-question.",
    )


def _executive_summary(
    assessments: list[SubQuestionAssessment],
    coverage: SourceCoverage,
) -> str:
    """A factual summary, assembled from counts.

    Composed rather than written, so it cannot drift from the numbers beneath
    it or acquire language the evidence does not support.
    """
    answered = sum(1 for a in assessments if a.status is SubQuestionStatus.ANSWERED)
    partial = sum(1 for a in assessments if a.status is SubQuestionStatus.PARTIAL)
    unknown = sum(1 for a in assessments if a.status is SubQuestionStatus.UNKNOWN)
    total = len(assessments)

    parts = [
        f"Of {total} sub-question(s): {answered} answered, {partial} partially "
        f"answered, {unknown} unknown."
    ]
    parts.append(
        f"{coverage.fetched} of {coverage.discovered} discovered source(s) were "
        f"read; {coverage.failed} could not be."
    )
    if not coverage.has_primary_source and coverage.fetched:
        parts.append(
            "No primary source was read, so no finding in this report exceeds "
            "moderate confidence on provenance grounds."
        )
    if unknown == total and total:
        parts.append(
            "No sub-question was answered with verified evidence; this run "
            "establishes nothing and the gaps below state why."
        )
    return " ".join(parts)


def _limitations(
    result: RunResult, coverage: SourceCoverage, assessments: list[SubQuestionAssessment]
) -> list[str]:
    """Limitations observed in this run. Measured, not boilerplate."""
    notes: list[str] = []

    if coverage.failed:
        notes.append(
            f"{coverage.failed} source(s) could not be read, so their content "
            "is absent from every finding."
        )
    if not coverage.has_primary_source and coverage.fetched:
        unvetted = coverage.by_credibility.get(CredibilityTier.UNVETTED.value, 0)
        notes.append(
            f"Source quality is weak: {unvetted} of {coverage.fetched} source(s) "
            "are unvetted and none is primary. Discovery, not verification, is "
            "the limiting factor."
        )
    if result.extraction is not None and result.extraction.unlocatable:
        notes.append(
            f"{len(result.extraction.unlocatable)} extracted quote(s) could not "
            "be located in their source and were discarded, so some evidence "
            "the extractor found is absent from this report."
        )
    if result.extraction is not None and result.extraction.failures:
        notes.append(
            f"Evidence extraction failed on {len(result.extraction.failures)} "
            "passage(s); any evidence they held is missing."
        )
    if result.claim_validation is not None and result.claim_validation.rejected_count:
        counts = result.claim_validation.failure_counts
        notes.append(
            f"{result.claim_validation.rejected_count} citation(s) were rejected "
            f"by verification ({counts}); the claims resting on them are "
            "reported UNKNOWN rather than removed."
        )
    if all(a.status is not SubQuestionStatus.ANSWERED for a in assessments) and assessments:
        notes.append(
            "No sub-question reached ANSWERED status, which requires at least "
            "one verified claim at moderate confidence or better."
        )
    notes.append(
        "Citation verification proves a quote is present in a named source. It "
        "does not establish that the source is correct."
    )
    return notes


def _next_steps(gaps: list[InformationGap], assessments: list[SubQuestionAssessment]) -> list[str]:
    """Next steps that follow from the gaps, in gap order."""
    steps: list[str] = []
    for gap in gaps:
        if gap.cause is GapCause.WEAK_SOURCE_QUALITY:
            steps.append(
                f"[{gap.sub_question_id}] Seek a primary source. {gap.what_would_resolve}"
                if gap.what_would_resolve
                else f"[{gap.sub_question_id}] Seek a primary source."
            )
        elif gap.cause in {GapCause.ACCESS_RESTRICTED, GapCause.SOURCE_FETCH_FAILED}:
            steps.append(
                f"[{gap.sub_question_id}] Obtain the source that could not be "
                f"read, or find an accessible equivalent."
            )
        elif gap.cause is GapCause.NO_SOURCES_DISCOVERED:
            steps.append(
                f"[{gap.sub_question_id}] Re-run discovery with different query "
                f"wording. {gap.what_would_resolve}".strip()
            )
        elif gap.cause is GapCause.NOTHING_RETRIEVED:
            steps.append(
                f"[{gap.sub_question_id}] The sources read do not cover this; "
                "search specifically for it."
            )
        else:
            steps.append(
                f"[{gap.sub_question_id}] {gap.what_would_resolve}"
                if gap.what_would_resolve
                else f"[{gap.sub_question_id}] Gather further evidence."
            )
    if not steps and assessments:
        steps.append(
            "Corroborate the findings above against an independent source before relying on them."
        )
    return steps


def build_report(result: RunResult) -> ResearchReport:
    """Assemble the final report from verified pipeline output.

    Pure: given the same `RunResult` this returns the same report, which is
    what makes the information gaps testable as a deterministic function of
    the pipeline rather than a narrative about it.
    """
    plan = result.plan
    claim_validation = result.claim_validation

    # -- source coverage ---------------------------------------------------
    by_credibility: dict[str, int] = {}
    for reference in result.sources.values():
        key = reference.credibility.value
        by_credibility[key] = by_credibility.get(key, 0) + 1

    failures = result.processed.failures if result.processed else []
    coverage = SourceCoverage(
        discovered=len(result.discovery.candidates) if result.discovery else 0,
        fetched=len(result.processed.sources) if result.processed else 0,
        failed=len(failures),
        by_credibility=by_credibility,
        failed_sources=[
            {
                "url": f.url,
                "code": f.code,
                # The classified cause, so a reader sees "access_restricted"
                # rather than a raw transport code. Same mapping the gap
                # diagnosis uses, so the two cannot disagree.
                "cause": _classify_failure(f.code, f.message).value,
                "message": f.message[:200],
            }
            for f in failures
        ],
        domains=sorted({r.domain for r in result.sources.values()}),
    )

    # -- group verified output by sub-question -----------------------------
    supported_by_sq: dict[str, list[Claim]] = {}
    unknown_by_sq: dict[str, list[Claim]] = {}
    if claim_validation is not None:
        for claim in claim_validation.supported_claims:
            supported_by_sq.setdefault(claim.sub_question_id or "", []).append(claim)
        for claim in claim_validation.unknown_claims:
            unknown_by_sq.setdefault(claim.sub_question_id or "", []).append(claim)

    evidence_by_sq: dict[str, int] = {}
    if result.extraction is not None:
        verified_evidence_ids = {
            v.citation.source_id
            for v in (result.evidence_validation.verdicts if result.evidence_validation else [])
            if v.ok
        }
        for item in result.extraction.items:
            if item.source_id in verified_evidence_ids:
                evidence_by_sq[item.sub_question_id] = (
                    evidence_by_sq.get(item.sub_question_id, 0) + 1
                )

    source_failures = [(f.code, f.message) for f in failures]
    all_unvetted = bool(result.sources) and all(
        r.credibility is CredibilityTier.UNVETTED for r in result.sources.values()
    )
    extraction_failed = bool(result.extraction and result.extraction.failures)

    # -- per sub-question ---------------------------------------------------
    assessments: list[SubQuestionAssessment] = []
    gaps: list[InformationGap] = []

    for sub_question in plan.ordered() if plan else []:
        sqid = sub_question.id
        supported = supported_by_sq.get(sqid, [])
        unknown = unknown_by_sq.get(sqid, [])
        verified_citations = sum(c.verified_citations for c in supported)
        rejected_citations = sum(len(c.failures) for c in unknown if c.failures)
        chunks_retrieved = len(result.retrieved.get(sqid, []))
        evidence_count = evidence_by_sq.get(sqid, 0)

        confidence = _best_confidence(supported)
        if not supported:
            status = SubQuestionStatus.UNKNOWN
        elif confidence in _USABLE_CONFIDENCE:
            status = SubQuestionStatus.ANSWERED
        else:
            # Supported, but only by low-confidence evidence. Reported as
            # PARTIAL rather than ANSWERED — see the module docstring.
            status = SubQuestionStatus.PARTIAL

        gap: InformationGap | None = None
        reason = ""
        if status is not SubQuestionStatus.ANSWERED:
            cause, why = _diagnose(
                sources_discovered=coverage.discovered,
                sources_fetched=coverage.fetched,
                chunks_retrieved=chunks_retrieved,
                evidence_items=evidence_count,
                verified_citations=verified_citations,
                rejected_citations=rejected_citations,
                extraction_failed=extraction_failed,
                all_sources_unvetted=all_unvetted,
                source_failures=source_failures,
            )
            reason = why
            gap = InformationGap(
                sub_question_id=sqid,
                question=sub_question.question,
                cause=cause,
                why_insufficient=why,
                # The plan's own criterion, written before any search ran.
                what_would_resolve=sub_question.answerable_if,
                sources_considered=coverage.discovered,
                chunks_retrieved=chunks_retrieved,
                evidence_items=evidence_count,
                verified_citations=verified_citations,
            )
            gaps.append(gap)

        assessments.append(
            SubQuestionAssessment(
                sub_question_id=sqid,
                question=sub_question.question,
                rank=sub_question.rank,
                status=status,
                confidence=confidence,
                supporting_claims=supported,
                unknown_claims=unknown,
                verified_evidence_count=evidence_count,
                verified_citation_count=verified_citations,
                rejected_citation_count=rejected_citations,
                unknown_reason=reason,
                information_gap=gap,
            )
        )

    return ResearchReport(
        run_id=result.run_id,
        question=result.request.question,
        restated_question=plan.restated_question if plan else "",
        executive_summary=_executive_summary(assessments, coverage),
        sub_questions=assessments,
        unknowns=[
            f"[{a.sub_question_id}] {a.question} — {a.unknown_reason}"
            for a in assessments
            if a.status is SubQuestionStatus.UNKNOWN
        ],
        information_gaps=gaps,
        source_coverage=coverage,
        limitations=_limitations(result, coverage, assessments),
        recommended_next_steps=_next_steps(gaps, assessments),
        total_claims=len(claim_validation.claims) if claim_validation else 0,
        supported_claims=len(claim_validation.supported_claims) if claim_validation else 0,
        unknown_claims=len(claim_validation.unknown_claims) if claim_validation else 0,
        verified_citations=claim_validation.verified_count if claim_validation else 0,
        rejected_citations=claim_validation.rejected_count if claim_validation else 0,
        total_cost_usd=result.trace.total_cost_usd,
        total_duration_ms=result.trace.total_duration_ms,
    )


def run_assess_stage(result: RunResult) -> tuple[ResearchReport, StageMetric]:
    """Stage 8. No model call, so no cost is attributed to it."""
    started = time.perf_counter()
    report = build_report(result)
    metric = StageMetric(
        stage=Stage.ASSESS,
        model=None,
        duration_ms=int((time.perf_counter() - started) * 1000),
        calls=0,
    )
    logger.info(
        "stage_complete",
        stage=Stage.ASSESS.value,
        sub_questions=len(report.sub_questions),
        answered=len(report.answered),
        partial=len(report.partial),
        unknown=len(report.unanswered),
        information_gaps=len(report.information_gaps),
        coverage_ratio=round(report.coverage_ratio, 3),
        duration_ms=metric.duration_ms,
    )
    return report, metric
