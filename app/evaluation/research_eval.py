"""The seven approved metrics (architecture §9).

| # | Metric | Method | Deterministic |
|---|---|---|---|
| 1 | Retrieval relevance | recall@k, MRR, nDCG vs annotated relevant sources | ✅ |
| 2 | Source coverage | fraction of annotated sources discovered | ✅ |
| 3 | Citation correctness | offsets re-verified against stored source text | ✅ |
| 4 | Unsupported-claim rate | claims with no verified citation ÷ total claims | ✅ |
| 5 | Tool selection | expected vs actual tool sequence | ✅ |
| 6 | Answer relevance | LLM judge (Haiku), rubric-scored | ✗ |
| 7 | Report quality | LLM judge against a written rubric | ✗ |

Definitions are taken from the architecture verbatim and are not restated in
a form that would be easier to score well on. Metrics 1–5 are computed here;
6 and 7 are produced by `app.evaluation.judge` and merged in, because they
need a model and the rest must stay runnable for free.

**Metric 3 re-verifies independently.** It does not read
`Claim.verified_citations`, which the pipeline's own verifier set. It
re-slices the stored source text itself. Trusting the field would make the
metric a tautology — the component under evaluation grading its own work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from app.evaluation.golden import GoldenCase
from app.evaluation.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank
from app.pipeline.validate import normalise_for_comparison
from app.schemas.report import ResearchReport

if TYPE_CHECKING:  # pragma: no cover
    from app.pipeline.orchestrator import RunResult

logger = structlog.get_logger(__name__)

EVALUATION_VERSION = "eval_v1"

# The cutoff metric 1 reports at. Fixed here so it cannot be chosen after
# seeing a score; the architecture specifies recall@k without fixing k, and
# k=5 sits well below the 20-document corpus (see F-008).
RETRIEVAL_K = 5


@dataclass(slots=True)
class CaseResult:
    """Per-case metric values. None means the metric does not apply."""

    case_id: str
    split: str
    question: str

    # 1 · retrieval relevance
    recall_at_k: float | None = None
    mrr: float | None = None
    ndcg_at_k: float | None = None
    # 2 · source coverage
    source_coverage: float | None = None
    # 3 · citation correctness
    citations_total: int = 0
    citations_correct: int = 0
    # 4 · unsupported-claim rate
    claims_total: int = 0
    claims_unsupported: int = 0
    # 5 · tool selection
    tool_sequence_correct: bool | None = None
    expected_tools: tuple[str, ...] = ()
    actual_tools: tuple[str, ...] = ()
    # 6, 7 · judged, merged in later
    answer_relevance: float | None = None
    report_quality: float | None = None

    # Did the run agree with the case's advance expectation?
    expected_unknown: bool = False
    observed_unknown: bool = False
    run_status: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def citation_correctness(self) -> float | None:
        if self.citations_total == 0:
            return None
        return self.citations_correct / self.citations_total

    @property
    def unsupported_claim_rate(self) -> float | None:
        if self.claims_total == 0:
            return None
        return self.claims_unsupported / self.claims_total

    @property
    def unknown_expectation_met(self) -> bool:
        return self.expected_unknown == self.observed_unknown


def _url_to_doc_id(url: str, doc_urls: dict[str, str]) -> str | None:
    """Map a fetched URL back to the golden corpus doc_id."""
    for doc_id, doc_url in doc_urls.items():
        if doc_url == url:
            return doc_id
    return None


def score_case(
    case: GoldenCase,
    result: RunResult,
    doc_urls: dict[str, str],
) -> CaseResult:
    """Compute metrics 1–5 for one case from its run.

    `doc_urls` maps golden corpus doc_id to the URL the fixture served it at,
    so retrieval and discovery output can be scored against doc-level
    judgments.
    """
    report: ResearchReport | None = result.report
    scored = CaseResult(
        case_id=case.case_id,
        split=case.split,
        question=case.question,
        expected_tools=case.expected_tools,
        expected_unknown=case.expect_unknown,
        run_status=result.status.value,
    )

    # -- 1 · retrieval relevance -----------------------------------------
    # Retrieved chunks collapsed to documents, first occurrence wins, so the
    # measure does not move when chunk size changes (as in M3).
    retrieved_docs: list[str] = []
    for chunks in result.retrieved.values():
        for chunk in chunks:
            doc_id = _url_to_doc_id(str(chunk.source_url), doc_urls)
            if doc_id is not None and doc_id not in retrieved_docs:
                retrieved_docs.append(doc_id)

    if case.is_answerable:
        scored.recall_at_k = recall_at_k(retrieved_docs, case.judgments, RETRIEVAL_K)
        scored.mrr = reciprocal_rank(retrieved_docs, case.judgments)
        scored.ndcg_at_k = ndcg_at_k(retrieved_docs, case.judgments, RETRIEVAL_K)
    else:
        # A case with nothing relevant cannot have retrieval scored; scoring
        # it 0.0 would drag the mean for a case that is not about retrieval.
        scored.notes.append("retrieval not scored: no annotated relevant document")

    # -- 2 · source coverage ---------------------------------------------
    if case.is_answerable:
        discovered_docs: set[str] = set()
        if result.discovery is not None:
            for candidate in result.discovery.candidates:
                doc_id = _url_to_doc_id(str(candidate.url), doc_urls)
                if doc_id is not None:
                    discovered_docs.add(doc_id)
        scored.source_coverage = len(discovered_docs & case.relevant_docs) / len(case.relevant_docs)

    # -- 3 · citation correctness ----------------------------------------
    # Re-verified here, independently of the pipeline's own verdict.
    if report is not None:
        for assessment in report.sub_questions:
            for claim in assessment.supporting_claims + assessment.unknown_claims:
                for citation in claim.citations:
                    scored.citations_total += 1
                    stored = result.source_texts.get(citation.source_id)
                    if stored is None:
                        continue
                    if citation.end_char > len(stored):
                        continue
                    actual_slice = stored[citation.start_char : citation.end_char]
                    if actual_slice == citation.cited_text or normalise_for_comparison(
                        actual_slice
                    ) == normalise_for_comparison(citation.cited_text):
                        scored.citations_correct += 1

    # -- 4 · unsupported-claim rate --------------------------------------
    if result.claim_validation is not None:
        claims = result.claim_validation.claims
        scored.claims_total = len(claims)
        scored.claims_unsupported = sum(1 for claim in claims if claim.verified_citations == 0)

    # -- 5 · tool selection ----------------------------------------------
    if result.discovery is not None:
        # The discovery stage exposes one tool; the sequence actually
        # exercised is one entry per query issued.
        actual_tools = tuple("search_sources" for _ in result.discovery.queries_issued)
        scored.actual_tools = actual_tools
        # Expected is the set of tools, not a repetition count: the plan does
        # not fix how many searches a question needs.
        scored.tool_sequence_correct = set(actual_tools) == set(case.expected_tools) or (
            not actual_tools and not case.expected_tools
        )

    # -- expectation agreement -------------------------------------------
    if report is not None:
        scored.observed_unknown = report.supported_claims == 0
    else:
        scored.observed_unknown = True

    return scored


@dataclass(slots=True)
class MetricSummary:
    """One metric's aggregate value and the cases it was computed over."""

    name: str
    value: float | None
    cases_scored: int
    deterministic: bool
    detail: str = ""


@dataclass(slots=True)
class EvaluationReport:
    """The committed evaluation artifact."""

    evaluation_version: str
    golden_set_version: str
    git_sha: str
    retrieval_k: int
    cases: list[CaseResult] = field(default_factory=list)
    label: str = "baseline"

    # -- the seven metrics -------------------------------------------------

    def _vals(self, attr: str, *, split: str | None = None) -> list[float]:
        out = []
        for case in self.cases:
            if split is not None and case.split != split:
                continue
            value = getattr(case, attr)
            if value is not None:
                out.append(float(value))
        return out

    def metrics(self, *, split: str | None = None) -> list[MetricSummary]:
        """All seven, in the architecture's order."""
        recall = self._vals("recall_at_k", split=split)
        mrr = self._vals("mrr", split=split)
        ndcg = self._vals("ndcg_at_k", split=split)
        coverage = self._vals("source_coverage", split=split)

        scoped = [c for c in self.cases if split is None or c.split == split]
        cit_total = sum(c.citations_total for c in scoped)
        cit_ok = sum(c.citations_correct for c in scoped)
        claims_total = sum(c.claims_total for c in scoped)
        claims_unsupported = sum(c.claims_unsupported for c in scoped)
        tool_scored = [
            c.tool_sequence_correct for c in scoped if c.tool_sequence_correct is not None
        ]
        answer = self._vals("answer_relevance", split=split)
        quality = self._vals("report_quality", split=split)

        return [
            MetricSummary(
                name="retrieval_relevance",
                value=mean(recall) if recall else None,
                cases_scored=len(recall),
                deterministic=True,
                detail=(
                    f"recall@{self.retrieval_k}={mean(recall):.3f} "
                    f"MRR={mean(mrr):.3f} "
                    f"nDCG@{self.retrieval_k}={mean(ndcg):.3f}"
                    if recall
                    else "no answerable case scored"
                ),
            ),
            MetricSummary(
                name="source_coverage",
                value=mean(coverage) if coverage else None,
                cases_scored=len(coverage),
                deterministic=True,
                detail="fraction of annotated relevant sources discovered",
            ),
            MetricSummary(
                name="citation_correctness",
                value=(cit_ok / cit_total) if cit_total else None,
                cases_scored=sum(1 for c in scoped if c.citations_total),
                deterministic=True,
                detail=f"{cit_ok}/{cit_total} citations re-verified independently",
            ),
            MetricSummary(
                name="unsupported_claim_rate",
                value=(claims_unsupported / claims_total) if claims_total else None,
                cases_scored=sum(1 for c in scoped if c.claims_total),
                deterministic=True,
                detail=(
                    f"{claims_unsupported}/{claims_total} claims carry no verified "
                    "citation (lower is better)"
                ),
            ),
            MetricSummary(
                name="tool_selection",
                value=(sum(tool_scored) / len(tool_scored)) if tool_scored else None,
                cases_scored=len(tool_scored),
                deterministic=True,
                detail="expected vs actual tool set per case",
            ),
            MetricSummary(
                name="answer_relevance",
                value=mean(answer) if answer else None,
                cases_scored=len(answer),
                deterministic=False,
                detail="LLM judge (Haiku), rubric-scored 0–1",
            ),
            MetricSummary(
                name="report_quality",
                value=mean(quality) if quality else None,
                cases_scored=len(quality),
                deterministic=False,
                detail="LLM judge against a written rubric, 0–1",
            ),
        ]

    @property
    def unknown_expectation_accuracy(self) -> float | None:
        """Share of cases whose UNKNOWN/answered outcome matched the annotation.

        Not one of the seven metrics. Reported because it is the clearest
        single signal of whether the system's honesty behaves as designed, and
        the golden set annotates it in advance.
        """
        if not self.cases:
            return None
        return sum(1 for c in self.cases if c.unknown_expectation_met) / len(self.cases)

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluation_version": self.evaluation_version,
            "golden_set_version": self.golden_set_version,
            "git_sha": self.git_sha,
            "label": self.label,
            "retrieval_k": self.retrieval_k,
            "metrics": {
                m.name: {
                    "value": m.value,
                    "cases_scored": m.cases_scored,
                    "deterministic": m.deterministic,
                    "detail": m.detail,
                }
                for m in self.metrics()
            },
            "metrics_holdout": {m.name: m.value for m in self.metrics(split="holdout")},
            "unknown_expectation_accuracy": self.unknown_expectation_accuracy,
            "cases": [
                {
                    "case_id": c.case_id,
                    "split": c.split,
                    "run_status": c.run_status,
                    "recall_at_k": c.recall_at_k,
                    "mrr": c.mrr,
                    "ndcg_at_k": c.ndcg_at_k,
                    "source_coverage": c.source_coverage,
                    "citation_correctness": c.citation_correctness,
                    "citations_total": c.citations_total,
                    "unsupported_claim_rate": c.unsupported_claim_rate,
                    "claims_total": c.claims_total,
                    "tool_sequence_correct": c.tool_sequence_correct,
                    "answer_relevance": c.answer_relevance,
                    "report_quality": c.report_quality,
                    "expected_unknown": c.expected_unknown,
                    "observed_unknown": c.observed_unknown,
                    "unknown_expectation_met": c.unknown_expectation_met,
                    "notes": c.notes,
                }
                for c in self.cases
            ],
        }

    def summary_lines(self) -> list[str]:
        lines = [
            f"{'metric':<26} {'value':>8}  {'cases':>5}  det  detail",
            "-" * 100,
        ]
        for metric in self.metrics():
            value = "n/a" if metric.value is None else f"{metric.value:.3f}"
            lines.append(
                f"{metric.name:<26} {value:>8}  {metric.cases_scored:>5}  "
                f"{'Y' if metric.deterministic else 'N':^3}  {metric.detail}"
            )
        accuracy = self.unknown_expectation_accuracy
        lines.append("-" * 100)
        lines.append(
            "unknown_expectation_accuracy "
            + ("n/a" if accuracy is None else f"{accuracy:.3f}")
            + "   (not one of the seven; annotated in advance)"
        )
        return lines


def compare(
    baseline: EvaluationReport, current: EvaluationReport
) -> list[tuple[str, float | None, float | None, str]]:
    """Metric-by-metric delta, in the architecture's order.

    `unsupported_claim_rate` is the one metric where lower is better, so its
    verdict is inverted rather than reported backwards.
    """
    lower_is_better = {"unsupported_claim_rate"}
    rows: list[tuple[str, float | None, float | None, str]] = []
    before = {m.name: m.value for m in baseline.metrics()}
    after = {m.name: m.value for m in current.metrics()}

    for metric in baseline.metrics():
        name = metric.name
        was, now = before.get(name), after.get(name)
        if was is None and now is None:
            verdict = "not measured"
        elif was is None:
            verdict = "newly measured"
        elif now is None:
            verdict = "no longer measured"
        elif abs(now - was) < 1e-9:
            verdict = "unchanged"
        else:
            better = (now < was) if name in lower_is_better else (now > was)
            verdict = "improved" if better else "regressed"
        rows.append((name, was, now, verdict))
    return rows
