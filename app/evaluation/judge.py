"""Metrics 6 and 7: the LLM judge (architecture §9).

These two metrics are the only non-deterministic ones in the approved design,
and the architecture names the judge explicitly: Haiku, rubric-scored. They
exist because "is this answer relevant to the question" and "is this report
usable" are not properties a string comparison can establish.

Two safeguards, because a judge is a model and models are the thing under
evaluation:

1. **The judge never sees the system's own confidence or verdicts.** It is
   given the question, the answer text and the report's structure, and scores
   against a fixed written rubric. Showing it `confidence=HIGH` would invite
   it to agree with the system rather than assess it.
2. **The rubric is written down and versioned here**, not improvised per call,
   so a score is comparable between runs and cannot be nudged by rewording
   the prompt after seeing a baseline.

Non-determinism is real and is reported as such: these two metrics may move
between runs on identical input. The other five will not.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.llm.client import LLMClient
from app.schemas.report import ResearchReport
from app.schemas.runs import TokenUsage

logger = structlog.get_logger(__name__)

JUDGE_RUBRIC_VERSION = "rubric_v1"

ANSWER_RELEVANCE_RUBRIC = """\
You are scoring whether an answer addresses the question that was asked. You
are not scoring whether the answer is true, well written, or complete.

Score 0.0 to 1.0:

1.0  Directly answers the question asked, in its own terms.
0.75 Answers the question but partially — a material part is unaddressed.
0.5  Addresses the general topic but not the specific question.
0.25 Related to the subject but does not address the question.
0.0  Does not address the question, or answers a different question.

An answer that states it cannot answer the question from the available
evidence scores 1.0 **if that is the honest outcome for the question asked**,
and 0.0 if the material plainly contained the answer. Declining to answer is
a correct response to an unanswerable question, not a failure.

Do not reward length, confidence, or fluency. Do not penalise an answer for
being short if it answers the question."""

REPORT_QUALITY_RUBRIC = """\
You are scoring whether a research report is usable by a human researcher.

Score 0.0 to 1.0 against these criteria, weighted equally:

- Can a reader tell what the research established and what it did not?
- Is every unanswered question named, with a reason, rather than omitted?
- Are the limitations specific to this run rather than generic boilerplate?
- Are the next steps actionable and tied to a stated gap?
- Is the report free of filler, marketing language and unsupported confidence?

1.0  All five hold.
0.8  Four hold.
0.6  Three hold.
0.4  Two hold.
0.2  One holds.
0.0  None hold, or the report asserts findings it does not support.

Penalise a report that reads as confident while its own figures show thin
evidence. Do not reward volume."""


class JudgeScore(BaseModel):
    """One judged score with its stated reason."""

    score: float = Field(ge=0.0, le=1.0, description="The rubric score between 0.0 and 1.0.")
    reason: str = Field(
        min_length=8,
        # Third time a guessed bound on a model-written field has rejected a
        # correct response (F-004, F-011, F-013). 600 was arbitrary; a judge
        # explaining a rubric score against five criteria legitimately writes
        # more than that.
        max_length=2000,
        description=(
            "One or two sentences citing what in the material drove the score. "
            "Reference the material, not general impressions."
        ),
    )


@dataclass(slots=True)
class JudgeResult:
    """Scores for one case, plus what judging cost."""

    case_id: str
    answer_relevance: float | None = None
    answer_reason: str = ""
    report_quality: float | None = None
    report_reason: str = ""
    usage: TokenUsage | None = None
    cost_usd: float = 0.0
    calls: int = 0


def _answer_payload(report: ResearchReport) -> str:
    """The answer text, without any of the system's own verdicts."""
    parts: list[str] = []
    for assessment in report.sub_questions:
        parts.append(f"Sub-question: {assessment.question}")
        parts.append(f"Answer: {assessment.answer}")
    return "\n".join(parts) if parts else "(no answer produced)"


def _report_payload(report: ResearchReport) -> str:
    """The report's structure, minus confidence labels the judge might copy."""
    lines = [
        f"Executive summary: {report.executive_summary}",
        "",
        f"Sub-questions: {len(report.sub_questions)}",
    ]
    for assessment in report.sub_questions:
        lines.append(f"- {assessment.question}")
        lines.append(f"  answer: {assessment.answer[:400]}")
        if assessment.unknown_reason:
            lines.append(f"  unanswered because: {assessment.unknown_reason}")
    lines.append("")
    lines.append(f"Unanswered questions listed: {len(report.unknowns)}")
    for unknown in report.unknowns:
        lines.append(f"- {unknown}")
    lines.append("")
    lines.append(f"Information gaps: {len(report.information_gaps)}")
    for gap in report.information_gaps:
        lines.append(f"- cause={gap.cause.value}: {gap.why_insufficient}")
        if gap.what_would_resolve:
            lines.append(f"  would be resolved by: {gap.what_would_resolve}")
    lines.append("")
    lines.append("Limitations:")
    for limitation in report.limitations:
        lines.append(f"- {limitation}")
    lines.append("")
    lines.append("Recommended next steps:")
    for step in report.recommended_next_steps:
        lines.append(f"- {step}")
    lines.append("")
    lines.append(
        f"Source coverage: {report.source_coverage.fetched} of "
        f"{report.source_coverage.discovered} discovered sources read, "
        f"{report.source_coverage.failed} failed."
    )
    return "\n".join(lines)


async def judge_case(
    case_id: str,
    question: str,
    report: ResearchReport,
    *,
    client: LLMClient,
    settings: Settings,
) -> JudgeResult:
    """Score metrics 6 and 7 for one case. Two calls on the cheap model."""
    result = JudgeResult(case_id=case_id)
    usage = TokenUsage()

    answer = await client.structured(
        model=settings.extraction_model,
        output_model=JudgeScore,
        system=ANSWER_RELEVANCE_RUBRIC,
        user_content=(
            f"Question asked:\n{question}\n\nAnswer produced:\n{_answer_payload(report)}"
        ),
        max_tokens=1024,
    )
    result.answer_relevance = answer.value.score
    result.answer_reason = answer.value.reason
    usage = usage + answer.usage
    result.cost_usd += answer.cost_usd
    result.calls += 1

    quality = await client.structured(
        model=settings.extraction_model,
        output_model=JudgeScore,
        system=REPORT_QUALITY_RUBRIC,
        user_content=f"Question asked:\n{question}\n\nReport:\n{_report_payload(report)}",
        max_tokens=1024,
    )
    result.report_quality = quality.value.score
    result.report_reason = quality.value.reason
    usage = usage + quality.usage
    result.cost_usd += quality.cost_usd
    result.calls += 1

    result.usage = usage
    logger.info(
        "case_judged",
        case_id=case_id,
        answer_relevance=result.answer_relevance,
        report_quality=result.report_quality,
        cost_usd=round(result.cost_usd, 6),
    )
    return result
