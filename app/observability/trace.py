"""Safe trace export for a research run (architecture capability 24).

**This is not a second tracing system.** Measurement already happens in
`app.schemas.runs` — `StageMetric` records each stage's duration, model, token
usage and cost as the stage runs, and `RunTrace` aggregates them. Adding
another collector would mean two sources of truth for the same numbers.

What was missing is the *export* half: turning a finished run into a record
that is safe to log, ship or attach to an artifact. That is this module's only
job, and it has two halves.

**1. Redaction.** A trace must be safe to put in a log aggregator. So it
carries counts, identifiers, hashes and durations — never prompt text, never
source text, never a secret. Source *content* is represented by its
`content_hash` and length, which is enough to tell two runs apart and prove
which text was read, without reproducing a document in a log line. The
question "did the trace leak anything" then has a testable answer, because
there is exactly one place that decides what goes in.

**2. Honesty about provenance.** Every number is tagged `measured`, `derived`
or `unavailable` (`Measurement`). The distinction is real and easy to blur:

- **measured** — the API or the clock reported it. Token counts come from
  `response.usage`; durations from `perf_counter`.
- **derived** — computed from measured inputs by code here. Cost is derived:
  the API returns tokens, not dollars, so cost is `tokens x price table`. It
  is exact for the table in `app.llm.pricing` and wrong the day prices change.
- **unavailable** — not exposed. Named rather than defaulted to zero, because
  a zero that means "we never saw it" and a zero that means "it was zero" are
  different facts.

Tracing observes; it never controls. Nothing in this module can change a
stage's outcome, and the orchestrator does not consult it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.llm.pricing import is_priced
from app.schemas.runs import RunTrace, Stage, StageMetric, StageStatus, TokenUsage

TRACE_VERSION = "trace_v1"


class Measurement(StrEnum):
    """Where a number came from. See the module docstring."""

    MEASURED = "measured"
    DERIVED = "derived"
    UNAVAILABLE = "unavailable"


class StageTrace(BaseModel):
    """One stage, as it is safe to export."""

    stage: Stage
    status: StageStatus
    duration_ms: int
    duration_provenance: Measurement = Measurement.MEASURED

    model: str | None = None
    calls: int = 0

    # Token counts, as reported by the API.
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    tokens_provenance: Measurement = Measurement.MEASURED

    cost_usd: float = 0.0
    # Derived from tokens and the price table, never returned by the API.
    cost_provenance: Measurement = Measurement.DERIVED

    # Failure information, kept rather than flattened: a stage that failed
    # silently in a trace is a stage nobody can diagnose.
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def cache_hit_rate(self) -> float | None:
        return TokenUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens,
        ).cache_hit_rate


class SourceTrace(BaseModel):
    """A source, identified without reproducing it.

    Deliberately no text and no title: a hash plus a length proves which
    document was read and lets two runs be compared, while a log line that
    carried the document would be both a leak and unreadable.
    """

    source_id: str
    domain: str
    content_hash: str
    text_chars: int
    credibility: str | None = None
    # Injection categories observed in this source's text. Telemetry only;
    # nothing was filtered on the strength of it.
    injection_signals: list[str] = Field(default_factory=list)


class RunTraceRecord(BaseModel):
    """A complete run, safe to log or ship."""

    trace_version: str = TRACE_VERSION
    run_id: str
    status: str
    started_at: str
    finished_at: str | None = None

    total_duration_ms: int
    total_duration_provenance: Measurement = Measurement.MEASURED
    total_cost_usd: float
    total_cost_provenance: Measurement = Measurement.DERIVED
    total_llm_calls: int

    stages: list[StageTrace] = Field(default_factory=list)
    sources: list[SourceTrace] = Field(default_factory=list)

    # Counts, which is what makes a run diagnosable without its content.
    counts: dict[str, int] = Field(default_factory=dict)
    # Metrics the run could not measure, named rather than silently zero.
    unavailable: list[str] = Field(default_factory=list)
    error: str | None = None

    @property
    def stage_order(self) -> list[str]:
        return [s.stage.value for s in self.stages]

    def failed_stages(self) -> list[StageTrace]:
        return [s for s in self.stages if s.status in {StageStatus.FAILED, StageStatus.PARTIAL}]


def _stage_trace(
    metric: StageMetric, status: StageStatus, errors: list[str], warnings: list[str]
) -> StageTrace:
    usage = metric.usage
    return StageTrace(
        stage=metric.stage,
        status=status,
        duration_ms=metric.duration_ms,
        model=metric.model,
        calls=metric.calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        # A stage that made no model call has no tokens to report; that is an
        # absence, not a measurement of zero.
        tokens_provenance=(Measurement.MEASURED if metric.calls else Measurement.UNAVAILABLE),
        cost_usd=metric.cost_usd,
        cost_provenance=(
            Measurement.DERIVED
            if metric.model and is_priced(metric.model)
            else Measurement.UNAVAILABLE
        ),
        errors=errors,
        warnings=warnings,
    )


def build_trace(result: Any) -> RunTraceRecord:
    """Export a finished `RunResult` as a safe trace record.

    Takes the result structurally rather than by import, so the observability
    layer does not become a dependency of the orchestrator it observes.
    """
    trace: RunTrace = result.trace
    outcomes = {outcome.stage: outcome for outcome in result.stages}
    metrics = {metric.stage: metric for metric in trace.stages}

    stages: list[StageTrace] = []
    # Ordered by the run's own stage sequence, so a trace reads in execution
    # order and a reordering would be visible.
    for outcome in result.stages:
        metric = metrics.get(outcome.stage) or StageMetric(stage=outcome.stage, duration_ms=0)
        stages.append(
            _stage_trace(
                metric,
                outcome.status,
                list(getattr(outcome, "errors", []) or []),
                list(getattr(outcome, "warnings", []) or []),
            )
        )
    # A stage that produced a metric but no outcome would otherwise vanish.
    for stage, metric in metrics.items():
        if stage not in outcomes:
            stages.append(_stage_trace(metric, StageStatus.PASSED, [], []))

    sources = _source_traces(result)
    counts = _counts(result)
    unavailable = [
        f"{s.stage.value}.tokens" for s in stages if s.tokens_provenance is Measurement.UNAVAILABLE
    ]

    return RunTraceRecord(
        run_id=result.run_id,
        status=result.status.value,
        started_at=trace.started_at.isoformat(),
        finished_at=trace.finished_at.isoformat() if trace.finished_at else None,
        total_duration_ms=trace.total_duration_ms,
        total_cost_usd=result.total_cost_usd,
        total_llm_calls=result.total_llm_calls,
        stages=stages,
        sources=sources,
        counts=counts,
        unavailable=unavailable,
        error=result.error,
    )


def _source_traces(result: Any) -> list[SourceTrace]:
    """Identify each source by hash, never by content."""
    from app.core.security import detect_injection_signals
    from app.pipeline.validate import classify_credibility

    processed = result.processed
    if processed is None:
        return []

    out: list[SourceTrace] = []
    for source in processed.sources:
        source_id = next(
            (sid for sid, ref in result.sources.items() if str(ref.url) == str(source.final_url)),
            "",
        )
        text = result.source_texts.get(source_id, source.text)
        out.append(
            SourceTrace(
                source_id=source_id,
                domain=source.domain,
                content_hash=source.content_hash,
                text_chars=len(text),
                credibility=classify_credibility(source.domain).value,
                injection_signals=list(detect_injection_signals(text)),
            )
        )
    return out


def _counts(result: Any) -> dict[str, int]:
    """The counts that make a run diagnosable without its content."""
    processed = result.processed
    extraction = result.extraction
    validation = result.claim_validation
    report = result.report

    return {
        "plan_sub_questions": len(result.plan.sub_questions) if result.plan else 0,
        "sources_discovered": len(result.discovery.candidates) if result.discovery else 0,
        "sources_fetched": len(processed.sources) if processed else 0,
        "sources_failed": len(processed.failures) if processed else 0,
        "chunks_indexed": len(processed.chunks) if processed else 0,
        "chunks_retrieved": sum(len(v) for v in result.retrieved.values()),
        "evidence_items": len(extraction.items) if extraction else 0,
        "evidence_unlocatable": len(extraction.unlocatable) if extraction else 0,
        "claims": len(result.synthesis.claims) if result.synthesis else 0,
        "citations_verified": validation.verified_count if validation else 0,
        "citations_rejected": validation.rejected_count if validation else 0,
        "claims_supported": result.verified_claims,
        "claims_unknown": len(validation.unknown_claims) if validation else 0,
        "information_gaps": len(report.information_gaps) if report else 0,
    }
