"""Pipeline orchestration.

This module coordinates. It contains no retrieval, no fetching, no SSRF
logic, no extraction, no verification and no LLM handling — each of those
lives in the module that owns it and was validated there. What is here is the
sequencing, the seams between stage contracts, and the decision about when a
failure is fatal.

Two kinds of failure, kept apart deliberately:

- **Source-level failure** — one URL 404s, is blocked by the SSRF guard, or
  yields no text. The run continues with fewer sources and records the gap.
  A research answer from six of eight sources is a usable result; aborting
  because one link rotted is not.
- **Stage-level failure** — a stage has no viable input, or its own call
  fails. Downstream stages are SKIPPED rather than run on nothing, and the
  run reports where it stopped. A stage never reports success on empty input.

One seam needs an adapter, and it is the only transformation here: stage 4
returns `dict[str, list[ScoredChunk]]` while stage 5 takes
`dict[str, list[Chunk]]`. The orchestrator unwraps `.chunk`, preserving
offsets and provenance untouched. Changing either stage's signature to avoid
this would be the orchestrator leaking into stages that are already verified.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import structlog

from app.core.config import Settings
from app.core.errors import CostCeilingExceededError, PipelineStageError
from app.llm.client import LLMClient
from app.pipeline.assess import run_assess_stage
from app.pipeline.discover import DiscoveryResult, run_discover_stage
from app.pipeline.extract import ExtractionResult, run_extract_stage
from app.pipeline.plan import run_plan_stage
from app.pipeline.process import ProcessResult, run_process_stage
from app.pipeline.retrieve import run_retrieve_stage
from app.pipeline.synthesize import SynthesisResult, run_synthesise_stage
from app.pipeline.validate import (
    CitationVerifier,
    ValidationResult,
    classify_credibility,
    run_validate_stage,
)
from app.retrieval.embeddings import Embedder
from app.schemas.evidence import (
    EvidenceItem,
    SourceRef,
    source_id_for,
)
from app.schemas.report import ResearchReport
from app.schemas.research import ResearchPlan, ResearchRequest, RunStatus, new_run_id
from app.schemas.runs import RunTrace, Stage, StageMetric, StageOutcome, StageStatus
from app.schemas.source import Chunk, FetchedSource
from app.tools.fetch import SourceFetcher
from app.tools.search import SourceProvider

logger = structlog.get_logger(__name__)


# Called after each stage is recorded, with the outcome and the run's status
# at that moment. Synchronous and best-effort by design: the orchestrator owns
# the pipeline, and an observer must not be able to slow it, reorder it or
# fail it. A caller needing to await something should hand off, not block.
StageObserver = Callable[[StageOutcome, RunStatus], None]


@dataclass(slots=True)
class RunResult:
    """Everything one research run produced, and how it got there."""

    run_id: str
    request: ResearchRequest
    status: RunStatus
    trace: RunTrace
    stages: list[StageOutcome] = field(default_factory=list)

    # Stage outputs, retained so a run is inspectable end to end.
    plan: ResearchPlan | None = None
    discovery: DiscoveryResult | None = None
    processed: ProcessResult | None = None
    retrieved: dict[str, list[Chunk]] = field(default_factory=dict)
    extraction: ExtractionResult | None = None
    evidence_validation: ValidationResult | None = None
    synthesis: SynthesisResult | None = None
    claim_validation: ValidationResult | None = None
    report: ResearchReport | None = None

    # Provenance registry: the canonical source identity and text every
    # citation is verified against.
    sources: dict[str, SourceRef] = field(default_factory=dict)
    source_texts: dict[str, str] = field(default_factory=dict)

    error: str | None = None

    # -- derived views ----------------------------------------------------

    @property
    def total_cost_usd(self) -> float:
        return self.trace.total_cost_usd

    @property
    def total_llm_calls(self) -> int:
        return sum(s.metric.calls for s in self.stages if s.metric)

    @property
    def verified_claims(self) -> int:
        return len(self.claim_validation.supported_claims) if self.claim_validation else 0

    def stage(self, stage: Stage) -> StageOutcome | None:
        return next((s for s in self.stages if s.stage is stage), None)

    def summary(self) -> dict[str, object]:
        """Measured facts about the run. No prose, no judgements."""
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "question": self.request.question,
            "plan_sub_questions": len(self.plan.sub_questions) if self.plan else 0,
            "queries_issued": len(self.discovery.queries_issued) if self.discovery else 0,
            "sources_discovered": len(self.discovery.candidates) if self.discovery else 0,
            "sources_fetched": len(self.processed.sources) if self.processed else 0,
            "sources_failed": len(self.processed.failures) if self.processed else 0,
            "chunks_indexed": len(self.processed.chunks) if self.processed else 0,
            "chunks_retrieved": sum(len(v) for v in self.retrieved.values()),
            "evidence_items": len(self.extraction.items) if self.extraction else 0,
            "evidence_unlocatable": (len(self.extraction.unlocatable) if self.extraction else 0),
            "evidence_verified": (
                self.evidence_validation.verified_count if self.evidence_validation else 0
            ),
            "claims": len(self.synthesis.claims) if self.synthesis else 0,
            "citations_verified": (
                self.claim_validation.verified_count if self.claim_validation else 0
            ),
            "citations_rejected": (
                self.claim_validation.rejected_count if self.claim_validation else 0
            ),
            "claims_supported": self.verified_claims,
            "claims_unknown": (
                len(self.claim_validation.unknown_claims) if self.claim_validation else 0
            ),
            "total_llm_calls": self.total_llm_calls,
            "total_cost_usd": self.total_cost_usd,
            "total_duration_ms": self.trace.total_duration_ms,
            "cost_by_stage": self.trace.cost_by_stage(),
            "stages": {s.stage.value: s.status.value for s in self.stages},
        }


class ResearchOrchestrator:
    """Runs the pipeline. Owns sequencing, not behaviour."""

    def __init__(
        self,
        *,
        client: LLMClient,
        provider: SourceProvider,
        fetcher: SourceFetcher,
        embedder: Embedder,
        settings: Settings,
        on_stage: StageObserver | None = None,
    ) -> None:
        self._client = client
        self._provider = provider
        self._fetcher = fetcher
        self._embedder = embedder
        self._settings = settings
        # Keyword-only with a default, so every existing caller is unaffected.
        self._on_stage = on_stage

    # -- helpers ----------------------------------------------------------

    def _record(
        self,
        result: RunResult,
        stage: Stage,
        status: StageStatus,
        *,
        metric: StageMetric | None = None,
        inputs: int = 0,
        outputs: int = 0,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> StageOutcome:
        outcome = StageOutcome(
            stage=stage,
            status=status,
            metric=metric,
            inputs=inputs,
            outputs=outputs,
            errors=errors or [],
            warnings=warnings or [],
        )
        result.stages.append(outcome)
        if outcome.metric is not None:
            result.trace.stages.append(outcome.metric)
        self._notify(outcome, result.status)
        return outcome

    def _notify(self, outcome: StageOutcome, status: RunStatus) -> None:
        """Tell the observer, and never let it break the run.

        A caller's logging or progress write failing is not a research
        failure, so it is logged and swallowed here rather than propagated.
        """
        if self._on_stage is None:
            return
        try:
            self._on_stage(outcome, status)
        except Exception:  # noqa: BLE001 - an observer must not fail a run
            logger.exception("stage_observer_failed", stage=outcome.stage.value)

    def _record_raised_stage(self, result: RunResult, exc: PipelineStageError) -> None:
        """Mark the stage that raised as FAILED, and the rest as SKIPPED.

        Stages that complete record themselves; a stage that raises never
        reaches its own `_record` call, which is how a failed stage could
        otherwise be missing from the trace entirely.
        """
        try:
            stage = Stage(exc.stage)
        except ValueError:
            # An unrecognised stage name is not worth guessing at.
            return
        if any(outcome.stage is stage for outcome in result.stages):
            return
        self._record(result, stage, StageStatus.FAILED, errors=[exc.message])
        self._skip_remaining(result, stage)

    def _skip_remaining(self, result: RunResult, after: Stage) -> None:
        """Mark unreached stages SKIPPED, so a run shows where it stopped.

        SKIPPED is not FAILED: these stages did not fail, they never ran.
        Conflating the two would misattribute the cause of a short run.
        """
        order = [
            Stage.PLAN,
            Stage.DISCOVER,
            Stage.PROCESS,
            Stage.RETRIEVE,
            Stage.EXTRACT,
            Stage.VALIDATE,
            Stage.SYNTHESISE,
            Stage.REPORT,
        ]
        reached = order.index(after)
        recorded = {s.stage for s in result.stages}
        for stage in order[reached + 1 :]:
            if stage not in recorded:
                self._record(result, stage, StageStatus.SKIPPED)

    def _check_budget(self, result: RunResult) -> None:
        """Abort rather than overspend on a pathological run."""
        spent = result.trace.total_cost_usd
        ceiling = self._settings.max_cost_usd_per_run
        if spent > ceiling:
            raise CostCeilingExceededError(
                f"Run has spent ${spent:.4f}, exceeding the ${ceiling:.2f} ceiling.",
                details={"spent_usd": spent, "ceiling_usd": ceiling},
            )

    def _register_sources(self, result: RunResult, sources: list[FetchedSource]) -> None:
        """Build the provenance registry the verifier works from.

        Credibility is assigned here by the same rule stage 7 uses, from the
        fetched domain. The canonical text is stored exactly as stage 3
        produced it — not re-normalised, because every chunk offset and every
        citation offset indexes into this precise string.
        """
        for source in sources:
            source_id = source_id_for(str(source.final_url))
            result.sources[source_id] = SourceRef(
                source_id=source_id,
                url=str(source.final_url),
                title=source.title,
                domain=source.domain,
                content_hash=source.content_hash,
                credibility=classify_credibility(source.domain),
                publisher=source.domain,
            )
            result.source_texts[source_id] = source.text

    # -- the run ----------------------------------------------------------

    async def run(self, request: ResearchRequest, *, run_id: str | None = None) -> RunResult:
        """Execute the pipeline for one request.

        Returns a `RunResult` in every case, including failure: a caller needs
        the partial outputs and the stage statuses to understand what
        happened, and an exception would discard both.

        `run_id` adopts an id the caller already owns. The HTTP API creates the
        run row before scheduling the pipeline, so minting a second id here
        would key the trace to an id the caller never sees. Omitted (MCP, the
        evaluation runner, tests), a fresh id is generated as before.
        """
        run_id = run_id or new_run_id()
        result = RunResult(
            run_id=run_id,
            request=request,
            status=RunStatus.PENDING,
            trace=RunTrace(run_id=run_id),
        )
        started = time.perf_counter()
        logger.info("run_start", run_id=run_id, question=request.question[:200])

        try:
            await self._execute(result, request)
        except PipelineStageError as exc:
            result.status = RunStatus.FAILED
            result.error = f"{exc.stage}: {exc.message}"
            # Record the stage that raised. Without this a stage that threw
            # left no outcome at all, so the run reported a failure that no
            # trace could attribute to a stage (M9, F-015).
            self._record_raised_stage(result, exc)
            logger.warning("run_failed", run_id=run_id, stage=exc.stage)
        except CostCeilingExceededError as exc:
            result.status = RunStatus.FAILED
            result.error = exc.message
            logger.warning("run_cost_ceiling", run_id=run_id, details=exc.details)
        except Exception as exc:  # noqa: BLE001 - a run must always return a result
            result.status = RunStatus.FAILED
            result.error = "Internal pipeline error."
            logger.exception("run_error", run_id=run_id, exc_type=type(exc).__name__)

        # Stage 8 runs even when the pipeline stopped early. A failed run
        # still has information gaps, and a reader needs to know why it
        # produced nothing rather than receiving silence.
        if result.plan is not None and result.report is None:
            try:
                report, assess_metric = run_assess_stage(result)
                result.report = report
                self._record(
                    result,
                    Stage.ASSESS,
                    StageStatus.PASSED,
                    metric=assess_metric,
                    inputs=len(result.plan.sub_questions),
                    outputs=len(report.sub_questions),
                    warnings=[
                        f"gap [{g.sub_question_id}]: {g.cause.value}"
                        for g in report.information_gaps
                    ],
                )
            except Exception as exc:  # noqa: BLE001 - assessment must not mask a run
                logger.exception("assess_failed", run_id=run_id)
                self._record(
                    result,
                    Stage.ASSESS,
                    StageStatus.FAILED,
                    errors=[f"{type(exc).__name__}: {exc}"[:200]],
                )
        logger.info(
            "run_complete",
            run_id=run_id,
            status=result.status.value,
            duration_ms=int((time.perf_counter() - started) * 1000),
            **{
                k: v
                for k, v in result.summary().items()
                if k in {"sources_fetched", "evidence_items", "claims_supported", "total_cost_usd"}
            },
        )
        return result

    async def _execute(self, result: RunResult, request: ResearchRequest) -> None:
        settings = self._settings

        # -- 1 PLAN -------------------------------------------------------
        result.status = RunStatus.PLANNING
        plan, metric = await run_plan_stage(request, client=self._client, settings=settings)
        result.plan = plan
        self._record(
            result,
            Stage.PLAN,
            StageStatus.PASSED,
            metric=metric,
            inputs=1,
            outputs=len(plan.sub_questions),
        )
        self._check_budget(result)

        # -- 2 DISCOVER ---------------------------------------------------
        result.status = RunStatus.DISCOVERING
        discovery, metric = await run_discover_stage(
            plan, provider=self._provider, client=self._client, settings=settings
        )
        result.discovery = discovery
        discover_status = (
            StageStatus.PASSED
            if discovery.candidates and not discovery.search_failures
            else StageStatus.PARTIAL
            if discovery.candidates
            else StageStatus.FAILED
        )
        self._record(
            result,
            Stage.DISCOVER,
            discover_status,
            metric=metric,
            inputs=len(plan.sub_questions),
            outputs=len(discovery.candidates),
            warnings=discovery.search_failures,
        )
        if not discovery.candidates:
            # No viable input for stage 3. Reported as a structured failure
            # rather than running the rest of the pipeline on nothing.
            result.status = RunStatus.FAILED
            result.error = "discover: no candidate sources were found."
            self._skip_remaining(result, Stage.DISCOVER)
            return
        self._check_budget(result)

        # -- 3 PROCESS (fetch, extract text, chunk) -----------------------
        result.status = RunStatus.PROCESSING
        processed, metric = await run_process_stage(
            discovery.candidates, fetcher=self._fetcher, settings=settings
        )
        result.processed = processed
        self._register_sources(result, processed.sources)
        process_status = (
            StageStatus.PASSED
            if processed.sources and not processed.failures
            else StageStatus.PARTIAL
            if processed.sources
            else StageStatus.FAILED
        )
        self._record(
            result,
            Stage.PROCESS,
            process_status,
            metric=metric,
            inputs=len(discovery.candidates),
            outputs=len(processed.sources),
            # Source-level failures are warnings: the run continues.
            warnings=[f"{f.url}: {f.code}" for f in processed.failures],
        )
        if not processed.sources:
            result.status = RunStatus.FAILED
            result.error = "process: no source could be fetched."
            self._skip_remaining(result, Stage.PROCESS)
            return

        # -- 4 RETRIEVE ---------------------------------------------------
        result.status = RunStatus.RETRIEVING
        scored, metric = await run_retrieve_stage(
            plan, processed.chunks, embedder=self._embedder, settings=settings
        )
        # The one adapter in this module: ScoredChunk -> Chunk, offsets and
        # provenance carried through untouched.
        result.retrieved = {
            sub_question_id: [s.chunk for s in scored_chunks]
            for sub_question_id, scored_chunks in scored.items()
        }
        retrieved_total = sum(len(v) for v in result.retrieved.values())
        self._record(
            result,
            Stage.RETRIEVE,
            StageStatus.PASSED if retrieved_total else StageStatus.FAILED,
            metric=metric,
            inputs=len(processed.chunks),
            outputs=retrieved_total,
        )
        if not retrieved_total:
            result.status = RunStatus.FAILED
            result.error = "retrieve: no chunk was retrieved for any sub-question."
            self._skip_remaining(result, Stage.RETRIEVE)
            return

        # -- 5 EXTRACT ----------------------------------------------------
        result.status = RunStatus.EXTRACTING
        sub_questions = {sq.id: sq.question for sq in plan.ordered()}
        extraction, metric = await run_extract_stage(
            result.retrieved, sub_questions, client=self._client, settings=settings
        )
        result.extraction = extraction
        extract_status = (
            StageStatus.PASSED
            if extraction.items and not extraction.unlocatable and not extraction.failures
            else StageStatus.PARTIAL
            if extraction.items
            else StageStatus.FAILED
        )
        self._record(
            result,
            Stage.EXTRACT,
            extract_status,
            metric=metric,
            inputs=retrieved_total,
            outputs=len(extraction.items),
            warnings=[
                *(f"unlocatable quote: {q[:80]}" for q, _ in extraction.unlocatable),
                # A chunk whose extraction call failed is a warning, not a
                # fatal error: the run continues on the chunks that worked.
                *(f"chunk extraction failed: {f[:80]}" for f in extraction.failures),
            ],
        )
        if not extraction.items:
            result.status = RunStatus.FAILED
            result.error = "extract: no evidence was found in any retrieved chunk."
            self._skip_remaining(result, Stage.EXTRACT)
            return
        self._check_budget(result)

        # -- 6 VERIFY the extracted evidence ------------------------------
        # Runs before synthesis so synthesis can only see sources whose
        # evidence actually verified.
        result.status = RunStatus.VALIDATING
        verifier = CitationVerifier(result.sources, result.source_texts)
        evidence_result = ValidationResult()
        verified_evidence: list[EvidenceItem] = []
        for item in extraction.items:
            verdict = verifier.verify_evidence(item)
            evidence_result.verdicts.append(verdict)
            if verdict.ok:
                verified_evidence.append(item)
        result.evidence_validation = evidence_result

        self._record(
            result,
            Stage.VALIDATE,
            StageStatus.PASSED
            if verified_evidence and not evidence_result.rejected_count
            else StageStatus.PARTIAL
            if verified_evidence
            else StageStatus.FAILED,
            inputs=len(extraction.items),
            outputs=len(verified_evidence),
            errors=[
                f"{v.failure.value if v.failure else 'rejected'}: {v.detail[:120]}"
                for v in evidence_result.verdicts
                if not v.ok
            ],
        )
        if not verified_evidence:
            result.status = RunStatus.FAILED
            result.error = "validate: no extracted evidence survived verification."
            self._skip_remaining(result, Stage.VALIDATE)
            return

        # -- 7 SYNTHESISE -------------------------------------------------
        result.status = RunStatus.SYNTHESISING
        sources_by_sub_question = self._sources_for_synthesis(verified_evidence, processed.sources)
        synthesis, metric = await run_synthesise_stage(
            plan, sources_by_sub_question, client=self._client, settings=settings
        )
        result.synthesis = synthesis
        self._record(
            result,
            Stage.SYNTHESISE,
            StageStatus.PASSED
            if synthesis.claims and not synthesis.unanswered
            else StageStatus.PARTIAL
            if synthesis.claims
            else StageStatus.FAILED,
            metric=metric,
            inputs=len(verified_evidence),
            outputs=len(synthesis.claims),
            warnings=[f"unanswered: {sq}" for sq in synthesis.unanswered],
        )

        # -- 8 VERIFY the synthesised claims ------------------------------
        # The same verifier, re-slicing the same stored text. Synthesis
        # cannot promote its own citations; this is the only thing that sets
        # a claim's status.
        evidence_by_id = {item.evidence_id: item for item in verified_evidence}
        claim_result, claim_metric = run_validate_stage(
            synthesis.claims,
            sources=result.sources,
            texts=result.source_texts,
            evidence=evidence_by_id,
        )
        result.claim_validation = claim_result
        # Recorded under REPORT because this is where verified claims are
        # assembled (architecture stage 9); VALIDATE above is the pre-synthesis
        # evidence gate. Two passes, one verifier, same stored text.
        self._record(
            result,
            Stage.REPORT,
            StageStatus.PASSED
            if claim_result.supported_claims and not claim_result.rejected_count
            else StageStatus.PARTIAL
            if claim_result.supported_claims
            else StageStatus.FAILED,
            metric=claim_metric,
            inputs=len(synthesis.claims),
            outputs=len(claim_result.supported_claims),
            errors=[f"{code}: {count}" for code, count in claim_result.failure_counts.items()],
        )

        result.status = RunStatus.COMPLETED if claim_result.supported_claims else RunStatus.FAILED
        if not claim_result.supported_claims:
            result.error = "No synthesised claim survived citation verification."

    @staticmethod
    def _sources_for_synthesis(
        verified_evidence: list[EvidenceItem], sources: list[FetchedSource]
    ) -> dict[str, list[FetchedSource]]:
        """Map each sub-question to the sources whose evidence verified.

        Synthesis may only cite sources that reached it this way, which is how
        "synthesis sees only verified evidence" is enforced through the
        contract rather than by instruction.
        """
        by_id = {source_id_for(str(s.final_url)): s for s in sources}
        grouped: dict[str, list[FetchedSource]] = {}
        for item in verified_evidence:
            source = by_id.get(item.source_id)
            if source is None:  # pragma: no cover - registry is built from these
                continue
            bucket = grouped.setdefault(item.sub_question_id, [])
            if source not in bucket:
                bucket.append(source)
        return grouped
