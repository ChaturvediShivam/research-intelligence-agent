# ADR-009: Deterministic pipeline with exactly one model-driven step

**Status:** Accepted · **Date:** 2026-10-08

## Context
"Agent" invites an architecture of many cooperating LLM-driven actors. That
design is hard to test, hard to cost, and non-deterministic in places where
determinism is available for free.

Examining the research workflow honestly: nine of its ten stages have a known,
fixed sequence. Only source discovery is genuinely open-ended — how many
searches, which queries, whether to fetch a given result — because it depends
on what the web returns.

## Decision
Ten sequential stages in `app/pipeline/orchestrator.py`. Each is a typed
function with explicit inputs and outputs, individually unit-testable, and
individually timed and costed.

**Stage 2 (DISCOVER) is the only model-driven step.** It uses the SDK tool
runner so the model chooses among search/fetch tools until it has enough
sources or hits the configured ceiling.

**Stages 7 and 8 contain no LLM call at all.**

## Alternatives considered
- **Multi-agent (planner / researcher / critic / writer).** Rejected: the
  coordination overhead is real, the determinism loss is real, and the quality
  gain for a fixed workflow is speculative. "Agent" in the project name does
  not oblige the architecture to contain ten of them.
- **Fully deterministic, no agentic step.** Rejected: discovery genuinely
  cannot be pre-scripted — the number of searches needed depends on results.
- **A re-planning loop (plan → research → re-plan).** Deferred, not rejected.
  It needs the evaluation harness (M7) to show that re-planning improves
  measured report quality. Adding it first would be adding unmeasurable
  complexity.

## Consequences
**Accepted cost:** the pipeline cannot adapt its own shape mid-run. A question
needing a fundamentally different sequence is answered by the fixed one, likely
producing information gaps instead of a restructured investigation. That
outcome is at least *visible* (stage 8 names the gaps) rather than silent.

**Benefit:** every stage is testable in isolation, per-stage cost and latency
are attributable, and a failure localises to one named stage via
`PipelineStageError`.

**Revisit when:** the eval shows a class of question the fixed sequence
systematically fails — which is the evidence that would justify a re-planning
loop.
