# ADR-005: No agent framework

**Status:** Accepted · **Date:** 2026-10-08

## Context
LangChain, LlamaIndex, CrewAI and similar frameworks offer prebuilt chains,
retrievers and agent loops that would cover much of this pipeline.

## Decision
No agent framework. The pipeline is ten explicit stages of plain Python in
`app/pipeline/`, orchestrated by a sequential function. The one genuinely
agentic step — stage 2, DISCOVER — uses the Anthropic SDK's own
`client.beta.messages.tool_runner`, not a third-party loop.

## Alternatives considered
- **LangChain / LCEL.** Rejected. The research workflow is a fixed, known
  sequence; expressing it as a framework graph adds an abstraction layer
  without removing a decision. It also obscures exactly the engineering this
  project exists to demonstrate: when a stage misbehaves, the stack trace
  should land in a 40-line function, not in framework internals.
- **LlamaIndex.** Rejected. Strongest where it owns ingestion and indexing of a
  persistent corpus — which, per ADR-004, this system does not have.
- **A hand-rolled `while stop_reason == "tool_use"` loop.** Rejected for stage
  2 specifically: the SDK's tool runner already implements that loop correctly,
  with per-turn hooks for approval and logging. Rewriting it would be
  reinventing a supported primitive — the opposite of the point.

## Consequences
**Accepted cost:** no free swappable retrievers, no community integrations, and
anything a framework provides out of the box must be written here. The
retrieval layer, RRF fusion and chunking are therefore first-party code
(~300 lines total).

**Benefit:** every stage is independently unit-testable with no framework
scaffolding, dependency surface stays small, and upgrade risk is limited to the
Anthropic SDK.

**Revisit when:** the pipeline needs genuinely dynamic graph topology decided
at runtime — which would be a different product.
