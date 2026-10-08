# What would change at larger scale

The current design is deliberately single-process, single-user, and
ephemeral-corpus. This file names what breaks first and what replaces it —
the honest answer to "what would you do differently at scale".

## 1. Vector store: sqlite-vec → pgvector → dedicated service

**Current:** sqlite-vec locally, pgvector deployed
([ADR-004](decisions/ADR-004-vector-store.md)). Justified because each run
builds and discards its own index.

**Breaks when:** the corpus becomes shared and persistent across runs, or
concurrent runs contend on one file.

**Then:** pgvector with HNSW indexes carries a surprisingly long way. Beyond
that — tens of millions of vectors, or sub-50ms p99 under concurrency — a
dedicated service (Qdrant, Vespa) earns its operational cost. The
`VectorStore` protocol is the seam.

## 2. Embeddings: local ONNX → hosted

**Current:** local `bge-small` ([ADR-003](decisions/ADR-003-local-embeddings.md)).

**Breaks when:** embedding throughput becomes the pipeline bottleneck, or
measured `recall@k` limits report quality.

**Then:** a hosted embedder (Voyage) behind the same `Embedder` protocol, with
a batch-embedding worker. The decision should follow the measured comparison in
M7, not intuition.

## 3. Execution: in-process background task → durable queue

**Current:** FastAPI `BackgroundTasks` with SQLite status polling.

**Breaks immediately under real concurrency.** A process restart loses
in-flight runs, and there is no retry.

**Then:** a durable queue with at-least-once delivery and idempotent stages.
Each stage already has typed inputs and outputs, so stages become queue
consumers with little restructuring — which was part of the reason for making
the pipeline explicit rather than a framework graph.

## 4. Corpus: per-question → shared, cached, deduplicated

**Current:** every run fetches and indexes independently; only a simple fetch
cache is shared.

**Breaks when:** fetch cost and latency dominate, or the same sources are
re-fetched across many runs.

**Then:** a content-addressed document store keyed by normalised URL + content
hash, with an embedding cache keyed by chunk hash. This changes the ADR-004
calculus, because the corpus stops being ephemeral — which is exactly the
condition under which a vector service becomes justified.

## 5. Single-user → multi-tenant

**Current:** no auth, no tenancy, by design.

**Then:** tenant id on every row, row-level isolation, per-tenant rate limits
and cost ceilings, and per-tenant domain allow/deny policy. The cost guard
already exists per run; it would become per tenant per period.

## 6. Cost control: per-run ceiling → budget accounting

**Current:** `MAX_COST_USD_PER_RUN` aborts a runaway run.

**Then:** pre-flight estimation via `count_tokens`, the Batch API (50% cheaper)
for non-interactive runs, longer cache TTLs on hot corpora, and spend
attribution per tenant and per stage.

## 7. Observability: structured logs → traces and metrics

**Current:** structlog JSON plus a `RunTrace` row per run. Adequate for one
process.

**Then:** OpenTelemetry spans per stage, metrics for p50/p95 stage latency,
cache hit rate, citation-verification failure rate, and UNKNOWN rate — that
last one being the quality signal most worth alerting on, since a rising
UNKNOWN rate means retrieval or discovery has regressed.

## 8. What would NOT change

The deterministic pipeline, the single agentic step, deterministic citation
verification, and the no-framework decision. Those are not scale compromises —
they get *more* valuable under load, because they keep behaviour attributable
and cost predictable.
