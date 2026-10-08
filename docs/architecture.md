# Architecture

**Status:** approved 2026-10-08, implementation in progress.
This document is the design of record. Decisions live in
[`decisions/`](decisions/); this file explains how they fit together.

---

## 1. The problem being engineered

A research question deserves an answer whose provenance can be audited. Two
failure modes make general-purpose chat unusable for decisions that matter:

1. **Unverifiable citations.** Nothing links an assertion to the words in a
   source, so a fabricated citation is indistinguishable from a real one.
2. **Silent absence of evidence.** A question the sources cannot answer is
   answered anyway, in the same confident register as a well-supported one.

Professional research handles both with method: provenance grading,
corroboration across independent sources, confidence stated with its basis, and
explicit naming of what could not be determined. This system implements that
method as code.

## 2. Shape of the system

One FastAPI process. One deterministic pipeline. One model-driven step.
No microservices, no orchestration framework, no queue, no auth, no frontend
beyond a demo page.

```
POST /research ──► ResearchRun persisted, id returned immediately
                   (background execution; GET /research/{id} polls)
      │
 1 PLAN         structured output → ResearchPlan (sub-questions, source types)
 2 DISCOVER     ◄── the ONE model-driven step: tool runner loop
 3 PROCESS      fetch → SSRF guard → extract → sanitise → chunk
                                            ▲ untrusted-content boundary
 4 RETRIEVE     dense (sqlite-vec | pgvector) + BM25 (FTS5) → RRF fusion
 5 EXTRACT      per-chunk EvidenceItem (Haiku, structured outputs, strict tools)
 6 SYNTHESISE   native document citations ON → cited prose (Opus)
 7 VALIDATE     ⚙ no LLM — citation offsets verified; credibility tiers;
                   independent corroboration counted; confidence derived;
                   unsupported claims → UNKNOWN
 8 ASSESS       ⚙ no LLM — information gaps
 9 REPORT       structured ResearchReport
10 METRICS      per-stage tokens / cost / latency → RunTrace
```

Each stage is a typed function with explicit inputs and outputs, separately
testable, separately timed and costed. A failure localises to one named stage
through `PipelineStageError`.

### Why only one agentic step

Nine stages have a fixed, knowable sequence. Only discovery is genuinely
open-ended — how many searches, which queries, whether a result is worth
fetching — because it depends on what the web returns. Making the other nine
model-driven would trade determinism for nothing. See
[ADR-009](decisions/ADR-009-deterministic-pipeline.md) and
[ADR-005](decisions/ADR-005-no-agent-framework.md).

### Why stages 7 and 8 contain no model call

They produce the system's quality guarantees. A guarantee that depends on the
component under audit is not a guarantee. See
[ADR-002](decisions/ADR-002-deterministic-citation-verification.md).

## 3. The constraint that shapes synthesis

Native document citations give `cited_text` plus `char_location` offsets, which
is what makes citation correctness verifiable in code. They are also
**incompatible with `output_config.format`** — sending both returns a 400.

Synthesis therefore cannot be cited *and* structured in one call. Hence the
two-pass design: **stage 6** produces cited prose, **stage 9** structures it.
Cost: roughly one extra model call per run. Accepted, because the alternative
is unverifiable citations. [ADR-001](decisions/ADR-001-native-citations.md).

## 4. Data flow

```
ResearchRequest ─┬─► ResearchPlan ──► SourceCandidate[] ──► FetchedSource[]
                 │                                               │
                 │                                          Chunk[] ──┐
                 │                                                    ▼
                 │                              RetrievedChunk[] ◄── hybrid search
                 │                                       │
                 │                               EvidenceItem[]
                 │                                       │
                 └──────────────────────────► CitedSynthesis
                                                         │
                                           ┌─────────────┴─────────────┐
                                     Claim[] (verified)        information_gaps[]
                                           └─────────────┬─────────────┘
                                                   ResearchReport + RunTrace
```

Every `Claim` carries: the sentence, ≥1 verified `Citation`, the `SourceRef`s
it rests on, a `corroboration` count, and a derived `confidence`. A claim whose
citations fail verification becomes UNKNOWN rather than being dropped — a
deleted claim is invisible, a marked one is auditable.

## 5. RAG architecture

**The corpus is per-question and ephemeral.** Each run discovers its own
sources, indexes them, answers, and discards the index. This is not a growing
knowledge base, which is what justifies SQLite over a vector service
([ADR-004](decisions/ADR-004-vector-store.md)).

| Component | Choice | Notes |
|---|---|---|
| Chunking | structure-aware, ~512 tokens, 64 overlap | Splits on headings/paragraphs before falling back to token windows; **character offsets into the original document are preserved**, because citation verification depends on them |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed (ONNX) | Local, $0 per eval run ([ADR-003](decisions/ADR-003-local-embeddings.md)) |
| Dense index | sqlite-vec locally, pgvector deployed | Behind one `VectorStore` protocol |
| Lexical index | SQLite FTS5 (BM25) | Built in; no extra dependency |
| Fusion | Reciprocal Rank Fusion, `k=60` | ~20 lines; no dependency |

**Why hybrid rather than dense alone:** research questions carry proper nouns,
tickers, statute numbers and figures. Dense retrieval is weak on exact tokens
and strong on paraphrase; BM25 is the reverse. RRF needs no score
normalisation between the two, which is why it is preferred to a weighted sum.

## 6. Tool / function-calling architecture

`app/tools/registry.py` is the **single source of truth** for tool
definitions, consumed by two callers:

```
                  ┌──────────────────────────┐
                  │  app/tools/registry.py   │
                  └────────┬────────┬────────┘
                           │        │
        stage 2 tool runner│        │app/mcp/server.py
                           ▼        ▼
                    internal agent   external MCP clients
```

That shared registry is what makes MCP a real capability rather than a
checkbox: the same definitions, validation and execution path serve both. Tools
use `strict: true` with `additionalProperties: false`, so arguments are
schema-valid by construction rather than by retry.

Forced tool choice (`tool_choice: any` / `tool`) returns a 400 on Opus 5.5, so
tool use is steered with `auto` plus an explicit instruction — not by forcing.

## 7. Agentic workflow (stage 2)

`client.beta.messages.tool_runner` drives the loop; `@beta_tool`-decorated
functions supply search and fetch. Bounded by `max_sources_per_run` and
`max_cost_usd_per_run`. Per-turn hooks provide logging and the cost check, so
the loop cannot run away. Writing the `while stop_reason == "tool_use"` loop by
hand would reinvent a supported primitive.

## 8. MCP integration point

`app/mcp/server.py` (FastMCP) exposes the registry outward:

| Tool | Purpose |
|---|---|
| `search_sources` | Discover candidate sources for a query |
| `fetch_and_index` | Fetch, sanitise, chunk and index a URL |
| `retrieve_evidence` | Hybrid retrieval over an indexed run |
| `run_research` | The full pipeline, returning a structured report |

Arriving at M8 — after the evaluation harness exists, so the effect of adding
surface area is measurable rather than assumed.

## 9. Evaluation architecture

| Metric | Method | Deterministic? |
|---|---|---|
| Retrieval relevance | `recall@k`, MRR, nDCG vs annotated relevant sources | ✅ |
| Source coverage | fraction of annotated sources discovered | ✅ |
| **Citation correctness** | offsets re-verified against stored source text | ✅ |
| Unsupported-claim rate | claims with no verified citation ÷ total claims | ✅ |
| Tool selection | expected vs actual tool sequence on fixture questions | ✅ |
| Answer relevance | LLM judge (Haiku), rubric-scored | ✗ |
| Report quality | LLM judge against a written rubric | ✗ |

Five of seven are deterministic. Golden set: 20–30 questions in
`evals/datasets/golden_v1.jsonl` with annotated relevant sources, **annotated
before retrieval is tuned** to avoid grading retrieval against itself. A
held-out split guards against overfitting prompts to the measured set.

Results land in `evals/results/` with a timestamp and the git SHA. **No metric
appears anywhere in this repository that was not produced by an actual run.**

## 10. Testing strategy

| Layer | Scope |
|---|---|
| `tests/unit/` | Pure functions: chunking offsets, RRF ordering, credibility rules, corroboration counting, confidence derivation, config validation |
| `tests/integration/` | API endpoints, pipeline stages with mocked transport (`respx`), full run against recorded fixtures |
| `tests/security/` | SSRF corpus, prompt-injection corpus, log-redaction assertions |
| `-m live` | Real billable calls. Deselected by default; CI has no key, so CI cannot spend money |

The test that matters most: **a deliberately fabricated citation must be caught
by the verifier.** It is in the Definition of Done, not the backlog.

## 11. Security considerations

**Prompt injection is the central risk.** The system reads attacker-controllable
web pages and feeds them to a model that also receives instructions.

| Control | Implementation |
|---|---|
| Untrusted-content boundary | Fetched text never enters the `system` prompt. It is wrapped in explicit delimiters and labelled untrusted data |
| Separate operator channel | Mid-conversation `{"role": "system"}` messages carry operator instructions, keeping them structurally distinct from fetched content — and preserving the prompt cache |
| Domain control | `allowed_domains` / `blocked_domains` on the server-side web tools: an enforced control, not a prompt request |
| SSRF defence | `app/core/security.py`: scheme allowlist, DNS-resolve-then-check against private/link-local/loopback ranges, redirect re-validation |
| Output constraint | Extraction returns schema-constrained objects, so injected prose cannot become a free-form instruction downstream |
| Verification backstop | An injection that fabricates a citation is caught deterministically by stage 7 |
| Secret handling | `SecretStr` throughout, two-layer log redaction, `.env` gitignored, no key in CI |
| Cost guard | `max_cost_usd_per_run` aborts a runaway run |

Injection resistance is a *mitigation*, not a solved problem. The honest claim
is defence in depth plus a deterministic backstop — stated as such in the
README.

## 12. Observability and cost tracking

Per run, a `RunTrace` records for each stage: wall-clock duration, model, and
`input_tokens` / `output_tokens` / `cache_read_input_tokens` /
`cache_creation_input_tokens` separately — cache reads are 20× cheaper and
collapsing them into one number would hide the thing worth knowing.

Cost is computed against a per-model price table in `app/llm/pricing.py`.
Logs are structured JSON via structlog, with secret redaction enforced by test.
OpenTelemetry is deliberately not included: at one process it would be
ceremony.

## 13. Deployment

Render, from the repository, with managed Postgres for the pgvector backend.
A `Dockerfile` (multi-stage, non-root, healthchecked) ships in the repo for
portability; Docker is not required locally, which matters because it is not
installed on the development machine.

Configuration is environment variables only. `/health` is liveness and depends
on nothing external; `/ready` reports which dependencies are configured.

## 14. Milestones

| M | Scope | Exit criterion |
|---|---|---|
| 0 | Foundation | CI green: ruff, mypy strict, real tests on real behaviour |
| 1 | API + planning | A real `ResearchPlan` from a real question |
| 2 | Source ingestion | Real fetches; SSRF tests prove private IPs and `file://` blocked |
| 3 | Retrieval + retrieval eval | Measured retrieval baseline on an annotated fixture |
| 4 | Evidence + citations | **A fabricated citation is caught by a test** |
| 5 | Orchestration | One full end-to-end run |
| 6 | Report + UNKNOWN | A thin-source question yields explicit UNKNOWNs |
| 7 | Evaluation | Committed baseline + one real improvement cycle |
| 8 | MCP | External client calls the tools; no eval regression |
| 9 | Security / hardening | Injection + SSRF + redaction suites pass; ≥80% coverage |
| 10 | Deployment + docs | Live URL serving a real run |

## 15. Definition of Done

**Functional** — a real question returns a structured report with ranked
sub-questions, ≥3 distinct sources, evidence bound to sources by character
offset, explicit UNKNOWNs, and a verified citation for every factual assertion.

**Verified, not asserted** — every citation passes the verifier; a fabricated
citation is caught by a test; `cache_read_input_tokens > 0` is asserted; SSRF
and injection suites pass.

**Measured** — the golden set runs and emits all seven metrics; a baseline is
committed; one documented improvement cycle shows before/after on the same
split. No unmeasured number appears anywhere.

**Engineering** — ≥80% coverage on `app/`, mypy strict clean, ruff clean, CI
green; typed error taxonomy; no secret in any log line, test-enforced.

**Operable** — per-stage latency and cost recorded per run and exposed on the
run detail response; `docs/cost-latency.md` reports real figures.

**MCP** — ≥4 tools exposed from the shared registry, verified from an external
client, with no eval regression.

**Portfolio** — README with the capability→evidence map; architecture diagram;
ADRs; `evaluation.md` with real numbers; `failure-analysis.md` with real
failures; reproducible demo; one example report; honest limitations. No claim
about professional experience anywhere — the code is the evidence.
