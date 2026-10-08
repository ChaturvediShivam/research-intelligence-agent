# Research Intelligence Agent

An evidence-first AI research and due-diligence agent. It takes a research
question, discovers and reads sources, extracts evidence, and produces a
structured report in which **every factual claim carries a citation that has
been verified by code against the source text**.

> **Status: in active development.** Milestone progress is tracked in
> [Build status](#build-status) below. Sections marked *not yet measured* have
> no numbers because the measurement has not been run — no figure appears
> anywhere in this repository that was not produced by an actual execution.

---

## The problem

Ask a general-purpose chatbot a research question and you get fluent prose with
plausible citations. Two things are wrong with that for any decision that
matters:

1. **Citations are often unverifiable or wrong.** The text asserts a source
   without a checkable link between the claim and the words in that source.
2. **Absence of evidence is reported as a conclusion.** A question the sources
   cannot answer gets an answer anyway.

Professional research — competitive intelligence, due diligence, AML/KYC —
handles both with discipline a chatbot does not apply: sources are graded by
credibility, claims are corroborated across independent sources, confidence is
stated with its basis, and **information gaps are named explicitly**.

This project engineers that discipline into an AI system.

## What it does differently

| | Typical RAG demo | This system |
|---|---|---|
| Citations | Model asserts them | **Verified in code** against source text at character offsets |
| Unsupported claims | Blend into the prose | Marked **UNKNOWN**, listed separately |
| Source quality | Undifferentiated | **Credibility tier** assigned by rule |
| Corroboration | Not tracked | **Independent-source count** per claim |
| Confidence | Model's adjective, or a fake percentage | **Derived** from corroboration × credibility × verification |
| Unanswerable sub-questions | Silently dropped | Reported as **information gaps** |
| Retrieval quality | Asserted | Measured (recall@k, MRR, nDCG) |

Stages 7 and 8 of the pipeline — the ones producing those guarantees — contain
**no LLM call at all**. The quality properties are code, not prompting.

## Architecture

```
POST /research ──► ResearchRun (id returned immediately; background execution)
      │
 1 PLAN         structured output → ResearchPlan (sub-questions, source types)
 2 DISCOVER     ◄── the ONE model-driven step (Tool Runner selects tools)
 3 PROCESS      fetch → SSRF guard → extract → sanitise → chunk  [untrusted boundary]
 4 RETRIEVE     dense (sqlite-vec | pgvector) + BM25 (FTS5) → RRF fusion
 5 EXTRACT      per-chunk EvidenceItem (structured outputs, strict tools)
 6 SYNTHESISE   Anthropic native document citations ON → cited prose
 7 VALIDATE     ⚙ no LLM: citation offsets verified, credibility tiers,
                   independent corroboration counted, confidence derived,
                   unsupported claims → UNKNOWN
 8 ASSESS       ⚙ no LLM: information gaps
 9 REPORT       structured ResearchReport
10 METRICS      per-stage tokens / cost / latency → RunTrace
```

Ten sequential stages of plain Python. Only stage 2 is model-driven. Full
reasoning, including the constraint that forces the two-pass synthesis, is in
[`docs/architecture.md`](docs/architecture.md).

## Capability → evidence map

Every capability points at the code that implements it and the test or
artifact that proves it. Unchecked rows are not yet built — they are not
claims.

| # | Capability | Implementation | Evidence | Status |
|---|---|---|---|---|
| 1 | Python application engineering | `app/` | mypy strict + ruff clean in CI | ✅ M0 |
| 2 | FastAPI | `app/api/`, `app/main.py` | `tests/integration/test_health.py` | ✅ M0 |
| 3 | Configuration & secret handling | `app/core/config.py` | `tests/unit/test_config.py` | ✅ M0 |
| 4 | Error taxonomy, no internal leakage | `app/core/errors.py` | `tests/unit/test_errors.py` | ✅ M0 |
| 5 | Secret redaction in logs | `app/core/logging.py` | `tests/security/test_log_redaction.py` | ✅ M0 |
| 6 | LLM API integration | `app/llm/client.py` | retry/error tests | ⬜ M1 |
| 7 | Structured outputs | `app/pipeline/plan.py`, `extract.py` | schema-validation tests | ⬜ M1 |
| 8 | Prompt / context engineering | `app/llm/context.py`, `app/llm/prompts/` | `docs/prompt-engineering.md` eval deltas | ⬜ M1→M7 |
| 9 | Source discovery | `app/tools/search.py` | integration tests | ⬜ M2 |
| 10 | SSRF defence | `app/core/security.py` | `tests/security/` | ⬜ M2 |
| 11 | RAG / chunking | `app/retrieval/chunking.py` | unit tests | ⬜ M3 |
| 12 | Embeddings | `app/retrieval/embeddings.py` | determinism + dimension tests | ⬜ M3 |
| 13 | Hybrid retrieval + RRF | `app/retrieval/hybrid.py` | ranking tests | ⬜ M3 |
| 14 | Retrieval evaluation | `app/evaluation/metrics.py` | `evals/results/` | ⬜ M3 |
| 15 | Evidence extraction | `app/pipeline/extract.py` | fixture tests | ⬜ M4 |
| 16 | **Citation verification** | `app/pipeline/validate.py` | **fabricated-citation test** | ⬜ M4 |
| 17 | Credibility + corroboration | `app/pipeline/validate.py` | rule tests | ⬜ M4 |
| 18 | Tool / function calling | `app/tools/registry.py` | `strict: true` schema tests | ⬜ M5 |
| 19 | Agentic orchestration | `app/pipeline/orchestrator.py` | stage tests + ADR-005 | ⬜ M5 |
| 20 | UNKNOWN / information gaps | `app/pipeline/assess.py` | thin-source test | ⬜ M6 |
| 21 | Evaluation harness | `app/evaluation/` | committed baseline | ⬜ M7 |
| 22 | MCP tool layer | `app/mcp/server.py` | external client transcript | ⬜ M8 |
| 23 | Prompt-injection resistance | `app/core/security.py` | injection corpus suite | ⬜ M9 |
| 24 | Observability | `app/observability/trace.py` | per-stage trace assertions | ⬜ M9 |
| 25 | Cost & latency measurement | `app/llm/pricing.py` | `docs/cost-latency.md` | ⬜ M9 |
| 26 | Testing & CI | `tests/`, `.github/workflows/ci.yml` | CI green | ✅ M0 |
| 27 | Deployment | `Dockerfile`, Render | live URL | ⬜ M10 |

## Build status

| Milestone | Scope | Status |
|---|---|---|
| M0 | Foundation: structure, config, logging, errors, CI, ADRs | ✅ |
| M1 | API + research planning | ⬜ |
| M2 | Source ingestion | ⬜ |
| M3 | Retrieval + retrieval evaluation | ⬜ |
| M4 | Evidence + citation verification | ⬜ |
| M5 | Orchestration | ⬜ |
| M6 | Report + UNKNOWN / information gaps | ⬜ |
| M7 | Evaluation | ⬜ |
| M8 | MCP | ⬜ |
| M9 | Security / hardening | ⬜ |
| M10 | Deployment + documentation | ⬜ |

## Running it

Requires [uv](https://docs.astral.sh/uv/). Python 3.13 is installed by uv.

```bash
git clone <this repo> && cd research-intelligence-agent
uv sync --all-extras

cp .env.example .env     # then add your ANTHROPIC_API_KEY
uv run fastapi dev app/main.py
```

The API serves on `http://127.0.0.1:8000`; `/docs` has the OpenAPI UI.
`GET /ready` reports which dependencies are configured and names anything
missing.

### Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest                      # unit + integration + security
uv run pytest -m live              # real API calls — costs money, needs a key
```

Tests never make billable calls unless explicitly selected with `-m live`.

## Documentation

| Document | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | The design and why each stage exists |
| [`docs/decisions/`](docs/decisions/) | ADRs — one per significant decision |
| `docs/evaluation.md` | Methodology and measured results |
| `docs/prompt-engineering.md` | Prompt versions paired with eval deltas |
| `docs/cost-latency.md` | Measured cost and latency per run |
| `docs/failure-analysis.md` | What broke, and what changed as a result |
| `docs/scaling.md` | What would change at larger scale |

## Limitations

Stated up front rather than discovered:

- **Single-user, single-process.** No authentication, no multi-tenancy, no
  queue. Deliberately out of scope.
- **The corpus is per-question and ephemeral.** This is not a persistent
  knowledge base; each run assembles and discards its own index. That is what
  justifies SQLite over a vector database ([ADR-004](docs/decisions/)).
- **English-language sources only** in V1.
- **Research quality is bounded by what is publicly reachable.** No paywalled
  filings, no proprietary databases.
- **Citation verification proves a quote exists in a source. It does not prove
  the source is correct.** Credibility tiers are a heuristic about provenance,
  not a fact-check.
- Further limitations are added as measurement reveals them, not removed.

## About

Built by Shivam Chaturvedi. The engineering intent is to take a research
workflow I know professionally — strategic research, competitive intelligence,
due diligence — and build the AI system that performs it reliably, with the
quality properties verified in code rather than asserted in prose.

No claim is made in this repository about professional experience. The code,
tests, and measured results are the evidence.
