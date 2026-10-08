# Cost and latency

**First real measurements below, from M1 live verification on 2026-10-08.**
Full-pipeline figures arrive with M9; these cover stage 1 only.

## What is measured

Per stage, per run, recorded in `RunTrace`:

- wall-clock duration
- model used
- `input_tokens`, `output_tokens`, `cache_read_input_tokens`,
  `cache_creation_input_tokens` — tracked **separately**, because cache reads
  cost $0.20/MTok against $4.00/MTok uncached on Claude Opus 5.5, and a single
  blended number would hide the one figure worth watching

Cost is derived from `app/llm/pricing.py`, not estimated.

## Price table in use

| Model | Input $/MTok | Output $/MTok | Cache read $/MTok |
|---|---|---|---|
| `claude-opus-5-5` | 4.00 | 20.00 | 0.20 |
| `claude-haiku-4-5` | 1.00 | 5.00 | — |

## Measured results

### Stage 1 (PLAN) — `claude-opus-5-5`, effort `high`

Measured 2026-10-08 via `uv run pytest -m live`. Question: *"How concentrated
is the UK pet insurance market, and which insurers hold the largest shares?"*
Every figure below is read from a real `response.usage`; none is estimated.

| Call | Input | Output | Cache read | Cache write | Cost | Latency |
|---|---|---|---|---|---|---|
| 1 | 40 | 2,813 | 2,326 | 0 | $0.056885 | 30,761 ms |
| 2 | 40 | 2,934 | 2,326 | 0 | $0.059305 | 31,813 ms |
| 3 | 40 | 2,582 | 2,326 | 0 | $0.052265 | 28,414 ms |

**Total measured: $0.168455 across 3 calls.**

### Stage 5 (EXTRACT) — `claude-haiku-4-5` · Stage 6 citations — `claude-opus-5-5`

Measured 2026-10-08 via `uv run pytest -m live tests/integration/test_live_evidence.py`.

| Call | Model | Input | Output | Cost | Latency |
|---|---|---|---|---|---|
| Extraction, 2 chunks | haiku-4-5 | 2,899 | 129 | $0.003544 | 3,600 ms |
| Extraction, irrelevant question | haiku-4-5 | ~1,400 | ~10 | $0.001513 | — |
| Cited synthesis, 2 documents | opus-5-5 | 1,360 | 424 | $0.013920 | 4,410 ms |

**Total for the M4 live suite: ~$0.032.**

The routing decision in ADR-007 is visible in these numbers: extraction
handled 2,899 input tokens for **$0.0035**, where the same volume on Opus
would have cost roughly 4x more — and extraction is the stage whose call count
scales with the number of retrieved chunks, so it is the one that had to be
cheap.

### What these numbers say

- **Cost is dominated by output, not input.** Only 40 input tokens were billed
  at full rate; output was 2,582–2,934 tokens. At $20/MTok output against
  $4/MTok input, roughly 97% of the cost of a planning call is the plan itself.
  Shrinking the prompt would save almost nothing; the lever is plan verbosity.
- **Caching works and is near-total.** 2,326 of 2,366 input tokens served from
  cache — a **98.3% hit rate** on the system prefix. Uncached, those 2,326
  tokens would cost $0.0093 per call instead of $0.00047.
- **Latency is 28–32 seconds** for one planning call at effort `high`. This is
  the single strongest justification for the API's return-an-id-and-poll
  design: a synchronous endpoint would hold a connection for half a minute on
  stage 1 alone, before any source has been fetched.
- **Output varied 2,582–2,934 tokens** across three identical requests, so
  per-run cost is not fixed. Cost ceilings must be enforced against measured
  spend, which is what `MAX_COST_USD_PER_RUN` does.

### Known measurement gap

A call rejected by schema validation discards its `usage` before it is read,
so the cost of a failed call is not captured (see F-004). Two such calls
occurred during M1 verification and are **not** included in the total above.

## Guardrail

`MAX_COST_USD_PER_RUN` (default 2.0) aborts a run that would exceed it, raising
`CostCeilingExceededError`. A truncated run reports as truncated; it never
returns a partial report that looks complete.

---

## Full pipeline — measured end-to-end run (M5)

`uv run pytest -m live tests/integration/test_live_e2e.py`, 2026-10-08.
Run id `run_5d9d076f1c0d49be`. Artifact:
`evals/results/live_e2e_run_5d9d076f1c0d49be.json`.

Question: *"How concentrated is the UK pet insurance market, and what is
driving claims costs upward?"*

| Stage | Status | Latency | Cost | Calls | In → Out |
|---|---|---|---|---|---|
| plan | passed | 38,805 ms | $0.071341 | 1 | 1 → 6 |
| discover | passed | 44,990 ms | $0.037060 | 2 | 6 → 5 |
| process | partial | 3,206 ms | $0.000000 | 0 | 5 → 4 |
| retrieve | passed | 1,023 ms | $0.000000 | 0 | 12 → 24 |
| extract | passed | 15,020 ms | $0.054884 | 24 | 24 → 26 |
| validate | passed | 0 ms | $0.000000 | 0 | 26 → 26 |
| synthesise | partial | 59,722 ms | $0.173460 | 5 | 26 → 30 |
| report | passed | 1 ms | $0.000000 | 0 | 30 → 16 |

**Total: $0.336745 · 162,774 ms wall clock · 32 LLM calls · 72,953 input /
10,456 output / 2,326 cache-read tokens.**

### What these numbers say

- **Synthesis dominates cost (51%) and latency (37%).** Five Opus calls, each
  carrying full source documents as input. This is the stage to attack if cost
  matters: fewer, larger synthesis calls would trade attribution for spend.
- **Extraction is cheap despite 24 of the 32 calls.** $0.054884 for 75% of the
  call volume — ADR-007's routing decision earning its keep, visible in the
  numbers rather than asserted.
- **Planning cost $0.071 and took 38.8 seconds** for a single call. High
  effort on Opus is expensive in wall clock; it is one call per run, so it is
  cheap per unit of value, but it is a third of the latency before any source
  is fetched.
- **Cache hit rate was 3.1%**, far below the 98.3% measured for planning
  alone in M1. Expected and not a regression: the dominant input is source
  documents, which differ per run and per sub-question, so there is little
  stable prefix to cache. The figure is worth watching rather than fixing.
- **Zero-cost stages are genuinely zero.** process, retrieve, validate and
  report make no model call, and report $0.00 rather than an estimate.
