# Cost and latency

**No measurements yet. Instrumentation arrives with M5; figures are published
from real runs in M9.**

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

_Empty until M9._

| Date | SHA | Question class | Sources | Total $ | p50 latency | Cache hit rate |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | not yet run |

## Guardrail

`MAX_COST_USD_PER_RUN` (default 2.0) aborts a run that would exceed it, raising
`CostCeilingExceededError`. A truncated run reports as truncated; it never
returns a partial report that looks complete.
