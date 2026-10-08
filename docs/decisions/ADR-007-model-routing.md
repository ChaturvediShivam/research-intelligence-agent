# ADR-007: Model routing by pipeline stage

**Status:** Accepted · **Date:** 2026-10-08

## Context
The stages differ by an order of magnitude in both volume and judgement
required. Stage 5 (EXTRACT) runs once per retrieved chunk — potentially dozens
of calls per run — and asks a narrow question against a single chunk. Stages 1
and 6 (PLAN, SYNTHESISE) run once and carry the reasoning.

Using one frontier model for all of them overpays for extraction; using one
cheap model for all of them underpowers planning and synthesis.

## Decision
Route by stage, configurable in `app/core/config.py`:

| Stage | Model | Effort | Reasoning |
|---|---|---|---|
| 1 PLAN | `claude-opus-5-5` | `high` | Decomposing a question into testable sub-questions is the step that determines whether the whole run is useful |
| 5 EXTRACT | `claude-haiku-4-5` | n/a | High volume, narrow scope, schema-constrained output |
| 6 SYNTHESISE | `claude-opus-5-5` | `high` | Cited synthesis across sources; the quality-critical output |
| 7, 8 VALIDATE/ASSESS | **none** | — | Deterministic code (ADR-002) |

Effort is set **explicitly** on every Opus call. Claude Opus 5.5 defaults to
`medium`, and thinking cannot be disabled on it — effort is the only control,
so leaving it implicit would silently pick a level.

## Alternatives considered
- **One model everywhere (Opus).** Rejected: extraction cost scales with chunk
  count for no quality gain on a schema-constrained task.
- **One model everywhere (Haiku).** Rejected: planning and cited synthesis are
  exactly where capability shows.
- **Sonnet 5.5 as a middle tier.** Not in V1 — two tiers are enough to
  demonstrate the pattern. A third would need measurement to justify.

## Consequences
**Accepted cost:** two model ids to keep current, and caches are model-scoped,
so the extraction and synthesis stages cannot share a prompt cache.

**Measurement obligation:** the cost split per stage is recorded in
`RunTrace` and reported in `docs/cost-latency.md` from real runs. If extraction
on Haiku measurably degrades evidence quality against the golden set, the
routing changes — and the before/after is published rather than assumed.


---

## Amendment, 2026-10-08 (M4 implementation)

**"Effort is always explicit" is now "effort is explicit wherever the model
accepts one."**

This ADR required `output_config.effort` on every call, because Claude Opus
5.5 defaults to `medium` and leaving it implicit silently picks a level. Stage
5 is the first stage to route to `claude-haiku-4-5`, and that model **rejects
the parameter**: `400 — "This model does not support the effort parameter."`
Every extraction call failed on the first live run (F-010).

**Resolved as:** request capability is recorded in the model table beside the
prices (`ModelPrice.supports_effort`). `LLMClient` applies effort where the
model accepts it and omits `output_config` entirely where it does not. Call
sites continue to declare the effort they want, so the intent behind this ADR
is preserved and the wire format is handled in one place.

**Why not simply drop effort from the extraction stage.** That would make the
correct request today and the wrong one the moment `EXTRACTION_MODEL` is
pointed at an effort-capable model — which is a configuration change, not a
code change, and would silently lose the setting. Deciding from the capability
table keeps both configurations correct.

**Accepted cost:** one more per-model fact to maintain. Unknown models are
assumed to support effort, on the grounds that a newer model is more likely to
accept it than not, and an unpriced model already fails loudly in `cost_usd`.
