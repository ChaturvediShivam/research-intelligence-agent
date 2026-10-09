# Prompt and context engineering

**No eval deltas yet. Prompts arrive in M1; measured deltas require the harness
from M3/M7.**

A well-written prompt is not evidence of prompt engineering. A prompt version
paired with its measured effect is. This file records the second kind.

## Conventions

- Prompts are versioned files in `app/llm/prompts/`, e.g. `plan.v1.md`.
- A new version is a new file; old versions stay for comparison.
- Context assembly — token budgeting, chunk ordering, cache-breakpoint
  placement, untrusted-content framing — lives in `app/llm/context.py`, not
  inlined at call sites.

## Context assembly rules

1. **Stable content first.** Frozen system prompt, then a deterministically
   ordered tool list (keys sorted), then volatile content. Caching is a prefix
   match; anything varying early invalidates everything after it
   ([ADR-008](decisions/ADR-008-prompt-caching.md)).
2. **No timestamps, UUIDs or run ids in the prefix.** The commonest silent
   cache invalidator.
3. **Untrusted content is framed, never merged.** Fetched page text is wrapped
   in explicit delimiters and labelled as data. It never enters `system`.
4. **Operator instructions use a separate channel.** Mid-conversation
   `{"role": "system"}` messages, which also preserve the cached prefix.

## Version log

| Prompt | Version | Change | Metric affected | Before | After | Verdict |
|---|---|---|---|---|---|---|
| plan | v1 → v2 | Source budget stated to the planner; sub-question count must respect it | none measurable offline | — | — | **not measured** |
| synthesize | v1 → v2 | Ask for cross-source citation; same publisher is one source | none measurable offline | — | — | **not measured** |

A row is only added after the eval has actually been executed on both versions
against the same split.

### Why those two rows say "not measured"

Neither change can be measured by the offline evaluation, and saying so is
more useful than a number that does not mean what it appears to.

- The eval harness builds its plan directly (`_plan_for(case)` in
  `tests/fixtures/eval_harness.py`) rather than calling stage 1, so the
  planning prompt is never sent. A budget-awareness change cannot move any
  offline metric.
- The harness's LLM transport is `ScriptedLLM`, which reads the documents it
  is sent and ignores the instruction text. A corroboration instruction
  cannot move an offline metric either.

Both need a live run to evaluate: the planner change against sub-question
count versus source budget, the synthesis change against the `corroboration`
field on supported claims. Until that run happens these are reasoned changes
with no measured delta, which is what the table says.

The offline evaluation was still run on both versions against the same split
and confirms **no regression** on the five deterministic metrics
(`evals/results/evaluation_stage1_quality_e9fb752.json` against
`evaluation_baseline_d69164e.json`): 0.833 / 1.000 / 1.000 / 0.000 / 1.000,
all unchanged.

The claim-filter change in the same batch is *not* a prompt change and does
have a measured delta — see F-020 in
[`docs/failure-analysis.md`](failure-analysis.md).
