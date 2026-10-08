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

_Empty until M1._

| Prompt | Version | Change | Metric affected | Before | After | Verdict |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | not yet run |

A row is only added after the eval has actually been executed on both versions
against the same split.
