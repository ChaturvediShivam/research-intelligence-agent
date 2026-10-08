# ADR-008: Prompt caching strategy, and verifying it actually works

**Status:** Accepted · **Date:** 2026-10-08

## Context
Stage 5 makes one call per chunk, and stage 6 sends the full source corpus as
document blocks. Both resend a large, stable prefix. Cache reads on Claude Opus
5.5 are $0.20/MTok against $4.00/MTok for uncached input — a 20× difference on
the dominant cost in this system.

Caching is a **prefix match**: any byte change anywhere in the prefix
invalidates everything after it. The render order is `tools` → `system` →
`messages`.

## Decision
1. Stable content first: a frozen system prompt and a deterministically
   ordered tool list, both before any volatile content.
2. `cache_control: {"type": "ephemeral"}` on the system prompt and on the
   source-corpus document blocks.
3. No timestamps, UUIDs, or per-request ids anywhere in the prefix. Volatile
   content (the specific question, the chunk under examination) goes after the
   last breakpoint.
4. Tool definitions are serialised with sorted keys so the prefix is
   byte-identical across runs.

## Alternatives considered
- **No caching.** Rejected: leaves the largest single cost saving unclaimed.
- **Trusting the configuration.** Rejected — see below. A silent invalidator
  produces zero cache hits with no error.

## Consequences
**Verification is mandatory, not optional.** An integration test asserts
`usage.cache_read_input_tokens > 0` across repeated identical-prefix requests.
Without that assertion a broken cache is indistinguishable from a working one
except on the invoice.

**Accepted cost:** the system prompt becomes a frozen artifact that cannot
cheaply carry per-run context, which is why operator context goes in
mid-conversation system messages instead — a choice that also keeps the
operator channel separate from untrusted fetched content (see `docs/architecture.md`,
security section).

**Revisit when:** measured cache hit rate on real runs falls below ~70%.
