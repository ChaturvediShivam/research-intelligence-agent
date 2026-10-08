# ADR-001: Anthropic native document citations as the grounding backbone

**Status:** Accepted · **Date:** 2026-10-08

## Context
The product's central promise is that every factual claim is traceable to a
source. Implemented naively — asking the model to emit `[1]` markers and a
bibliography — citations are unverifiable: nothing links the sentence to the
words in the source, and a plausible-looking citation can be fabricated with
no way to detect it.

The Anthropic Messages API supports `citations: {enabled: true}` on `document`
content blocks. The response splits into text blocks, and cited blocks carry a
`citations` array where each entry has `cited_text` plus a location —
`char_location` with `start_char_index` / `end_char_index` for plain text.

## Decision
Use native document citations for stage 6 (SYNTHESISE). Source documents are
passed as `document` blocks with citations enabled; the returned `cited_text`
and character offsets become the verifiable link between claim and source.

## Alternatives considered
- **Prompt-instructed markers + bibliography.** Rejected: unverifiable, which
  defeats the purpose.
- **Post-hoc string search for every sentence in every source.** Rejected:
  O(claims × sources) fuzzy matching with no ground truth about *which* source
  the model actually drew on.
- **A separate verifier LLM call.** Rejected — see ADR-002.

## Consequences
**Accepted cost:** citations are **incompatible with `output_config.format`**;
sending both returns a 400. Synthesis therefore cannot be cited *and*
structured in one call, which forces the two-pass design (stage 6 produces
cited prose, stage 9 structures it). That is roughly one extra model call per
run. The tradeoff is accepted because the alternative is unverifiable
citations, which would remove the reason for the project to exist.

**Also accepted:** sources must be sent as documents, which raises input
tokens. Mitigated by prompt caching (ADR-008).

**Revisit when:** the API supports structured output alongside citations.
