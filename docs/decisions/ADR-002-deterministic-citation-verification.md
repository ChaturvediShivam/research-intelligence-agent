# ADR-002: Citation correctness verified in code, not by an LLM judge

**Status:** Accepted · **Date:** 2026-10-08

## Context
Citation correctness is the quality property this system is most responsible
for. The obvious implementation — a second model call asking "is this citation
accurate?" — makes the guarantee depend on the same class of component that
produces the error.

## Decision
Stage 7 (VALIDATE) contains **no LLM call**. For every citation returned by
stage 6, the verifier:

1. retrieves the stored source text for `document_index`;
2. slices it at `[start_char_index:end_char_index]`;
3. normalises both the slice and `cited_text` (unicode NFKC, whitespace
   collapse, quote folding);
4. asserts equality.

A citation whose quoted text is not present at the stated offsets is rejected,
and the claim depending on it is downgraded to UNKNOWN.

## Alternatives considered
- **LLM-as-judge.** Rejected: circular, non-deterministic, and costs money per
  verification.
- **Embedding similarity between claim and chunk.** Rejected: measures
  topical relatedness, not quotation. A paraphrase that inverts the meaning
  scores highly.
- **Exact byte equality with no normalisation.** Rejected: smart quotes,
  non-breaking spaces and HTML-entity decoding cause false rejections.

## Consequences
**Accepted cost:** normalisation is a judgement call and can mask a real
mismatch (e.g. collapsing whitespace hides a line-break difference that
changed meaning). Normalisation is therefore narrow, individually unit-tested,
and logged when it changes the string.

**Benefit worth stating:** verification is free, instant, and deterministic, so
it runs on every citation of every run rather than on a sample.

**Enforced by:** a test that injects a fabricated citation and asserts it is
caught. That test is part of the Definition of Done.

**Revisit when:** sources arrive in formats where character offsets are not
meaningful (scanned PDFs needing OCR).
