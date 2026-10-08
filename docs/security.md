# Security boundaries

Two boundaries, both in `app/core/security.py`: which URLs the service will
fetch (M2), and what retrieved text is allowed to influence once fetched (M9).
This document covers the second. The first is described in the module
docstring and tested in `tests/security/test_ssrf.py`.

## What counts as untrusted content

Everything the system retrieves from outside itself:

- fetched page text, in full
- page titles, `<h1>` headings and any other page-supplied label
- search-result titles, snippets and URLs
- extracted quotes, because they are substrings of the above
- document text attached to a synthesis call

Caller-supplied input sits in a middle tier: a research question and its
optional `context` are more trusted than a web page but are still not the
operator, so `context` is framed as background rather than instruction.

## The boundary is structural, not lexical

Retrieved text never enters a system prompt. It enters in exactly two ways:

1. **Fenced user-channel content** — `app.llm.context.frame_untrusted` wraps
   it in markers with the operator instruction *outside* the fence. Any
   occurrence of a fence marker in the content is stripped, so content cannot
   close the fence and escape into the instruction channel.
2. **A native `document` block** with citations enabled, alongside the
   `CITED_SYSTEM` operator text, which states that documents are source
   material and not instructions.

**This is deliberately not keyword filtering.** A genuine regulatory document
may contain the sentence "ignore previous guidance", and a system that deleted
it would corrupt the evidence it exists to report. Hostile text is therefore
carried faithfully and denied authority rather than edited.

One asymmetry is worth naming: a document's **body** is passed verbatim,
because the citation verifier re-slices that exact string and any
transformation would break offset alignment. A document's **title** is
sanitised, because it sits in a structural slot where a newline or an angle
bracket could change how the frame parses, and a title is not evidence.

### What retrieved content cannot change

| Protected | Why it holds |
|---|---|
| Tool selection | A fixed registry (`app/tools/registry.py`); no text path adds or names a tool |
| System instructions | Retrieved text is never concatenated into a system prompt |
| Pipeline ordering | The orchestrator sequences stages; no stage reads an order from content |
| Verification rules | Stage 7 re-slices stored text and reads no instruction from it |
| Citation offsets | Code locates quotes; the model never supplies an offset |
| Secrets | Settings are read server-side and never placed in a prompt or trace |

## What the tests actually demonstrate

`tests/security/test_prompt_injection.py` — 157 tests over a 15-case corpus.
Each case is a realistic regulatory document carrying an attack *and* a true,
citable sentence, so a case cannot pass merely because nothing was extractable.

Proved, by running hostile documents through the real pipeline:

- the payload appears in no prompt file and not in `CITED_SYSTEM`
- the fence survives with exactly one opening and one closing marker, for
  every case, including the one that tries to close it
- a hostile title cannot introduce `<`, `>`, a newline or a fence marker,
  while the body it accompanies stays byte-identical
- stage order is unchanged for all 15 cases
- every surviving citation is verbatim at its recorded offsets in the stored
  source text, for all 15 cases
- no forced answer string (`COMPROMISED`, `ESCAPED`, `BURIED`,
  `100 per cent share`) ever becomes a supported claim
- a fabricated citation is still rejected when the document also claims
  verification is unnecessary — both attacks fail together
- the API key and the source text appear in neither the run nor its trace
- each hostile source is still fetched, chunked and researched: the attack
  fails without the evidence being lost

### What is **not** claimed

**Prompt injection is not solved, and nothing here claims it is.**

The suite proves properties of the *code*. Whether a model would obey an
injected instruction is a property of the model, and this design does not rely
on it not doing so. The guarantee that holds regardless is citation
verification: a claim whose quote is not verbatim in the stored source is
rejected by code **even if the model fully complied with the attack**. A model
that obeyed an injection could still produce a fluent, confident,
*uncited* — and therefore unsupported — claim, and the report would mark it
UNKNOWN rather than suppress it.

No live adversarial testing against a real model has been performed. Every
injection result above is a structural or deterministic one.

## Injection-signal detection is telemetry, not defence

`detect_injection_signals` categorises injection-like content for
observability. It **never** filters, rejects or edits anything — no code path
treats a non-empty result as a reason to act, and a test asserts the process
stage does not consult it.

Measured on the corpus: **13 of 15 cases carry a detectable signal.** The two
that do not are the ones that matter most for understanding the limit:

| Case | Why detection misses it |
|---|---|
| INJ10 social engineering | Contains no marker word; reads exactly like legitimate methodology prose |
| INJ14 pseudo-configuration | Config-looking syntax (`SET verification.enabled = false`) with no marker word |

Both are stopped structurally — neither reorders a stage, and neither forced
figure becomes a supported claim. That is the argument for the structural
boundary rather than a better word list: a lexical detector cannot catch
these two without flagging real documents.

## Remaining limitations

- No live adversarial evaluation against a real model.
- Detection is a fixed substring list; it is telemetry and will miss any
  phrasing not on it, as INJ10 and INJ14 demonstrate by construction.
- The corpus is English-language and text-only. No non-Latin script,
  homoglyph, zero-width-character, base64 or image-based injection is tested.
- Nothing defends against a source that is simply **wrong** rather than
  hostile. Citation verification proves a quote is present, not that it is
  true; credibility tiers and corroboration counts are reported for the
  reader to weigh, not enforced.
- The middle tier (caller `context`) is framed but not fenced, on the
  reasoning that a caller already controls the question.
