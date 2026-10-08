# Evaluation

**Retrieval results below, measured 2026-10-08 (M3). Report-level evaluation
arrives in M7.**

This file will carry measured numbers only. Nothing is published here that was
not produced by an actual run, with the git SHA and timestamp of that run.

## Methodology

Golden set: 20–30 research questions in `evals/datasets/golden_v1.jsonl`, each
annotated with the sources a competent analyst would consider relevant.

**Annotation happens before retrieval is tuned.** Annotating afterwards grades
retrieval against its own output. A held-out split is scored every round and is
the headline number, so prompt changes cannot be overfitted to the measured set.

| Metric | Method | Deterministic |
|---|---|---|
| Retrieval relevance | `recall@k`, MRR, nDCG | ✅ |
| Source coverage | annotated sources discovered ÷ total annotated | ✅ |
| Citation correctness | offsets re-verified against stored source text | ✅ |
| Unsupported-claim rate | claims lacking a verified citation ÷ total claims | ✅ |
| Tool selection | expected vs actual tool sequence | ✅ |
| Answer relevance | LLM judge (Haiku) against a written rubric | ✗ |
| Report quality | LLM judge against a written rubric | ✗ |

Five of seven are deterministic and therefore free to re-run on every change.

## Retrieval results — M3 baseline

Fixture: `evals/datasets/retrieval_v1` — 20 documents (10 on-topic, 10 hard
negatives), 10 queries, 14 graded judgments. Embedder:
`BAAI/bge-small-en-v1.5` (384-d, local). Reproduce with
`uv run python scripts/run_retrieval_eval.py`.

**Read the fixture's [README](../evals/datasets/retrieval_v1/README.md) before
quoting these numbers.** It is a hand-authored fixture: it proves the plumbing
works and catches regressions. It is not a measurement of open-web retrieval
quality.

### Shipped configuration — `chunk_tokens=512`, `overlap=64`

20 documents produce 20 chunks: every fixture document is shorter than the
chunk budget, so each is a single chunk.

| Retriever | k | recall@k | P@k | MRR | nDCG@k |
|---|---|---|---|---|---|
| **hybrid** | 1 | 0.800 | 1.000 | **1.000** | **1.000** |
| **hybrid** | 3 | 0.950 | 0.433 | 1.000 | 0.983 |
| **hybrid** | 5 | 1.000 | 0.280 | 1.000 | 0.994 |
| dense | 1 | 0.800 | 1.000 | 1.000 | 1.000 |
| dense | 3 | 0.950 | 0.433 | 1.000 | 0.983 |
| dense | 5 | 1.000 | 0.280 | 1.000 | 0.994 |
| lexical | 1 | 0.750 | 0.900 | 0.950 | 0.900 |
| lexical | 3 | 0.900 | 0.400 | 0.950 | 0.931 |
| lexical | 5 | 0.950 | 0.260 | 0.950 | 0.943 |

### Fine chunking — `chunk_tokens=40`, `overlap=8`

The same 20 documents produce 88 chunks, so fusion has multiple chunks per
document to reorder.

| Retriever | k | recall@k | P@k | MRR | nDCG@k |
|---|---|---|---|---|---|
| **hybrid** | 1 | 0.800 | 1.000 | 1.000 | 0.933 |
| **hybrid** | 3 | **1.000** | **0.467** | 1.000 | **0.972** |
| dense | 1 | 0.800 | 1.000 | 1.000 | 0.933 |
| dense | 3 | 0.950 | 0.433 | 1.000 | 0.962 |
| lexical | 1 | 0.700 | 0.800 | 0.883 | 0.800 |
| lexical | 3 | 0.850 | 0.367 | 0.883 | 0.845 |

## What these numbers actually support

**Dense beats lexical, consistently.** MRR 1.000 against 0.950 at the shipped
config, and nDCG higher at every cutoff. The single lexical failure is `q07`
("how much am I covered for if my insurer goes bust"), whose relevant document
says "unable to meet claims" and never uses the word "bust". That is the
paraphrase case dense retrieval exists for, and BM25 cannot reach it.

**Hybrid is identical to dense at the shipped chunk size — not better.**
Every number matches to three decimal places. This does **not** vindicate
hybrid search, and it is worth being plain about: at `chunk_tokens=512` every
fixture document is a single chunk, so there is nothing within a document for
fusion to reorder, and the fused ranking collapses onto the dense ranking.

**Hybrid does beat dense once documents are chunked finely.** At
`chunk_tokens=40`: recall@3 1.000 against 0.950, nDCG@3 0.972 against 0.962.
With 88 chunks, the lexical half contributes rankings that change the fused
order.

**The honest conclusion:** this fixture cannot test fusion at the shipped
configuration, because its documents are shorter than one chunk. That is a
limitation of the fixture, not evidence that hybrid retrieval is useless — and
it is not evidence that it is useful either. Hybrid is retained because
removing it on the strength of a 10-query synthetic fixture would be
overfitting to a weak signal, and because the `q07`-shaped and
rare-exact-token cases it is designed for are both real. **The claim stays
open until M7 measures it on real fetched sources.**

**k must be below the corpus size.** At k=10 with 20 documents recall is 1.000
for every retriever; at k=20 it would be trivially 1.000 by construction. The
harness prints a warning when asked for a cutoff at or above the corpus size.

## Known limitations of this baseline

- 10 queries is too few for a difference of 0.05 to be meaningful. These
  numbers are a regression gate, not a ranking of approaches.
- The corpus is prose I wrote and the judgments are mine. Annotator bias is
  unmeasured and, with one annotator, unmeasurable.
- Documents are ~120 words; real fetched sources ran 2,722–78,133 characters
  in M2. Chunking behaviour at realistic length is exercised by the live
  tests, not by this fixture.
- No train/test split yet. Required before any prompt or parameter is tuned
  against these numbers, and due in M7.

## Cost of running the eval

Measured per run and recorded here once M7 executes. Estimated beforehand at
$3–8 per full pass; the estimate will be replaced by the measurement, not
confirmed by it.
