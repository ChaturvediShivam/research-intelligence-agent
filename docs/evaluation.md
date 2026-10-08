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

---

# M7 — the seven approved metrics

Architecture §9 defines seven metrics; all seven are implemented and measured.
Golden set: `evals/datasets/golden_v1` — 20 cases, 13 train / 7 holdout.
Cases G01–G10 reuse the queries and judgments already annotated in
`retrieval_v1` (written before retrieval was tuned); G11–G20 are failure-mode
cases derived from documented system behaviour, each with its expected outcome
stated in advance.

Reproduce with `uv run python scripts/run_evaluation.py` (metrics 1–5, free)
or `--judge` (adds 6–7, billable).

## Baseline

`evals/results/evaluation_baseline_d69164e.json`, golden_v1, eval_v1, k=5.

| # | Metric | Value | Cases | Deterministic |
|---|---|---|---|---|
| 1 | retrieval_relevance | **0.833** | 18 | ✅ |
| 2 | source_coverage | **1.000** | 18 | ✅ |
| 3 | citation_correctness | **1.000** | 18 | ✅ |
| 4 | unsupported_claim_rate | **0.000** | 18 | ✅ |
| 5 | tool_selection | **1.000** | 20 | ✅ |
| 6 | answer_relevance | **0.583** | 6 | ✗ |
| 7 | report_quality | **0.183** | 6 | ✗ |

Metric 1 decomposes as recall@5 = 0.833, MRR = 0.833, nDCG@5 = 0.829.
Holdout split: retrieval_relevance 1.000, the other deterministic metrics
equal to the full-set values.

Also reported, though not one of the seven: `unknown_expectation_accuracy`
= **0.900** — the share of cases whose answered/UNKNOWN outcome matched the
advance annotation. Two cases miss it (G11, G19), both out-of-domain questions
that the harness's scripted extractor answers anyway; the real extractor
returns nothing for an irrelevant question, measured live in M4.

## Improvement cycle

**Weakness selected:** `report_quality` = 0.183, the lowest of the seven, and
a property of deterministic M6 code rather than of a model.

**Diagnosis (F-014):** on a run with no failures, `limitations` collapsed to
one boilerplate caveat and `next_steps` to one generic fallback, while the run
held unused measured facts — on G02, all four supported claims uncorroborated,
two at LOW confidence, six of eight sources unvetted.

**Change:** `_limitations` and `_next_steps` in `app/pipeline/assess.py` now
also report evidence-level limitations derived from existing verified output.
No new model call, no schema change.

**Result:**

| Metric | Baseline | After | Verdict |
|---|---|---|---|
| retrieval_relevance | 0.833 | 0.833 | unchanged |
| source_coverage | 1.000 | 1.000 | unchanged |
| citation_correctness | 1.000 | 1.000 | unchanged |
| unsupported_claim_rate | 0.000 | 0.000 | unchanged |
| tool_selection | 1.000 | 1.000 | unchanged |
| answer_relevance | 0.583 | 0.542 | regressed |
| report_quality | 0.183 | 0.208 | improved |

**The improvement is not demonstrated.** `answer_relevance` moved further than
`report_quality` did, and the change cannot affect answer generation. A third
run on identical code gave answer_relevance 0.583 and report_quality 0.200 —
so the judge's noise floor on 6 cases is **±0.04**, larger than the +0.025
measured. The change is kept because it is independently correct and regresses
none of the five deterministic metrics, not because the metric improved.

A reverted earlier attempt is documented in F-014's neighbours: a dense
similarity floor in retrieval, calibrated from measured similarities
(relevant 0.582–0.638, off-domain 0.478–0.507). It cut a chunk from G16 — the
paraphrase case dense retrieval exists for — while leaving both off-domain
cases still answered, because the lexical half bypasses it. Reverted.

## Reproducibility

- **Metrics 1–5 are reproducible.** Identical across all three runs.
- **Metrics 6–7 are not**, by design: they are LLM-judge scores. Measured
  variation is ±0.04 on 6 cases.

## Limitations

- 20 cases, and metrics 6–7 judged on only 6 of them. Differences below ~0.05
  on the judged metrics are noise.
- Five of seven deterministic metrics sit at or near ceiling (1.000, 1.000,
  0.000, 1.000), so they currently detect regressions rather than discriminate
  between good and better.
- The harness's LLM transport is scripted. Citation correctness is a genuine
  measurement — the evaluator re-slices the stored source independently — but
  answer text comes from a scripted synthesiser, so metrics 6–7 assess the
  report's deterministic assembly more than a real model's prose.
- Judgments are mine, single-annotator. Annotator bias is unmeasured.

---

# Failure analysis — per metric

What each metric means, how it is computed, why it can be trusted, and where
it misleads. Figures are from `evaluation_baseline_d69164e.json`.

## 1. retrieval_relevance — 0.833

**Means:** did retrieval surface the documents annotated as relevant.
**Computed:** recall@5, MRR and nDCG@5 against graded judgments (0/1/2), fused
by averaging the three; `k=5` from `RETRIEVAL_K`. Cases with no annotated
relevant document are *not scored* rather than scored zero.
**Trustworthy because:** judgments were written before retrieval was tuned
(G01–G10 reuse `retrieval_v1`), the corpus is fixed, and the holdout split is
never used to make a change.

**Where it misleads, and this matters.** The headline 0.833 reads as a
retrieval weakness. It is not. All three misses are the fetch-failure cases:

| Case | Condition | recall@5 |
|---|---|---|
| G12 | `all_fetch_fail` | 0.000 |
| G13 | `access_restricted` | 0.000 |
| G14 | `one_fetch_fail` | 0.000 |

**On all 15 healthy cases, recall@5 is 1.000 with no exceptions.** A document
that cannot be fetched cannot be retrieved, so scoring zero is defensible —
but the single number conflates "retrieval ranked badly" with "the document
was unreachable", and only the first is a retrieval problem. Read the
condition breakdown, not the aggregate.

This is also why the reverted improvement attempt (a dense similarity floor)
was aimed at a non-weakness, and why the accepted cycle targeted
`report_quality` instead.

## 2. source_coverage — 1.000

**Means:** fraction of annotated relevant sources the discovery stage found.
**Computed:** `|discovered ∩ relevant| / |relevant|`, URL-matched.
**Trustworthy because:** discovery runs against a fixed candidate list per
case, sized to fit inside the 8-per-query cap (F-012).
**Limitation:** at ceiling, so it currently detects regressions rather than
discriminating between good and better. It also measures discovery against an
*annotated* set, not against everything that exists.

## 3. citation_correctness — 1.000 (57/57)

**Means:** share of citations whose quote is genuinely at its recorded offsets.
**Computed:** the evaluator re-slices the stored source text itself and
compares, byte for byte.
**Trustworthy because this is the one metric that cannot be gamed by the
component under test.** It does **not** read `Claim.verified_citations` —
trusting the pipeline's own verdict would make the metric a tautology, the
verifier grading its own work. A test corrupts that field and asserts the
metric is unmoved.
**Limitation:** proves a quote is *present*, never that it is *true*, and not
that the quote supports the claim built around it.

## 4. unsupported_claim_rate — 0.000 (0/57)

**Means:** share of claims carrying no verified citation. Lower is better, and
`compare()` knows the direction.
**Computed:** `claims with zero verified citations / total claims`.
**Trustworthy because:** it counts against the independently re-verified
citations from metric 3, not the pipeline's own flags.
**Limitation:** 0.000 here is partly a property of the harness — the scripted
synthesiser quotes real sentences. The live M5 run is the stronger evidence:
30 claims, 16 supported, **14 marked UNKNOWN rather than dropped**. A real
model produces unsupported claims; the system's job is to mark them, and it
did.

## 5. tool_selection — 1.000

**Means:** did the run use the expected tools.
**Computed:** expected vs actual tool set per case, exact match.
**Limitation — the weakest metric by construction.** The pipeline is
deterministic with exactly one model-driven step, so the tool set is nearly
fixed and this is close to a tautology. It would catch a stage silently
skipping its tool; it measures no real decision quality. Reported because
architecture §9 names it, not because it discriminates.

## 6. answer_relevance — 0.583 (6 cases) · non-deterministic

**Means:** does the answer address the question, judged by Haiku against a
written rubric (`rubric_v1`).
**Computed:** one judge call per case, score 0–1.
**Guards:** the judge never sees the system's own confidence or verdicts, so
it cannot be anchored by them; the rubric explicitly credits an honest refusal
so declining an unanswerable question is not punished.
**Limitation:** 6 cases and a measured ±0.04 noise floor. Differences below
~0.05 are noise. The answer text comes from a scripted synthesiser, so this
scores the report's assembly more than a real model's prose.

## 7. report_quality — 0.183 (6 cases) · non-deterministic · **weakest**

**Means:** is the report honest and usable — judged on whether it distinguishes
established from unestablished, names unanswered questions with reasons, gives
run-specific rather than boilerplate limitations, and ties next steps to a
stated gap.
**Computed:** one judge call per case against `rubric_v1`.

**This is the weakest component, and the diagnosis is concrete (F-014):** on a
run where nothing failed, `limitations` collapsed to one boilerplate sentence
and `next_steps` to one generic fallback, failing two of five rubric criteria
outright — while the run held unused measured facts (on G02, all four
supported claims uncorroborated, two at LOW confidence, six of eight sources
unvetted).

Per-case judged scores show the floor is uniform, not driven by an outlier:

| Case | answer_relevance | report_quality |
|---|---|---|
| G01 | 0.25 | 0.15 |
| G02 | 1.00 | 0.20 |
| G03 | 0.75 | 0.15 |
| G04 | 0.75 | 0.20 |
| G05 | 0.50 | 0.20 |
| G06 | 0.25 | 0.20 |

The fix raised it to 0.208 — **inside the noise floor, so the improvement is
not demonstrated.** See the improvement cycle above.

## Deterministic vs not

| Deterministic (1–5) | Non-deterministic (6–7) |
|---|---|
| Identical across five consecutive runs | ±0.04 measured on 6 cases |
| Safe for regression gating | Not safe for gating; trend only |

## Examples of failed cases

- **G12 / G13 / G14** — relevant document annotated but unfetchable. Recall
  0.000, and correctly so; `unknown_expectation_met` is true for all three,
  meaning the system reported UNKNOWN as required.
- **G11 / G19** — `unknown_expectation_accuracy` misses here (0.900 overall).
  Both are out-of-domain questions with no relevant document, and both produce
  supported claims anyway. **This is a harness artifact, not a system fault:**
  the scripted extractor quotes a real sentence from whatever chunk it is
  given, whereas the real extractor returns nothing for an irrelevant
  question — measured live in M4. The honest reading is that these two cases
  do not currently test what they were written to test.

## What the weakest component is

**`report_quality` (0.183)** — and unlike the judged answer metric, it is a
property of deterministic M6 code rather than of a model, so it is fixable by
engineering rather than by prompting. The attempted fix is documented, measured
and *not* claimed as a win.

The weakest *deterministic* signal is **`tool_selection`**, for the opposite
reason: at 1.000 on a near-fixed tool set, it is close to a tautology and
discriminates almost nothing.
