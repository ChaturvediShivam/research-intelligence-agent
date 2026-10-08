# Evaluation

**No results yet. The harness arrives in M3 (retrieval) and M7 (full report).**

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

## Results

_Empty until M3 produces the first retrieval baseline._

| Date | SHA | Split | recall@12 | Citation correctness | Unsupported-claim rate | Notes |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | not yet run |

## Cost of running the eval

Measured per run and recorded here once M7 executes. Estimated beforehand at
$3–8 per full pass; the estimate will be replaced by the measurement, not
confirmed by it.
