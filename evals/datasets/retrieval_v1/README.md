# Retrieval evaluation fixture v1

**What this is:** a small, hand-annotated, fully self-contained fixture for
measuring the retrieval layer. 10 documents, 10 queries, 14 graded judgments.

**What this is not:** a benchmark of real-world retrieval quality.

That distinction matters, so it is stated here rather than buried. The corpus
is prose I wrote about a domain I know. The judgments are my own. A number
produced against this fixture tells you:

- the retrieval plumbing works end to end — embed, index, search both ways,
  fuse, rehydrate with offsets intact;
- whether a change to chunking, embedding, fusion or `k` **made retrieval
  better or worse than it was yesterday**, which is what a regression gate is
  for.

It does **not** tell you how the system performs on the open web, where
documents are longer, noisier, partly irrelevant, and not written by the
person grading them. Claims of that kind need a different dataset built from
real fetched sources with independent annotation, which is a later milestone
and will be labelled separately.

## Files

| File | Contents |
|---|---|
| `corpus.jsonl` | `{doc_id, url, title, text}` — 10 documents, ~120 words each |
| `queries.jsonl` | `{query_id, query, judgments: {doc_id: grade}}` |

## Grading scale

| Grade | Meaning |
|---|---|
| 2 | Directly answers the query |
| 1 | Relevant context — useful, but does not answer it |
| 0 | Irrelevant (unjudged documents are treated as 0) |

Graded rather than binary so nDCG has something to work with: the ranking
difference between a grade-2 and a grade-1 document at position 1 is exactly
what nDCG is for, and binary judgments throw that away.

## Why these queries

The set is deliberately mixed, because a fixture where every query favours the
same retriever cannot show whether hybrid search earns its complexity:

| Query shape | Example | Expected to favour |
|---|---|---|
| Exact terminology | `q03` "Solvency Capital Requirement", `q06` "Insurance Premium Tax higher rate" | **BM25** — rare exact tokens |
| Natural paraphrase | `q07` "how much am I covered for if my insurer goes bust", `q09` "what is pushing up the cost of motor claims" | **Dense** — no term overlap with the source wording |
| Mixed | `q01`, `q02` | Either |

`q07` is the clearest case: the document says "unable to meet claims" and
"protection is ninety per cent", never "goes bust". A purely lexical retriever
should struggle; a dense one should not.

`insurtech-funding` is a near-distractor — plausibly insurance-adjacent but
relevant to only one query — so a retriever that returns topically-related
documents indiscriminately is penalised.

## Annotation order

The corpus and judgments were written **before** the retrieval parameters were
tuned, which is the only way the numbers mean anything. Grading retrieval
against judgments written after seeing its output measures nothing.

## Reproducing

```bash
uv run python scripts/run_retrieval_eval.py
```

Deterministic: same code and same fixture produce identical numbers. Results
are written to `evals/results/` with a timestamp and the git SHA.
