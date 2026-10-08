"""Retrieval metrics: recall@k, MRR, nDCG.

All three are pure functions of a ranked list and a set of judgments, which is
what lets them be tested against hand-computed values rather than against
their own output. Every metric here is deterministic — no model call, no
sampling — so re-running the eval on unchanged code must reproduce the number
exactly. An eval that drifts cannot detect a regression.

Judgments are **graded**, not binary: a source can be 2 (directly answers),
1 (relevant context) or 0 (irrelevant). recall@k and MRR collapse that to
binary at a threshold; nDCG uses the grades, which is the point of having
them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

# A judgment at or above this grade counts as relevant for binary metrics.
RELEVANT_THRESHOLD = 1


def recall_at_k(
    retrieved: Sequence[str],
    judgments: Mapping[str, int],
    k: int,
    *,
    threshold: int = RELEVANT_THRESHOLD,
) -> float:
    """Fraction of all relevant documents that appear in the top k.

    Returns 0.0 when nothing is relevant — not 1.0. A query with no relevant
    documents is a degenerate case, and scoring it as perfect would inflate
    the mean across a dataset.
    """
    if k <= 0:
        return 0.0
    relevant = {doc for doc, grade in judgments.items() if grade >= threshold}
    if not relevant:
        return 0.0
    # Deduplicate while preserving rank order: a retriever that returns the
    # same document twice must not get credit twice.
    top_k = _dedupe(retrieved)[:k]
    found = sum(1 for doc in top_k if doc in relevant)
    return found / len(relevant)


def precision_at_k(
    retrieved: Sequence[str],
    judgments: Mapping[str, int],
    k: int,
    *,
    threshold: int = RELEVANT_THRESHOLD,
) -> float:
    """Fraction of the top k that is relevant."""
    if k <= 0:
        return 0.0
    top_k = _dedupe(retrieved)[:k]
    if not top_k:
        return 0.0
    relevant = {doc for doc, grade in judgments.items() if grade >= threshold}
    return sum(1 for doc in top_k if doc in relevant) / len(top_k)


def reciprocal_rank(
    retrieved: Sequence[str],
    judgments: Mapping[str, int],
    *,
    threshold: int = RELEVANT_THRESHOLD,
) -> float:
    """1 / rank of the first relevant document, or 0.0 if none is retrieved.

    Measures how far a reader has to scroll before the first useful result —
    the metric that matters most when only the top few results get read.
    """
    relevant = {doc for doc, grade in judgments.items() if grade >= threshold}
    for rank, doc in enumerate(_dedupe(retrieved), start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    """Discounted cumulative gain, using graded relevance.

    Gain is `2**grade - 1`, so a grade-2 document is worth three times a
    grade-1 one rather than twice — the standard exponential form, which
    rewards getting the directly-answering source into the top ranks.
    """
    total = 0.0
    for rank, doc in enumerate(_dedupe(retrieved)[:k], start=1):
        grade = judgments.get(doc, 0)
        if grade > 0:
            total += (2**grade - 1) / math.log2(rank + 1)
    return total


def ndcg_at_k(retrieved: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    """DCG normalised by the best achievable DCG for this query.

    Returns 0.0 when no document is relevant, for the same reason as
    `recall_at_k`.
    """
    if k <= 0:
        return 0.0
    ideal_order = sorted(
        (doc for doc, grade in judgments.items() if grade > 0),
        key=lambda doc: -judgments[doc],
    )
    ideal = dcg_at_k(ideal_order, judgments, k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(retrieved, judgments, k) / ideal


def _dedupe(items: Sequence[str]) -> list[str]:
    """Preserve first occurrence order, drop repeats."""
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0
