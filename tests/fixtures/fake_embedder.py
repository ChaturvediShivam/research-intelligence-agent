"""A deterministic embedder for tests.

Real embeddings are local and free but take ~2s of model load, which is too
slow for unit tests that run on every save. This stand-in is deterministic and
instant, built from a hash so the same text always produces the same vector.

It is a test double, not a fallback: nothing in `app/` may use it. Retrieval
*quality* is measured with the real model in the eval harness; these tests
measure retrieval *mechanics* — offsets surviving, fusion ordering, SQL
correctness — which a deterministic vector exercises perfectly well.
"""

from __future__ import annotations

import hashlib
import math

from app.retrieval.embeddings import l2_normalise


class FakeEmbedder:
    """Hash-based deterministic embeddings."""

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(self._dimension)]
        return l2_normalise(raw)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class TermOverlapEmbedder:
    """Embeddings that encode term presence, so similarity is meaningful.

    FakeEmbedder is deterministic but its vectors are unrelated to meaning, so
    it cannot test that a *relevant* chunk ranks above an irrelevant one. This
    one places each vocabulary term on its own axis, which makes cosine
    similarity equal normalised term overlap — enough to assert ordering.
    """

    def __init__(self, vocabulary: list[str]) -> None:
        self._vocabulary = [v.lower() for v in vocabulary]

    @property
    def dimension(self) -> int:
        return len(self._vocabulary)

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        raw = [1.0 if term in lowered else 0.0 for term in self._vocabulary]
        if not any(raw):
            # Avoid a zero vector: sqlite-vec would return distance 0 for it.
            raw = [1.0 / math.sqrt(self.dimension)] * self.dimension
        return l2_normalise(raw)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
