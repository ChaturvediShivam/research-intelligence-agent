"""The real local embedding model.

Not marked `live`: fastembed runs locally and costs nothing (ADR-003), so
these belong in the default suite. They are the only tests that exercise the
actual ONNX model, and they assert the properties retrieval depends on —
determinism, unit length, and that semantic similarity actually orders the way
retrieval assumes.

First run downloads ~67 MB to .fastembed_cache; subsequent runs are fast.
"""

from __future__ import annotations

import math

import pytest

from app.retrieval.embeddings import FastEmbedEmbedder

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def embedder() -> FastEmbedEmbedder:
    """Module-scoped: the model loads once for the whole file."""
    return FastEmbedEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class TestRealEmbeddings:
    def test_dimension_matches_the_declared_schema_width(self, embedder: FastEmbedEmbedder) -> None:
        """A mismatch here would corrupt the vec0 table silently."""
        vectors = embedder.embed_documents(["a short passage of text"])
        assert len(vectors) == 1
        assert len(vectors[0]) == embedder.dimension == 384

    def test_vectors_are_unit_length(self, embedder: FastEmbedEmbedder) -> None:
        """Normalisation is what makes cosine and inner product coincide."""
        vector = embedder.embed_documents(["gross written premium"])[0]
        assert math.sqrt(sum(c * c for c in vector)) == pytest.approx(1.0, abs=1e-5)

    def test_deterministic_across_calls(self, embedder: FastEmbedEmbedder) -> None:
        """An index built now must agree with a query embedded later."""
        first = embedder.embed_documents(["the same sentence"])[0]
        second = embedder.embed_documents(["the same sentence"])[0]
        assert first == pytest.approx(second)

    def test_batch_matches_individual_embedding(self, embedder: FastEmbedEmbedder) -> None:
        """Batching must not change a vector, or eval results shift with batch size."""
        texts = ["first passage", "second passage"]
        batched = embedder.embed_documents(texts)
        individually = [embedder.embed_documents([t])[0] for t in texts]
        for a, b in zip(batched, individually, strict=True):
            assert a == pytest.approx(b, abs=1e-6)

    def test_semantic_similarity_orders_as_retrieval_assumes(
        self, embedder: FastEmbedEmbedder
    ) -> None:
        """The property the dense half of hybrid retrieval rests on.

        A paraphrase with no shared vocabulary must still score above an
        unrelated passage — this is exactly the case BM25 cannot handle.
        """
        query = embedder.embed_query("how much am I covered for if my insurer fails")
        relevant = embedder.embed_documents(
            [
                "The compensation scheme protects policyholders when an "
                "authorised insurer is unable to meet claims."
            ]
        )[0]
        irrelevant = embedder.embed_documents(
            ["Telematics underwriting uses driving data captured by a device."]
        )[0]
        assert cosine(query, relevant) > cosine(query, irrelevant)

    def test_query_prefix_changes_the_vector(self, embedder: FastEmbedEmbedder) -> None:
        """bge is asymmetric: a query and the same text as a document differ."""
        as_query = embedder.embed_query("market concentration")
        as_document = embedder.embed_documents(["market concentration"])[0]
        assert as_query != pytest.approx(as_document)

    def test_empty_list_short_circuits_without_loading(self) -> None:
        assert FastEmbedEmbedder().embed_documents([]) == []

    def test_dimension_mismatch_is_detected(self) -> None:
        """Configuring a model whose width differs must fail loudly."""
        wrong = FastEmbedEmbedder(dimension=128)
        with pytest.raises(ValueError, match="requires a reindex"):
            wrong.embed_documents(["text"])
