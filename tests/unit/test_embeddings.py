"""Embedder contract and vector normalisation."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.config import Settings
from app.retrieval.embeddings import (
    BGE_QUERY_PREFIX,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DIMENSION,
    Embedder,
    FastEmbedEmbedder,
    l2_normalise,
)
from tests.fixtures.fake_embedder import FakeEmbedder, TermOverlapEmbedder


class TestL2Normalise:
    def test_unit_length(self) -> None:
        out = l2_normalise([3.0, 4.0])
        assert out == pytest.approx([0.6, 0.8])
        assert sum(x * x for x in out) == pytest.approx(1.0)

    def test_already_normal_is_unchanged(self) -> None:
        assert l2_normalise([1.0, 0.0]) == pytest.approx([1.0, 0.0])

    def test_zero_vector_is_returned_unchanged(self) -> None:
        """Dividing by zero magnitude would raise; returning it is safe."""
        assert l2_normalise([0.0, 0.0]) == [0.0, 0.0]

    def test_idempotent(self) -> None:
        once = l2_normalise([2.0, 5.0, 1.0])
        assert l2_normalise(once) == pytest.approx(once)

    def test_negative_components(self) -> None:
        out = l2_normalise([-3.0, 4.0])
        assert sum(x * x for x in out) == pytest.approx(1.0)


class TestProtocolConformance:
    @pytest.mark.parametrize(
        "embedder",
        [FakeEmbedder(8), TermOverlapEmbedder(["a", "b"]), FastEmbedEmbedder()],
    )
    def test_satisfies_the_protocol(self, embedder: object) -> None:
        """ADR-003's hosted-embedder swap path depends on this protocol."""
        assert isinstance(embedder, Embedder)

    def test_default_dimension_matches_the_documented_model(self) -> None:
        # The vec0 column width is fixed at schema creation, so a change here
        # is a reindex and must be deliberate.
        assert DEFAULT_DIMENSION == 384
        assert FastEmbedEmbedder().dimension == 384


class TestFastEmbedConfiguration:
    def test_model_is_not_loaded_on_construction(self) -> None:
        """Construction must stay cheap: a factory may never use it."""
        embedder = FastEmbedEmbedder()
        assert embedder._model is None

    def test_query_prefix_is_applied_for_bge_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """bge is asymmetric; omitting the prefix silently costs recall."""
        embedder = FastEmbedEmbedder()
        seen: list[str] = []

        def fake_embed_documents(texts: list[str]) -> list[list[float]]:
            seen.extend(texts)
            return [[0.0] * 384 for _ in texts]

        monkeypatch.setattr(embedder, "embed_documents", fake_embed_documents)
        embedder.embed_query("market size")
        assert seen == [f"{BGE_QUERY_PREFIX}market size"]

    def test_no_prefix_for_a_non_bge_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        embedder = FastEmbedEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        seen: list[str] = []
        monkeypatch.setattr(
            embedder,
            "embed_documents",
            lambda texts: (seen.extend(texts), [[0.0] * 384])[1],
        )
        embedder.embed_query("market size")
        assert seen == ["market size"]

    def test_empty_document_list(self) -> None:
        assert FastEmbedEmbedder().embed_documents([]) == []


class TestTestDoubles:
    def test_fake_embedder_is_deterministic(self) -> None:
        """An index built in one process must agree with a query in another."""
        a, b = FakeEmbedder(8), FakeEmbedder(8)
        assert a.embed_query("same text") == b.embed_query("same text")

    def test_fake_embedder_vectors_are_unit_length(self) -> None:
        vector = FakeEmbedder(16).embed_query("x")
        assert sum(c * c for c in vector) == pytest.approx(1.0)

    def test_different_text_gives_different_vectors(self) -> None:
        e = FakeEmbedder(8)
        assert e.embed_query("alpha") != e.embed_query("beta")

    def test_term_overlap_embedder_ranks_by_shared_terms(self) -> None:
        e = TermOverlapEmbedder(["premium", "solvency"])
        query = e.embed_query("premium")
        close = e.embed_documents(["gross written premium"])[0]
        far = e.embed_documents(["solvency capital"])[0]
        dot_close = sum(a * b for a, b in zip(query, close, strict=True))
        dot_far = sum(a * b for a, b in zip(query, far, strict=True))
        assert dot_close > dot_far

    def test_term_overlap_handles_text_with_no_known_terms(self) -> None:
        """A zero vector would make sqlite-vec report distance 0 for everything."""
        vector = TermOverlapEmbedder(["premium"]).embed_documents(["unrelated"])[0]
        assert any(c != 0.0 for c in vector)


class RecordingModel:
    """Stands in for fastembed's `TextEmbedding`, recording how it was called.

    A fake rather than the real model on purpose: the thing under test is the
    keyword argument that reaches `embed`, and loading 130 MB of ONNX to check
    an integer would make the unit suite slow for no added confidence.
    """

    def __init__(self, dimension: int = DEFAULT_DIMENSION) -> None:
        self.calls: list[dict[str, Any]] = []
        self._dimension = dimension

    def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.calls.append({"count": len(texts), **kwargs})
        return [[1.0] + [0.0] * (self._dimension - 1) for _ in texts]


class TestEmbeddingBatchIsBounded:
    """F-019.

    fastembed's default batch size is 256. Stage 4 embeds every chunk of
    every source in one `embed_documents` call, and transformer attention
    allocates activations proportional to batch x sequence^2 — so that
    default asked ONNX for gigabytes and the kernel SIGKILLed the container
    mid-stage on a 512 MB instance. No Python exception is raised by a
    SIGKILL, so the background task vanished, nothing was logged, and the run
    sat at `planning` forever.

    These tests pin the bound. They cannot observe memory, so they assert the
    one thing that caused it: the batch size actually handed to fastembed.
    """

    def test_batch_size_is_passed_to_fastembed(self) -> None:
        embedder = FastEmbedEmbedder(batch_size=4)
        embedder._model = RecordingModel()
        embedder.embed_documents(["one", "two", "three"])
        assert embedder._model.calls == [{"count": 3, "batch_size": 4}]

    def test_the_default_is_never_fastembeds_own(self) -> None:
        """256 is the value that OOM-killed production. Anything near it is a bug."""
        assert DEFAULT_BATCH_SIZE == 4
        embedder = FastEmbedEmbedder()
        embedder._model = RecordingModel()
        embedder.embed_documents(["one"])
        assert embedder._model.calls[0]["batch_size"] == DEFAULT_BATCH_SIZE

    def test_a_query_is_embedded_with_the_same_bound(self) -> None:
        embedder = FastEmbedEmbedder(batch_size=2)
        embedder._model = RecordingModel()
        embedder.embed_query("what drove the change")
        assert embedder._model.calls[0]["batch_size"] == 2

    def test_batch_size_is_exposed(self) -> None:
        assert FastEmbedEmbedder(batch_size=7).batch_size == 7
        assert FastEmbedEmbedder().batch_size == DEFAULT_BATCH_SIZE

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_non_positive_batch_size_is_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            FastEmbedEmbedder(batch_size=bad)

    def test_settings_default_agrees_with_the_code_default(self) -> None:
        """The two defaults are written separately so `core` need not import
        `retrieval`. This is what stops them drifting apart."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.embedding_batch_size == DEFAULT_BATCH_SIZE

    @pytest.mark.parametrize("bad", [0, -3])
    def test_settings_rejects_a_non_positive_batch_size(self, bad: int) -> None:
        with pytest.raises(ValueError, match="embedding_batch_size"):
            Settings(_env_file=None, embedding_batch_size=bad)  # type: ignore[call-arg]

    def test_the_configured_batch_size_reaches_the_embedder(self) -> None:
        """The wiring, not just the default: a knob nothing reads is not a knob."""
        from app.tools.registry import ToolContext

        context = ToolContext(
            settings=Settings(_env_file=None, embedding_batch_size=3)  # type: ignore[call-arg]
        )
        embedder = context.embedder
        assert isinstance(embedder, FastEmbedEmbedder)
        assert embedder.batch_size == 3
        # Construction must still not have loaded the model.
        assert embedder._model is None
