"""RRF fusion and the hybrid retriever."""

from __future__ import annotations

import pytest

from app.retrieval.hybrid import RRF_K, FusionInput, HybridRetriever, reciprocal_rank_fusion
from app.retrieval.store import Hit, SqliteVecStore
from app.schemas.source import Chunk
from tests.fixtures.fake_embedder import FakeEmbedder, TermOverlapEmbedder


class TestReciprocalRankFusion:
    def test_single_list_preserves_order(self) -> None:
        hits = [Hit(10, 0.9), Hit(20, 0.5), Hit(30, 0.1)]
        fused = reciprocal_rank_fusion([FusionInput("dense", hits)])
        assert [chunk_id for chunk_id, _, _ in fused] == [10, 20, 30]

    def test_score_uses_rank_not_the_retriever_score(self) -> None:
        """The whole point of RRF: incomparable score scales don't matter."""
        # A huge score at rank 2 must not beat a small score at rank 1.
        hits = [Hit(1, 0.001), Hit(2, 9999.0)]
        fused = reciprocal_rank_fusion([FusionInput("x", hits)])
        assert fused[0][0] == 1

    def test_first_rank_contribution_is_one_over_k_plus_one(self) -> None:
        """Ranks are 1-based; 0-based would skew every fused score."""
        fused = reciprocal_rank_fusion([FusionInput("x", [Hit(1, 0.5)])], k=60)
        assert fused[0][1] == pytest.approx(1 / 61)

    def test_agreement_between_retrievers_outranks_a_single_top_hit(self) -> None:
        """A document both retrievers like should beat one only one likes."""
        dense = [Hit(1, 0.9), Hit(2, 0.8)]
        lexical = [Hit(3, 5.0), Hit(2, 4.0)]
        fused = reciprocal_rank_fusion(
            [FusionInput("dense", dense), FusionInput("lexical", lexical)]
        )
        # chunk 2: 1/62 + 1/62 = 0.03226 ; chunk 1: 1/61 = 0.01639
        assert fused[0][0] == 2
        assert set(fused[0][2]) == {"dense", "lexical"}

    def test_contributing_retrievers_are_recorded(self) -> None:
        fused = reciprocal_rank_fusion(
            [FusionInput("dense", [Hit(1, 0.1)]), FusionInput("lexical", [Hit(1, 0.2)])]
        )
        assert fused[0][2] == ("dense", "lexical")

    def test_union_not_intersection(self) -> None:
        """Fusion must keep documents only one retriever found."""
        fused = reciprocal_rank_fusion(
            [FusionInput("dense", [Hit(1, 0.1)]), FusionInput("lexical", [Hit(2, 0.1)])]
        )
        assert {chunk_id for chunk_id, _, _ in fused} == {1, 2}

    def test_ties_break_deterministically(self) -> None:
        """Unstable ordering would make eval numbers irreproducible."""
        a = reciprocal_rank_fusion(
            [FusionInput("x", [Hit(5, 0.1)]), FusionInput("y", [Hit(3, 0.1)])]
        )
        b = reciprocal_rank_fusion(
            [FusionInput("y", [Hit(3, 0.1)]), FusionInput("x", [Hit(5, 0.1)])]
        )
        assert [c for c, _, _ in a] == [c for c, _, _ in b] == [3, 5]

    def test_empty_inputs(self) -> None:
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([FusionInput("x", [])]) == []

    def test_larger_k_flattens_the_ranking(self) -> None:
        hits = [Hit(1, 0.9), Hit(2, 0.8)]
        tight = reciprocal_rank_fusion([FusionInput("x", hits)], k=1)
        loose = reciprocal_rank_fusion([FusionInput("x", hits)], k=1000)
        assert tight[0][1] - tight[1][1] > loose[0][1] - loose[1][1]

    def test_default_k_is_the_published_value(self) -> None:
        assert RRF_K == 60


class TestHybridRetriever:
    def _store(self, embedder: object, texts: list[str]) -> SqliteVecStore:
        store = SqliteVecStore(":memory:", dimension=embedder.dimension)  # type: ignore[attr-defined]
        chunks, cursor = [], 0
        for i, text in enumerate(texts):
            chunks.append(
                Chunk(
                    source_url=f"https://example.com/{i}",
                    index=0,
                    text=text,
                    start_char=cursor,
                    end_char=cursor + len(text),
                )
            )
            cursor += len(text) + 2
        store.add(chunks, embedder.embed_documents(texts))  # type: ignore[attr-defined]
        return store

    def test_returns_scored_chunks_with_provenance(self) -> None:
        embedder = TermOverlapEmbedder(["premium", "solvency", "telematics"])
        store = self._store(
            embedder,
            ["gross written premium", "solvency capital", "telematics data"],
        )
        try:
            results = HybridRetriever(store, embedder).retrieve("premium", k=3)
            assert results
            top = results[0]
            assert "premium" in top.chunk.text
            assert top.rank == 1
            assert top.chunk_id > 0
            assert top.retrievers
            # Offsets survive retrieval, which later citation checks depend on.
            assert top.chunk.end_char > top.chunk.start_char
        finally:
            store.close()

    def test_ranks_are_sequential_and_scores_descending(self) -> None:
        embedder = FakeEmbedder(8)
        store = self._store(embedder, [f"chunk number {i}" for i in range(6)])
        try:
            results = HybridRetriever(store, embedder).retrieve("chunk number 3", k=5)
            assert [r.rank for r in results] == list(range(1, len(results) + 1))
            assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
        finally:
            store.close()

    def test_k_limits_results(self) -> None:
        embedder = FakeEmbedder(8)
        store = self._store(embedder, [f"text {i}" for i in range(20)])
        try:
            assert len(HybridRetriever(store, embedder).retrieve("text", k=4)) == 4
        finally:
            store.close()

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_returns_nothing(self, k: int) -> None:
        embedder = FakeEmbedder(8)
        store = self._store(embedder, ["a"])
        try:
            assert HybridRetriever(store, embedder).retrieve("a", k=k) == []
        finally:
            store.close()

    def test_empty_corpus_returns_nothing(self) -> None:
        embedder = FakeEmbedder(8)
        store = SqliteVecStore(":memory:", dimension=8)
        try:
            assert HybridRetriever(store, embedder).retrieve("anything", k=5) == []
        finally:
            store.close()

    def test_lexical_only_match_is_still_retrieved(self) -> None:
        """A rare exact token the dense side misses must survive fusion.

        This is the case that justifies hybrid at all.
        """
        embedder = TermOverlapEmbedder(["unrelated"])
        store = self._store(
            embedder,
            ["nothing in the vocabulary", "mentions zzyzyx specifically"],
        )
        try:
            results = HybridRetriever(store, embedder).retrieve("zzyzyx", k=5)
            assert any("zzyzyx" in r.chunk.text for r in results)
            hit = next(r for r in results if "zzyzyx" in r.chunk.text)
            assert "lexical" in hit.retrievers
        finally:
            store.close()

    def test_retrieval_is_deterministic(self) -> None:
        embedder = FakeEmbedder(8)
        store = self._store(embedder, [f"document {i} content" for i in range(10)])
        try:
            retriever = HybridRetriever(store, embedder)
            first = [r.chunk_id for r in retriever.retrieve("document", k=5)]
            second = [r.chunk_id for r in retriever.retrieve("document", k=5)]
            assert first == second
        finally:
            store.close()
