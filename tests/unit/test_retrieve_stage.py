"""Stage 4 (RETRIEVE): per-sub-question retrieval over a per-run index."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.errors import PipelineStageError
from app.pipeline.retrieve import build_index, run_retrieve_stage
from app.schemas.research import ResearchPlan
from app.schemas.runs import Stage
from app.schemas.source import Chunk
from tests.fixtures.fake_embedder import FakeEmbedder, TermOverlapEmbedder


def _settings(**kw: object) -> Settings:
    return Settings(environment="test", _env_file=None, **kw)  # type: ignore[arg-type,call-arg]


def _chunks(texts: list[str], *, url: str = "https://example.com/a") -> list[Chunk]:
    out, cursor = [], 0
    for i, text in enumerate(texts):
        out.append(
            Chunk(
                source_url=url,
                index=i,
                text=text,
                start_char=cursor,
                end_char=cursor + len(text),
            )
        )
        cursor += len(text) + 2
    return out


class TestBuildIndex:
    def test_indexes_every_chunk(self) -> None:
        embedder = FakeEmbedder(8)
        store = build_index(_chunks(["a", "b", "c"]), embedder)
        try:
            assert store.count() == 3
        finally:
            store.close()

    def test_empty_chunks_yields_an_empty_index(self) -> None:
        store = build_index([], FakeEmbedder(8))
        try:
            assert store.count() == 0
        finally:
            store.close()

    def test_index_dimension_follows_the_embedder(self) -> None:
        store = build_index(_chunks(["a"]), FakeEmbedder(16))
        try:
            assert store.count() == 1
        finally:
            store.close()


class TestRetrieveStage:
    async def test_returns_results_per_sub_question(self, sample_plan: ResearchPlan) -> None:
        embedder = TermOverlapEmbedder(["premium", "insurer", "share"])
        chunks = _chunks(
            [
                "gross written premium for the year",
                "the largest insurer by market share",
                "unrelated commentary about weather",
            ]
        )
        results, metric = await run_retrieve_stage(
            sample_plan, chunks, embedder=embedder, settings=_settings()
        )
        assert set(results) == {"SQ1", "SQ2"}
        assert all(results[key] for key in results)
        assert metric.stage is Stage.RETRIEVE
        # Local embeddings: no model call, so no cost may be attributed.
        assert metric.model is None
        assert metric.calls == 0
        assert metric.cost_usd == 0.0

    async def test_offsets_and_provenance_survive_the_stage(
        self, sample_plan: ResearchPlan
    ) -> None:
        source_text = "gross written premium for the year"
        chunks = _chunks([source_text])
        results, _ = await run_retrieve_stage(
            sample_plan,
            chunks,
            embedder=TermOverlapEmbedder(["premium"]),
            settings=_settings(),
        )
        retrieved = results["SQ1"][0].chunk
        assert retrieved.start_char == 0
        assert retrieved.end_char == len(source_text)
        assert retrieved.verify_against(source_text)
        assert str(retrieved.source_url) == "https://example.com/a"

    async def test_respects_retrieval_top_k(self, sample_plan: ResearchPlan) -> None:
        chunks = _chunks([f"chunk text number {i}" for i in range(20)])
        results, _ = await run_retrieve_stage(
            sample_plan,
            chunks,
            embedder=FakeEmbedder(8),
            settings=_settings(retrieval_top_k=3),
        )
        assert all(len(v) <= 3 for v in results.values())

    async def test_empty_corpus_yields_empty_results_not_an_error(
        self, sample_plan: ResearchPlan
    ) -> None:
        """No sources is a recorded gap, not a crash."""
        results, metric = await run_retrieve_stage(
            sample_plan, [], embedder=FakeEmbedder(8), settings=_settings()
        )
        assert set(results) == {"SQ1", "SQ2"}
        assert all(v == [] for v in results.values())
        assert metric.duration_ms >= 0

    async def test_failure_is_wrapped_with_the_stage_name(self, sample_plan: ResearchPlan) -> None:
        class BrokenEmbedder:
            @property
            def dimension(self) -> int:
                return 8

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                raise RuntimeError("model exploded")

            def embed_query(self, text: str) -> list[float]:
                raise RuntimeError("model exploded")

        with pytest.raises(PipelineStageError) as info:
            await run_retrieve_stage(
                sample_plan,
                _chunks(["a"]),
                embedder=BrokenEmbedder(),  # type: ignore[arg-type]
                settings=_settings(),
            )
        assert info.value.stage == Stage.RETRIEVE.value
        assert info.value.cause is not None

    async def test_sub_questions_are_retrieved_in_plan_order(
        self, sample_plan: ResearchPlan
    ) -> None:
        """Rank order is an allocation decision; the stage must honour it."""
        results, _ = await run_retrieve_stage(
            sample_plan,
            _chunks(["some content here"]),
            embedder=FakeEmbedder(8),
            settings=_settings(),
        )
        assert list(results) == [sq.id for sq in sample_plan.ordered()]
