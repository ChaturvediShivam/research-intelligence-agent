"""Chunk store: offset and provenance survival, dense and lexical search."""

from __future__ import annotations

import pytest

from app.retrieval.store import SqliteVecStore, VectorStore, to_fts_query
from app.schemas.source import Chunk
from tests.fixtures.fake_embedder import FakeEmbedder, TermOverlapEmbedder

URL = "https://example.com/doc"


def make_chunks(texts: list[str]) -> list[Chunk]:
    """Chunks with plausible, non-overlapping offsets."""
    chunks, cursor = [], 0
    for i, text in enumerate(texts):
        chunks.append(
            Chunk(
                source_url=URL,
                index=i,
                text=text,
                start_char=cursor,
                end_char=cursor + len(text),
                section=f"Section {i}" if i % 2 == 0 else None,
            )
        )
        cursor += len(text) + 2
    return chunks


@pytest.fixture
def store() -> SqliteVecStore:
    s = SqliteVecStore(":memory:", dimension=8)
    yield s
    s.close()


class TestOffsetAndProvenanceSurvival:
    def test_round_trip_preserves_every_field(self, store: SqliteVecStore) -> None:
        """Citation verification slices the source at these offsets later, so
        losing them in storage would break the whole guarantee."""
        chunks = make_chunks(["first chunk text", "second chunk text"])
        store.add(chunks, FakeEmbedder(8).embed_documents([c.text for c in chunks]))

        restored = store.get_chunks([1, 2])
        assert len(restored) == 2
        for original, recovered in zip(chunks, restored.values(), strict=True):
            assert recovered.text == original.text
            assert recovered.start_char == original.start_char
            assert recovered.end_char == original.end_char
            assert recovered.index == original.index
            assert str(recovered.source_url) == str(original.source_url)
            assert recovered.section == original.section

    def test_offsets_still_verify_against_the_source_after_a_round_trip(
        self, store: SqliteVecStore
    ) -> None:
        source_text = "alpha beta gamma\n\ndelta epsilon zeta"
        chunk = Chunk(
            source_url=URL,
            index=0,
            text="alpha beta gamma",
            start_char=0,
            end_char=16,
            section=None,
        )
        store.add([chunk], FakeEmbedder(8).embed_documents([chunk.text]))
        recovered = store.get_chunks([1])[1]
        assert recovered.verify_against(source_text)

    def test_section_none_round_trips_as_none(self, store: SqliteVecStore) -> None:
        chunk = Chunk(source_url=URL, index=0, text="t", start_char=0, end_char=1, section=None)
        store.add([chunk], FakeEmbedder(8).embed_documents(["t"]))
        assert store.get_chunks([1])[1].section is None


class TestAddValidation:
    def test_mismatched_lengths_are_rejected(self, store: SqliteVecStore) -> None:
        chunks = make_chunks(["a", "b"])
        with pytest.raises(ValueError, match="correspond one to one"):
            store.add(chunks, FakeEmbedder(8).embed_documents(["a"]))

    def test_wrong_dimension_is_rejected(self, store: SqliteVecStore) -> None:
        """A silent dimension mismatch would corrupt the vector table."""
        chunks = make_chunks(["a"])
        with pytest.raises(ValueError, match="dimensions"):
            store.add(chunks, [[0.1, 0.2]])

    def test_empty_add_is_a_no_op(self, store: SqliteVecStore) -> None:
        store.add([], [])
        assert store.count() == 0

    def test_count_reflects_inserts(self, store: SqliteVecStore) -> None:
        chunks = make_chunks(["a", "b", "c"])
        store.add(chunks, FakeEmbedder(8).embed_documents([c.text for c in chunks]))
        assert store.count() == 3


class TestDenseSearch:
    def test_returns_the_nearest_chunk_first(self) -> None:
        vocabulary = ["premium", "solvency", "telematics"]
        embedder = TermOverlapEmbedder(vocabulary)
        store = SqliteVecStore(":memory:", dimension=embedder.dimension)
        try:
            chunks = make_chunks(
                [
                    "gross written premium figures",
                    "solvency capital requirement",
                    "telematics underwriting data",
                ]
            )
            store.add(chunks, embedder.embed_documents([c.text for c in chunks]))

            hits = store.search_dense(embedder.embed_query("premium"), k=3)
            assert hits
            top = store.get_chunks([hits[0].chunk_id])[hits[0].chunk_id]
            assert "premium" in top.text
        finally:
            store.close()

    def test_k_limits_results(self, store: SqliteVecStore) -> None:
        chunks = make_chunks([f"chunk {i}" for i in range(10)])
        store.add(chunks, FakeEmbedder(8).embed_documents([c.text for c in chunks]))
        assert len(store.search_dense(FakeEmbedder(8).embed_query("chunk 1"), k=4)) == 4

    def test_empty_store_returns_nothing(self, store: SqliteVecStore) -> None:
        assert store.search_dense(FakeEmbedder(8).embed_query("x"), k=5) == []

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k(self, store: SqliteVecStore, k: int) -> None:
        assert store.search_dense(FakeEmbedder(8).embed_query("x"), k) == []

    def test_scores_are_higher_is_better(self, store: SqliteVecStore) -> None:
        """Distance is converted to similarity so fusion needs no sign rules."""
        chunks = make_chunks(["alpha", "beta", "gamma"])
        store.add(chunks, FakeEmbedder(8).embed_documents([c.text for c in chunks]))
        hits = store.search_dense(FakeEmbedder(8).embed_query("alpha"), k=3)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 < s <= 1.0 for s in scores)


class TestLexicalSearch:
    def test_matches_an_exact_term(self, store: SqliteVecStore) -> None:
        chunks = make_chunks(
            ["gross written premium reached one billion", "telematics underwriting"]
        )
        store.add(chunks, FakeEmbedder(8).embed_documents([c.text for c in chunks]))
        hits = store.search_lexical("premium", k=5)
        assert len(hits) == 1
        assert "premium" in store.get_chunks([hits[0].chunk_id])[hits[0].chunk_id].text

    def test_stemming_matches_a_variant(self, store: SqliteVecStore) -> None:
        """The porter tokenizer is configured, so 'insurers' finds 'insurer'."""
        chunks = make_chunks(["the insurer must hold capital"])
        store.add(chunks, FakeEmbedder(8).embed_documents([c.text for c in chunks]))
        assert store.search_lexical("insurers", k=5)

    def test_scores_are_higher_is_better(self, store: SqliteVecStore) -> None:
        chunks = make_chunks(["premium premium premium", "premium once", "nothing relevant here"])
        store.add(chunks, FakeEmbedder(8).embed_documents([c.text for c in chunks]))
        hits = store.search_lexical("premium", k=5)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_no_match_returns_nothing(self, store: SqliteVecStore) -> None:
        chunks = make_chunks(["alpha beta"])
        store.add(chunks, FakeEmbedder(8).embed_documents([c.text for c in chunks]))
        assert store.search_lexical("zzzznotpresent", k=5) == []

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k(self, store: SqliteVecStore, k: int) -> None:
        assert store.search_lexical("premium", k) == []


class TestFtsQuerySanitisation:
    """A research question is full of characters FTS5 treats as syntax."""

    @pytest.mark.parametrize(
        "raw",
        [
            "what's the FCA's position?",
            'a "quoted phrase" here',
            "term1 OR term2 AND term3",
            "NEAR(a b)",
            "hyphen-separated words",
            "trailing* wildcard",
            "col:umn syntax",
            "parens (and) brackets [x]",
            "^caret and -minus",
        ],
    )
    def test_hostile_queries_do_not_raise(self, store: SqliteVecStore, raw: str) -> None:
        chunks = make_chunks(["the fca position on quoted phrases and hyphens"])
        store.add(chunks, FakeEmbedder(8).embed_documents([c.text for c in chunks]))
        store.search_lexical(raw, k=5)  # must not raise

    def test_fts_keywords_become_terms_not_operators(self) -> None:
        """OR/AND/NEAR must be quoted as terms, not left as FTS5 operators."""
        assert to_fts_query("alpha OR beta") == '"alpha" OR "or" OR "beta"'
        assert to_fts_query("alpha NEAR beta") == '"alpha" OR "near" OR "beta"'

    def test_single_character_terms_are_dropped(self) -> None:
        """Deliberate: one-character tokens match almost everything in BM25.

        The cost is that an entity named by a single letter is lost from the
        lexical half of the query; the dense half still sees the full text.
        """
        assert to_fts_query("a OR b") == '"or"'
        assert to_fts_query("X Corp revenue") == '"corp" OR "revenue"'

    def test_terms_are_lowercased_and_deduplicated(self) -> None:
        assert to_fts_query("Premium premium PREMIUM") == '"premium"'

    def test_single_characters_are_dropped(self) -> None:
        assert to_fts_query("a premium") == '"premium"'

    def test_punctuation_only_yields_empty(self) -> None:
        assert to_fts_query("?!...") == ""
        assert to_fts_query("") == ""

    def test_empty_query_returns_no_results_rather_than_raising(
        self, store: SqliteVecStore
    ) -> None:
        chunks = make_chunks(["content"])
        store.add(chunks, FakeEmbedder(8).embed_documents([c.text for c in chunks]))
        assert store.search_lexical("???", k=5) == []


class TestGetChunks:
    def test_unknown_ids_are_absent_not_an_error(self, store: SqliteVecStore) -> None:
        assert store.get_chunks([999]) == {}

    def test_empty_request(self, store: SqliteVecStore) -> None:
        assert store.get_chunks([]) == {}


class TestProtocolConformance:
    def test_sqlite_store_satisfies_the_protocol(self, store: SqliteVecStore) -> None:
        """ADR-004's pgvector swap path is only real if the protocol is met."""
        assert isinstance(store, VectorStore)


class TestContextManager:
    def test_closes_on_exit(self) -> None:
        with SqliteVecStore(":memory:", dimension=8) as s:
            assert s.count() == 0
