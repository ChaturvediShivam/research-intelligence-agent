"""Chunking, and the offset invariant deterministic citation rests on.

The property under test throughout: for every chunk,

    source.text[chunk.start_char : chunk.end_char] == chunk.text

If that ever fails, citation verification (ADR-002) silently degrades into
rejecting correct citations.
"""

from __future__ import annotations

import pytest

from app.retrieval.chunking import (
    CHARS_PER_TOKEN,
    chunk_source,
    split_blocks,
    verify_chunks,
)
from app.schemas.source import FetchedSource, content_hash


def make_source(text: str) -> FetchedSource:
    return FetchedSource(
        url="https://example.com/a",
        final_url="https://example.com/a",
        title="t",
        text=text,
        content_hash=content_hash(text),
        status_code=200,
        byte_length=len(text.encode()),
    )


PROSE = "\n\n".join(
    f"Paragraph {i} discusses a specific aspect of the market in detail, with "
    f"enough words to be a realistic unit of prose rather than a stub."
    for i in range(30)
)


class TestOffsetInvariant:
    def test_every_chunk_is_a_verbatim_slice(self) -> None:
        source = make_source(PROSE)
        chunks = chunk_source(source, chunk_tokens=100, overlap_tokens=20)
        assert chunks
        for chunk in chunks:
            assert source.text[chunk.start_char : chunk.end_char] == chunk.text
            assert chunk.verify_against(source.text)
        assert verify_chunks(source, chunks) == []

    @pytest.mark.parametrize("chunk_tokens", [16, 64, 128, 512, 2048])
    @pytest.mark.parametrize("overlap_tokens", [0, 8, 64])
    def test_invariant_holds_across_configurations(
        self, chunk_tokens: int, overlap_tokens: int
    ) -> None:
        source = make_source(PROSE)
        chunks = chunk_source(source, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
        assert verify_chunks(source, chunks) == []

    @pytest.mark.parametrize(
        "text",
        [
            "Single short paragraph.",
            "One.\n\nTwo.\n\nThree.",
            # Irregular separators are the case a rejoin implementation breaks on.
            "A.\n\n\n\nB.\n\nC.",
            "Line with trailing spaces   \n\nNext.",
            "Unicode — em dashes, curly ’quotes’, and accents café.\n\nNext.",
            "Tabs\there.\n\nAnd\tthere.",
            "A" * 5000,
            "short\n\n" + "B" * 3000,
        ],
    )
    def test_invariant_holds_on_awkward_text(self, text: str) -> None:
        source = make_source(text)
        chunks = chunk_source(source, chunk_tokens=64, overlap_tokens=8)
        assert verify_chunks(source, chunks) == []
        # Nothing is invented: every chunk's text exists in the source.
        for chunk in chunks:
            assert chunk.text in source.text

    def test_a_tampered_chunk_is_detected(self) -> None:
        """verify_chunks must actually catch a mismatch, not always return []."""
        source = make_source(PROSE)
        chunks = chunk_source(source, chunk_tokens=100, overlap_tokens=0)
        tampered = chunks[1].model_copy(update={"text": "not what is at those offsets"})
        assert tampered.verify_against(source.text) is False
        assert verify_chunks(source, [tampered]) == [tampered.index]


class TestCoverage:
    def test_chunks_cover_the_document_start_to_end(self) -> None:
        source = make_source(PROSE)
        chunks = chunk_source(source, chunk_tokens=100, overlap_tokens=0)
        assert chunks[0].start_char == 0
        assert chunks[-1].end_char == len(source.text)

    def test_no_content_is_dropped_between_chunks(self) -> None:
        """With zero overlap, consecutive chunks must be contiguous in content.

        Only separator whitespace may fall between them.
        """
        source = make_source(PROSE)
        chunks = chunk_source(source, chunk_tokens=100, overlap_tokens=0)
        for previous, nxt in zip(chunks, chunks[1:], strict=False):
            gap = source.text[previous.end_char : nxt.start_char]
            assert gap.strip() == "", f"dropped content: {gap!r}"

    def test_indexes_are_sequential(self) -> None:
        chunks = chunk_source(make_source(PROSE), chunk_tokens=100)
        assert [c.index for c in chunks] == list(range(len(chunks)))


class TestOverlap:
    def test_overlap_repeats_content_between_chunks(self) -> None:
        source = make_source(PROSE)
        with_overlap = chunk_source(source, chunk_tokens=100, overlap_tokens=40)
        without = chunk_source(source, chunk_tokens=100, overlap_tokens=0)
        # Overlap means more chunks for the same document.
        assert len(with_overlap) >= len(without)
        # And consecutive chunks share a boundary region.
        assert with_overlap[1].start_char < with_overlap[0].end_char

    def test_zero_overlap_produces_no_repetition(self) -> None:
        source = make_source(PROSE)
        chunks = chunk_source(source, chunk_tokens=100, overlap_tokens=0)
        for previous, nxt in zip(chunks, chunks[1:], strict=False):
            assert nxt.start_char >= previous.end_char


class TestOversizedBlocks:
    def test_a_paragraph_larger_than_the_budget_is_split(self) -> None:
        words = " ".join(f"word{i}" for i in range(2000))
        source = make_source(words)
        chunks = chunk_source(source, chunk_tokens=64, overlap_tokens=0)
        assert len(chunks) > 1
        assert verify_chunks(source, chunks) == []
        budget = 64 * CHARS_PER_TOKEN
        # Allow a small margin: splits land on word boundaries.
        assert all(c.char_length <= budget + 50 for c in chunks)

    def test_split_prefers_word_boundaries(self) -> None:
        words = " ".join(f"word{i:04d}" for i in range(500))
        source = make_source(words)
        chunks = chunk_source(source, chunk_tokens=32, overlap_tokens=0)
        # No chunk should start or end mid-word for text that is all words.
        for chunk in chunks:
            assert not chunk.text.startswith(" ")

    def test_single_enormous_word_still_terminates(self) -> None:
        """A pathological input must not loop forever."""
        source = make_source("x" * 10_000)
        chunks = chunk_source(source, chunk_tokens=16, overlap_tokens=0)
        assert chunks
        assert verify_chunks(source, chunks) == []


class TestSections:
    def test_heading_is_attached_to_following_chunks(self) -> None:
        text = (
            "Market Overview\n\n"
            + "Body text about the market. " * 40
            + "\n\nRegulatory Position\n\n"
            + "Body text about regulation. " * 40
        )
        source = make_source(text)
        chunks = chunk_source(source, chunk_tokens=60, overlap_tokens=0)
        sections = {c.section for c in chunks if c.section}
        assert "Market Overview" in sections
        assert "Regulatory Position" in sections

    def test_markdown_heading_prefix_is_stripped(self) -> None:
        text = "## Key Findings\n\n" + "Detail sentence here. " * 40
        chunks = chunk_source(make_source(text), chunk_tokens=50)
        assert any(c.section == "Key Findings" for c in chunks)

    def test_prose_is_not_mistaken_for_a_heading(self) -> None:
        text = "This sentence ends with a period and is ordinary prose.\n\nNext."
        blocks = split_blocks(text)
        assert blocks[0].is_heading is False


class TestEdgeCases:
    def test_empty_text_yields_no_chunks(self) -> None:
        assert chunk_source(make_source("")) == []

    def test_whitespace_only_yields_no_chunks(self) -> None:
        assert chunk_source(make_source("   \n\n  \n")) == []

    def test_single_short_paragraph_is_one_chunk(self) -> None:
        source = make_source("Just one short paragraph of text.")
        chunks = chunk_source(source, chunk_tokens=512)
        assert len(chunks) == 1
        assert chunks[0].text == source.text
        assert chunks[0].start_char == 0

    def test_chunks_carry_the_final_url(self) -> None:
        source = make_source(PROSE)
        chunks = chunk_source(source, chunk_tokens=100)
        assert all(str(c.source_url) == str(source.final_url) for c in chunks)


class TestSplitBlocks:
    def test_offsets_select_their_own_segments(self) -> None:
        text = "First block.\n\nSecond block.\n\n\nThird."
        for block in split_blocks(text):
            segment = text[block.start : block.end]
            assert segment.strip() == segment
            assert segment

    def test_blank_lines_belong_to_no_block(self) -> None:
        text = "A.\n\n\n\nB."
        blocks = split_blocks(text)
        assert len(blocks) == 2
        assert text[blocks[0].start : blocks[0].end] == "A."
        assert text[blocks[1].start : blocks[1].end] == "B."
