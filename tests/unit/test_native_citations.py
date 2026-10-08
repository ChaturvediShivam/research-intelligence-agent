"""Native citation parsing and document-block alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from app.llm.citations import (
    build_document_blocks,
    parse_cited_response,
    source_ids_in_order,
)
from app.schemas.evidence import source_id_for
from app.schemas.source import FetchedSource, content_hash

TEXT_A = "Acceptance rates averaged 61% across reporting firms in the period."
TEXT_B = "Gross written premium reached 1,650 million pounds in the year."


def _source(url: str, text: str, title: str = "T") -> FetchedSource:
    return FetchedSource(
        url=url,
        final_url=url,
        title=title,
        text=text,
        content_hash=content_hash(text),
        status_code=200,
        byte_length=len(text.encode()),
    )


SOURCES = [
    _source("https://www.fca.org.uk/a", TEXT_A, "FCA data"),
    _source("https://www.abi.org.uk/b", TEXT_B, "ABI data"),
]


@dataclass
class FakeCitation:
    document_index: int
    cited_text: str
    start_char_index: int
    end_char_index: int
    type: str = "char_location"
    document_title: str = "T"


@dataclass
class FakeTextBlock:
    text: str
    citations: list[Any] = field(default_factory=list)
    type: str = "text"


class TestDocumentBlocks:
    def test_sends_canonical_text_verbatim(self) -> None:
        """Any transformation here breaks every returned offset."""
        blocks = build_document_blocks(SOURCES)
        assert blocks[0]["source"]["data"] == TEXT_A
        assert blocks[1]["source"]["data"] == TEXT_B

    def test_citations_are_enabled_on_every_block(self) -> None:
        """The API requires all blocks or none."""
        for block in build_document_blocks(SOURCES):
            assert block["citations"] == {"enabled": True}

    def test_block_shape(self) -> None:
        block = build_document_blocks(SOURCES)[0]
        assert block["type"] == "document"
        assert block["source"]["type"] == "text"
        assert block["source"]["media_type"] == "text/plain"
        assert block["title"] == "FCA data"

    def test_title_falls_back_to_domain(self) -> None:
        untitled = _source("https://example.com/x", "body text", title="")
        assert build_document_blocks([untitled])[0]["title"] == "example.com"

    def test_source_ids_are_positionally_aligned_with_the_blocks(self) -> None:
        """document_index is the only link back to the source."""
        ids = source_ids_in_order(SOURCES)
        assert ids == [
            source_id_for("https://www.fca.org.uk/a"),
            source_id_for("https://www.abi.org.uk/b"),
        ]

    def test_empty_source_list(self) -> None:
        assert build_document_blocks([]) == []


class TestParseCitedResponse:
    def test_maps_a_citation_to_its_source_and_offsets(self) -> None:
        ids = source_ids_in_order(SOURCES)
        start = TEXT_A.index("averaged 61%")
        content = [
            FakeTextBlock(
                text="Acceptance averaged 61%.",
                citations=[
                    FakeCitation(
                        document_index=0,
                        cited_text="averaged 61%",
                        start_char_index=start,
                        end_char_index=start + len("averaged 61%"),
                    )
                ],
            )
        ]
        blocks = parse_cited_response(content, ids)
        assert len(blocks) == 1
        citation = blocks[0].citations[0]
        assert citation.source_id == ids[0]
        assert TEXT_A[citation.start_char : citation.end_char] == "averaged 61%"

    def test_a_citation_on_the_second_document_resolves_correctly(self) -> None:
        """An off-by-one in index mapping would attribute it to the wrong source."""
        ids = source_ids_in_order(SOURCES)
        start = TEXT_B.index("1,650 million")
        content = [
            FakeTextBlock(
                text="Premium reached 1,650 million pounds.",
                citations=[
                    FakeCitation(
                        document_index=1,
                        cited_text="1,650 million",
                        start_char_index=start,
                        end_char_index=start + len("1,650 million"),
                    )
                ],
            )
        ]
        citation = parse_cited_response(content, ids)[0].citations[0]
        assert citation.source_id == ids[1]
        assert TEXT_B[citation.start_char : citation.end_char] == "1,650 million"

    def test_uncited_text_blocks_are_kept_with_no_citations(self) -> None:
        """Connecting prose is legitimate; it just carries no citation."""
        blocks = parse_cited_response(
            [FakeTextBlock(text="In summary,")], source_ids_in_order(SOURCES)
        )
        assert len(blocks) == 1
        assert blocks[0].citations == []

    def test_blank_text_blocks_are_dropped(self) -> None:
        assert parse_cited_response([FakeTextBlock(text="   ")], ["s"]) == []

    def test_non_text_blocks_are_ignored(self) -> None:
        @dataclass
        class Thinking:
            type: str = "thinking"
            thinking: str = ""

        assert parse_cited_response([Thinking()], ["s"]) == []

    def test_dict_shaped_blocks_are_accepted(self) -> None:
        """The SDK may hand back plain dicts depending on the call path."""
        ids = source_ids_in_order(SOURCES)
        start = TEXT_A.index("61%")
        content = [
            {
                "type": "text",
                "text": "x",
                "citations": [
                    {
                        "type": "char_location",
                        "document_index": 0,
                        "cited_text": "61%",
                        "start_char_index": start,
                        "end_char_index": start + 3,
                    }
                ],
            }
        ]
        assert parse_cited_response(content, ids)[0].citations[0].source_id == ids[0]


class TestUnverifiableCitationsAreDropped:
    @pytest.mark.parametrize("location_type", ["page_location", "content_block_location"])
    def test_non_char_locations_are_dropped(self, location_type: str) -> None:
        """A page citation cannot be verified by slicing text.

        Keeping one would put an unverifiable citation into the report, which
        is precisely what the verifier exists to prevent.
        """
        content = [
            FakeTextBlock(
                text="x",
                citations=[
                    FakeCitation(
                        document_index=0,
                        cited_text="q",
                        start_char_index=0,
                        end_char_index=1,
                        type=location_type,
                    )
                ],
            )
        ]
        assert parse_cited_response(content, ["s"])[0].citations == []

    def test_document_index_out_of_range_is_dropped_not_guessed(self) -> None:
        """Guessing would attribute a quote to the wrong source."""
        content = [
            FakeTextBlock(
                text="x",
                citations=[
                    FakeCitation(
                        document_index=9,
                        cited_text="q",
                        start_char_index=0,
                        end_char_index=1,
                    )
                ],
            )
        ]
        assert parse_cited_response(content, ["s0", "s1"])[0].citations == []

    @pytest.mark.parametrize(("start", "end"), [(5, 5), (10, 3)])
    def test_invalid_offsets_are_dropped(self, start: int, end: int) -> None:
        content = [
            FakeTextBlock(
                text="x",
                citations=[
                    FakeCitation(
                        document_index=0,
                        cited_text="q",
                        start_char_index=start,
                        end_char_index=end,
                    )
                ],
            )
        ]
        assert parse_cited_response(content, ["s"])[0].citations == []

    def test_incomplete_citation_is_dropped(self) -> None:
        content = [
            {
                "type": "text",
                "text": "x",
                "citations": [{"type": "char_location", "document_index": 0}],
            }
        ]
        assert parse_cited_response(content, ["s"])[0].citations == []
