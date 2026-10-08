"""Structure-aware chunking with exact character-offset preservation.

The design decision that matters here: **a chunk's text is always a slice of
the source**, never a rebuilt string. `Chunk.text = source[start:end]`, taken
verbatim. Joining paragraphs back together with `"\\n\\n".join(...)` would be
the obvious implementation and would quietly break the invariant the moment
the original separator was anything else — three newlines, a tab, a trailing
space. Slicing makes `chunk.verify_against(source.text)` true by construction
rather than by diligence, and deterministic citation verification (ADR-002)
rests on it.

Chunk size is expressed in tokens for configuration, but measured in
characters here.

ponytail: 4-chars-per-token heuristic, not a real tokenizer. Good enough for
English prose and costs nothing; `messages.count_tokens` would be exact but
adds a network call per chunk. Upgrade if measured retrieval quality turns out
to depend on precise chunk sizing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas.source import Chunk, FetchedSource

CHARS_PER_TOKEN = 4

# A heading, heuristically: a short line with no sentence-ending punctuation,
# optionally markdown-prefixed. trafilatura returns plain text, so there is no
# markup to rely on.
_HEADING_MAX_CHARS = 120
_SENTENCE_END = (".", "!", "?", ":", ";", ",")


@dataclass(frozen=True, slots=True)
class Block:
    """A paragraph-ish span of the source, with its exact offsets."""

    start: int
    end: int
    is_heading: bool

    @property
    def length(self) -> int:
        return self.end - self.start


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > _HEADING_MAX_CHARS:
        return False
    if stripped.startswith("#"):
        return True
    if stripped.endswith(_SENTENCE_END):
        return False
    # A line of a few words with no terminal punctuation, in a document that
    # otherwise uses punctuation, is usually a heading.
    return len(stripped.split()) <= 12


def split_blocks(text: str) -> list[Block]:
    """Split text into offset-tracked blocks on blank lines.

    Offsets are into `text` exactly as given. Separator whitespace belongs to
    no block, which is what lets a chunk be reassembled as a single slice
    spanning several blocks without inventing or losing characters.
    """
    blocks: list[Block] = []
    for match in re.finditer(r"[^\n]+(?:\n[^\n]+)*", text):
        start, end = match.start(), match.end()
        segment = text[start:end]
        # A run of spaces or tabs matches the pattern but carries no content.
        # Emitting it would cost an embedding and return a meaningless chunk.
        if not segment.strip():
            continue
        # A block may be a heading followed by its paragraph on the next line;
        # treat the whole run as a heading only if every line looks like one.
        lines = segment.split("\n")
        is_heading = len(lines) == 1 and _looks_like_heading(lines[0])
        blocks.append(Block(start=start, end=end, is_heading=is_heading))
    return blocks


def _hard_split(text: str, block: Block, max_chars: int) -> list[Block]:
    """Split an oversized block at word boundaries, preserving offsets."""
    pieces: list[Block] = []
    cursor = block.start
    while cursor < block.end:
        limit = min(cursor + max_chars, block.end)
        if limit < block.end:
            # Prefer a space near the limit so a word is not cut in half.
            window = text.rfind(" ", cursor + max_chars // 2, limit)
            if window != -1:
                limit = window
        pieces.append(Block(start=cursor, end=limit, is_heading=False))
        cursor = limit
        # Skip the separator space so the next piece does not start on it.
        while cursor < block.end and text[cursor] == " ":
            cursor += 1
    return pieces


def chunk_source(
    source: FetchedSource,
    *,
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Chunk a fetched source, preserving offsets into `source.text`.

    Blocks are packed greedily up to the size budget. Overlap is applied by
    re-including trailing blocks of the previous chunk, so overlap always
    falls on a block boundary — an overlap that cut mid-sentence would produce
    chunks that embed poorly and read badly in a citation.
    """
    text = source.text
    if not text:
        return []

    max_chars = max(chunk_tokens * CHARS_PER_TOKEN, 1)
    overlap_chars = max(overlap_tokens * CHARS_PER_TOKEN, 0)

    blocks: list[Block] = []
    for block in split_blocks(text):
        if block.length > max_chars:
            blocks.extend(_hard_split(text, block, max_chars))
        else:
            blocks.append(block)

    if not blocks:
        return []

    chunks: list[Chunk] = []
    current: list[Block] = []
    current_section: str | None = None
    section_for_chunk: str | None = None

    def flush() -> None:
        """Emit the accumulated blocks as one chunk."""
        if not current:
            return
        start, end = current[0].start, current[-1].end
        chunks.append(
            Chunk(
                source_url=source.final_url,
                index=len(chunks),
                # The invariant: a verbatim slice, never a rejoin.
                text=text[start:end],
                start_char=start,
                end_char=end,
                section=section_for_chunk,
            )
        )

    for block in blocks:
        if block.is_heading:
            current_section = text[block.start : block.end].lstrip("# ").strip()[:300]

        prospective = (block.end - current[0].start) if current else block.length
        if current and prospective > max_chars:
            flush()
            # Carry trailing blocks forward as overlap, newest first.
            carried: list[Block] = []
            carried_chars = 0
            for previous in reversed(current):
                if carried_chars + previous.length > overlap_chars:
                    break
                carried.insert(0, previous)
                carried_chars += previous.length
            current = carried
            section_for_chunk = current_section

        if not current:
            section_for_chunk = current_section
        current.append(block)

    flush()
    return chunks


def verify_chunks(source: FetchedSource, chunks: list[Chunk]) -> list[int]:
    """Return the indexes of any chunks whose offsets do not match their text.

    Used as an assertion in the pipeline, not only in tests: a silent offset
    drift would surface much later as unexplainable citation failures.
    """
    return [c.index for c in chunks if not c.verify_against(source.text)]
