"""Stage 5 — EXTRACT. Per-chunk evidence extraction.

**The model never supplies character offsets.** It returns a verbatim quote;
this module locates that quote in the chunk and computes the absolute offsets
itself. That split is deliberate and load-bearing:

- Language models are unreliable at character arithmetic. Asking for offsets
  produces numbers that look plausible and are wrong, and a wrong offset is
  indistinguishable from a fabricated one at verification time.
- Locating the quote in code makes fabrication **structurally impossible at
  extraction**: a quote that cannot be found verbatim in the chunk is dropped
  along with its evidence item. A model that paraphrases instead of quoting
  loses the item rather than producing an unverifiable one.

So stage 7 verifies citations against the source, and stage 5 already
guarantees every offset it emits was computed from a real match. The two
checks are independent, which is the point — stage 7 still catches a stage 5
bug.

Runs on Haiku (ADR-007): one call per chunk, narrow scope, schema-constrained.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.errors import PipelineStageError
from app.llm.client import LLMClient
from app.llm.context import frame_untrusted, load_prompt
from app.schemas.evidence import EvidenceGrade, EvidenceItem, Quote, source_id_for
from app.schemas.runs import Stage, StageMetric, TokenUsage
from app.schemas.source import Chunk

logger = structlog.get_logger(__name__)

PROMPT_VERSION = "extract.v1"

# Characters that vary between a page and a copy of it without the meaning
# changing. The locator tolerates these; it tolerates nothing else.
_QUOTE_CLASS = "['‘’′‛]"
_DQUOTE_CLASS = '["“”„‟″«»]'
_DASH_CLASS = "[-‐‑‒–—―−]"


class EvidenceDraft(BaseModel):
    """What the model returns for one piece of evidence. No offsets."""

    statement: str = Field(
        min_length=4,
        max_length=1000,
        description=(
            "What this passage establishes, in one sentence, stating only "
            "what the passage supports."
        ),
    )
    verbatim_quote: str = Field(
        min_length=8,
        max_length=2000,
        description=(
            "The exact words from the passage supporting the statement, "
            "copied character for character. A quote not found verbatim in "
            "the passage is discarded."
        ),
    )
    grade: EvidenceGrade = Field(
        description=(
            "talk if someone said it, behavior if someone did something "
            "costly, money if someone paid."
        )
    )


class ChunkExtraction(BaseModel):
    """Stage 5's per-chunk output. An empty list is a valid result."""

    items: list[EvidenceDraft] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Zero or more evidence items. Zero is correct and common: most "
            "passages contain no evidence for a given sub-question."
        ),
    )


@dataclass(slots=True)
class ExtractionResult:
    """Evidence that survived location, plus what was discarded and why."""

    items: list[EvidenceItem] = field(default_factory=list)
    # Quotes the model returned that could not be found in their chunk. Kept
    # because a rising rate here means the extractor is paraphrasing, which is
    # a prompt problem worth seeing rather than a silently smaller result.
    unlocatable: list[tuple[str, str]] = field(default_factory=list)

    @property
    def unlocatable_rate(self) -> float:
        total = len(self.items) + len(self.unlocatable)
        return len(self.unlocatable) / total if total else 0.0


def _flexible_pattern(quote: str) -> re.Pattern[str]:
    """Build a whitespace- and punctuation-tolerant pattern for a quote.

    Tokens are matched in order, separated by any whitespace, so a quote
    copied across a line break still locates. Quote and dash characters match
    their typographic variants. Everything else must match exactly.
    """
    tokens = []
    for token in quote.split():
        escaped = re.escape(token)
        # re.escape leaves these literal; widen them to their variant classes.
        escaped = re.sub(r"\\?'", _QUOTE_CLASS, escaped)
        escaped = re.sub(r'\\?"', _DQUOTE_CLASS, escaped)
        escaped = re.sub(r"\\?-", _DASH_CLASS, escaped)
        tokens.append(escaped)
    return re.compile(r"\s+".join(tokens), re.IGNORECASE)


def locate_quote(chunk: Chunk, quote_text: str) -> Quote | None:
    """Find a quote in a chunk and return it with absolute source offsets.

    Returns None when the quote is not present — which is how a paraphrase is
    rejected rather than stored as unverifiable evidence.
    """
    cleaned = quote_text.strip()
    if not cleaned:
        return None

    source_id = source_id_for(str(chunk.source_url))

    # Exact match first: the common case, and the only one needing no
    # tolerance at all.
    index = chunk.text.find(cleaned)
    if index != -1:
        start = chunk.start_char + index
        return Quote(
            source_id=source_id,
            start_char=start,
            end_char=start + len(cleaned),
            # Store what the source actually says, not what the model sent.
            text=chunk.text[index : index + len(cleaned)],
        )

    match = _flexible_pattern(cleaned).search(chunk.text)
    if match is None:
        return None

    start = chunk.start_char + match.start()
    return Quote(
        source_id=source_id,
        start_char=start,
        end_char=chunk.start_char + match.end(),
        # The source's own characters, so stage 7 compares like with like.
        text=chunk.text[match.start() : match.end()],
    )


def build_user_content(sub_question: str, chunk: Chunk) -> str:
    """Assemble the volatile half of an extraction call.

    The chunk is framed as untrusted data (the boundary established in M2), so
    a page instructing the extractor is inert text inside the fence.
    """
    label = f"{chunk.source_url}"
    if chunk.section:
        label = f"{label} — {chunk.section}"
    return (
        f"Sub-question you are extracting evidence for:\n{sub_question}\n\n"
        f"{frame_untrusted(chunk.text, source_label=label)}"
    )


async def extract_from_chunk(
    sub_question_id: str,
    sub_question: str,
    chunk: Chunk,
    *,
    client: LLMClient,
    settings: Settings,
    sequence: int,
) -> tuple[list[EvidenceItem], list[tuple[str, str]], TokenUsage, float]:
    """Extract evidence from one chunk. Returns items, discards, usage, cost."""
    result = await client.structured(
        model=settings.extraction_model,
        output_model=ChunkExtraction,
        system=load_prompt(PROMPT_VERSION),
        user_content=build_user_content(sub_question, chunk),
        # Narrow, schema-constrained work, so low effort is what this stage
        # wants. Haiku 4.5 rejects the effort parameter outright, and the
        # client drops it from the capability table rather than erroring —
        # the request is correct either way if the model is changed.
        effort="low",
        max_tokens=2048,
    )

    items: list[EvidenceItem] = []
    unlocatable: list[tuple[str, str]] = []

    for position, draft in enumerate(result.value.items):
        quote = locate_quote(chunk, draft.verbatim_quote)
        if quote is None:
            unlocatable.append((draft.verbatim_quote[:200], draft.statement[:200]))
            logger.info(
                "evidence_quote_unlocatable",
                sub_question_id=sub_question_id,
                chunk_index=chunk.index,
                quote=draft.verbatim_quote[:120],
            )
            continue
        items.append(
            EvidenceItem(
                evidence_id=f"E{sequence}_{chunk.index}_{position}",
                sub_question_id=sub_question_id,
                statement=draft.statement,
                quote=quote,
                grade=draft.grade,
                chunk_index=chunk.index,
            )
        )

    return items, unlocatable, result.usage, result.cost_usd


async def run_extract_stage(
    retrieved: dict[str, list[Chunk]],
    sub_questions: dict[str, str],
    *,
    client: LLMClient,
    settings: Settings,
    max_concurrency: int = 4,
) -> tuple[ExtractionResult, StageMetric]:
    """Extract evidence from every retrieved chunk, per sub-question.

    Calls run concurrently with a bound: unbounded concurrency over dozens of
    chunks is the fastest way to hit a rate limit, and the retry that follows
    costs more wall-clock than the bound does.
    """
    started = time.perf_counter()
    result = ExtractionResult()
    usage = TokenUsage()
    cost = 0.0
    calls = 0

    semaphore = asyncio.Semaphore(max_concurrency)

    async def one(
        sub_question_id: str, question: str, chunk: Chunk, sequence: int
    ) -> tuple[list[EvidenceItem], list[tuple[str, str]], TokenUsage, float]:
        async with semaphore:
            return await extract_from_chunk(
                sub_question_id,
                question,
                chunk,
                client=client,
                settings=settings,
                sequence=sequence,
            )

    tasks = [
        one(sub_question_id, sub_questions.get(sub_question_id, ""), chunk, sequence)
        for sequence, (sub_question_id, chunks) in enumerate(retrieved.items())
        for chunk in chunks
    ]

    if not tasks:
        metric = StageMetric(
            stage=Stage.EXTRACT,
            model=settings.extraction_model,
            duration_ms=int((time.perf_counter() - started) * 1000),
            calls=0,
        )
        return result, metric

    try:
        outcomes = await asyncio.gather(*tasks)
    except Exception as exc:
        raise PipelineStageError(
            Stage.EXTRACT.value, f"Evidence extraction failed: {exc}", cause=exc
        ) from exc

    for items, unlocatable, call_usage, call_cost in outcomes:
        result.items.extend(items)
        result.unlocatable.extend(unlocatable)
        usage = usage + call_usage
        cost += call_cost
        calls += 1

    metric = StageMetric(
        stage=Stage.EXTRACT,
        model=settings.extraction_model,
        duration_ms=int((time.perf_counter() - started) * 1000),
        usage=usage,
        cost_usd=round(cost, 6),
        calls=calls,
    )
    logger.info(
        "stage_complete",
        stage=Stage.EXTRACT.value,
        prompt_version=PROMPT_VERSION,
        chunks=calls,
        evidence_items=len(result.items),
        unlocatable=len(result.unlocatable),
        unlocatable_rate=round(result.unlocatable_rate, 3),
        cost_usd=metric.cost_usd,
        duration_ms=metric.duration_ms,
    )
    return result, metric
