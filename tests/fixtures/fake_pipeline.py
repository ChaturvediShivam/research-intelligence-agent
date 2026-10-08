"""Fakes for the two network edges, and nothing else.

The point of these is that the pipeline *really runs*. Fetching, text
extraction, chunking, embedding, indexing, hybrid retrieval, quote location,
offset arithmetic and citation verification are all the real
implementations — only the search provider and the LLM transport are stood
in for, because those are the two places that would otherwise cost money and
require a network.

The LLM stand-in is deliberately **content-aware** rather than canned: for an
extraction call it reads the chunk it was actually sent and quotes a real
sentence from it; for a synthesis call it reads the document blocks and emits
citations with real character offsets. That matters, because a canned reply
would make the verifier's job trivial and the test would prove nothing. Here
the verifier independently re-slices the stored source and either agrees or
does not — exactly as it would against the real API.

`FabricatingLLM` inverts that: it emits a plausible citation whose text is
not in the source, so the test can prove the orchestrator preserves M4's
rejection rather than quietly accepting it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.pipeline.extract import ChunkExtraction, EvidenceDraft
from app.schemas.evidence import EvidenceGrade
from app.schemas.research import (
    ResearchPlan,
    SourceType,
    SubQuestion,
)
from app.schemas.source import SourceCandidate

# --------------------------------------------------------------------------
# Source provider
# --------------------------------------------------------------------------


class FakeSourceProvider:
    """A `SourceProvider` that returns pre-set candidates.

    Returns real `SourceCandidate` objects, so url validation and domain
    extraction are the real ones.
    """

    def __init__(
        self,
        candidates: list[SourceCandidate] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._candidates = candidates or []
        self._error = error
        self.queries: list[str] = []

    async def search(
        self, query: str, *, max_results: int = 8, sub_question_id: str | None = None
    ) -> list[SourceCandidate]:
        self.queries.append(query)
        if self._error is not None:
            raise self._error
        return [
            c.model_copy(update={"sub_question_id": sub_question_id})
            for c in self._candidates[:max_results]
        ]


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------


def make_plan(sub_question_ids: tuple[str, ...] = ("SQ1", "SQ2")) -> ResearchPlan:
    """A valid plan, built through the real schema so validators apply."""
    questions = {
        "SQ1": "What were claims acceptance rates for home emergency cover?",
        "SQ2": "What was UK pet insurance gross written premium?",
        "SQ3": "Which insurers held the largest market shares?",
    }
    return ResearchPlan(
        restated_question=(
            "Claims acceptance rates and gross written premium for UK "
            "general insurance, most recent reporting period."
        ),
        sub_questions=[
            SubQuestion(
                id=sub_question_id,
                question=questions.get(sub_question_id, "A sub-question of substance?"),
                rationale="Needed to answer the overall question.",
                rank=rank,
                expected_source_types=[SourceType.REGULATORY_FILING],
                answerable_if="A regulator or trade body publishes the figure.",
            )
            for rank, sub_question_id in enumerate(sub_question_ids, start=1)
        ],
    )


# --------------------------------------------------------------------------
# LLM transport
# --------------------------------------------------------------------------

_FENCE = re.compile(
    r"<<<UNTRUSTED_SOURCE_CONTENT>>>\n.*?\n---\n(.*?)\n<<<END_UNTRUSTED_SOURCE_CONTENT>>>",
    re.DOTALL,
)


def _chunk_text_from(user_content: str) -> str:
    """Recover the chunk the extractor was actually given."""
    match = _FENCE.search(user_content)
    return match.group(1) if match else ""


def _first_sentence(text: str, *, min_words: int = 6) -> str | None:
    """Pick a real, quotable sentence out of a passage."""
    for candidate in re.split(r"(?<=[.!?])\s+|\n\n", text):
        cleaned = candidate.strip()
        if len(cleaned.split()) >= min_words:
            return cleaned
    return None


@dataclass
class FakeUsage:
    input_tokens: int = 1200
    output_tokens: int = 300
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class FakeCitationEntry:
    document_index: int
    cited_text: str
    start_char_index: int
    end_char_index: int
    type: str = "char_location"
    document_title: str = "doc"


@dataclass
class FakeTextBlock:
    text: str
    citations: list[Any] = field(default_factory=list)
    type: str = "text"


@dataclass
class FakeCreateResponse:
    content: list[Any]
    usage: FakeUsage = field(default_factory=FakeUsage)
    stop_reason: str = "end_turn"


@dataclass
class FakeParseResponse:
    parsed_output: Any
    usage: FakeUsage = field(default_factory=FakeUsage)
    stop_reason: str = "end_turn"


@dataclass
class FakeRunnerMessage:
    usage: FakeUsage = field(default_factory=FakeUsage)
    content: list[Any] = field(default_factory=list)


class _FakeToolRunner:
    """Drives the discovery tool the way the real runner would.

    Calls the stage's own `search_sources` tool once per sub-question, so the
    tool body — budget checks, de-duplication, failure handling — is the real
    one under test.
    """

    def __init__(self, tools: list[Any], sub_question_ids: list[str], turns: int) -> None:
        self._tools = tools
        self._sub_question_ids = sub_question_ids
        self._turns = turns

    def __aiter__(self) -> _FakeToolRunner:
        self._index = 0
        return self

    async def __anext__(self) -> FakeRunnerMessage:
        if self._index >= min(self._turns, len(self._sub_question_ids)):
            raise StopAsyncIteration
        sub_question_id = self._sub_question_ids[self._index]
        self._index += 1
        tool = self._tools[0]
        # `BetaAsyncFunctionTool.func` is the undecorated coroutine; `.call`
        # takes a raw input object. Invoking `.func` runs the real tool body —
        # budget check, de-duplication, failure handling — which is what is
        # under test here.
        await tool.func(query=f"query for {sub_question_id}", sub_question_id=sub_question_id)
        return FakeRunnerMessage()


class ScriptedLLM:
    """A content-aware stand-in for `anthropic.AsyncAnthropic`."""

    def __init__(
        self,
        *,
        plan: ResearchPlan | None = None,
        sub_question_ids: list[str] | None = None,
        discovery_turns: int = 2,
        extraction_enabled: bool = True,
        synthesis_enabled: bool = True,
        plan_error: Exception | None = None,
        extract_error: Exception | None = None,
        synthesis_error: Exception | None = None,
        fabricate_citations: bool = False,
    ) -> None:
        self._plan = plan or make_plan()
        self._sub_question_ids = sub_question_ids or [sq.id for sq in self._plan.ordered()]
        self._discovery_turns = discovery_turns
        self._extraction_enabled = extraction_enabled
        self._synthesis_enabled = synthesis_enabled
        self._plan_error = plan_error
        self._extract_error = extract_error
        self._synthesis_error = synthesis_error
        self._fabricate = fabricate_citations

        self.parse_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.messages = _Messages(self)
        self.beta = _Beta(self)

    async def close(self) -> None:
        return None

    # -- call handlers ----------------------------------------------------

    async def _parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(kwargs)
        output_format = kwargs.get("output_format")

        if output_format is ResearchPlan:
            if self._plan_error is not None:
                raise self._plan_error
            return FakeParseResponse(parsed_output=self._plan)

        if output_format is ChunkExtraction:
            if self._extract_error is not None:
                raise self._extract_error
            if not self._extraction_enabled:
                return FakeParseResponse(parsed_output=ChunkExtraction(items=[]))
            chunk_text = _chunk_text_from(kwargs["messages"][0]["content"])
            sentence = _first_sentence(chunk_text)
            if sentence is None:
                return FakeParseResponse(parsed_output=ChunkExtraction(items=[]))
            quote = (
                # A fabrication: plausible, and absent from the source.
                sentence.replace("61%", "94%")
                if self._fabricate and "61%" in sentence
                else sentence
            )
            return FakeParseResponse(
                parsed_output=ChunkExtraction(
                    items=[
                        EvidenceDraft(
                            statement="The passage reports a figure of record.",
                            verbatim_quote=quote,
                            grade=EvidenceGrade.BEHAVIOR,
                        )
                    ]
                )
            )

        raise AssertionError(f"unexpected output_format {output_format!r}")

    async def _create(self, **kwargs: Any) -> Any:
        """Synthesis: read the documents sent and cite them for real."""
        self.create_calls.append(kwargs)
        if self._synthesis_error is not None:
            raise self._synthesis_error
        if not self._synthesis_enabled:
            return FakeCreateResponse(content=[FakeTextBlock(text="No answer.")])

        documents = [
            block
            for block in kwargs["messages"][0]["content"]
            if isinstance(block, dict) and block.get("type") == "document"
        ]
        blocks: list[Any] = []
        for index, document in enumerate(documents):
            text = document["source"]["data"]
            sentence = _first_sentence(text)
            if sentence is None:
                continue
            start = text.index(sentence)
            if self._fabricate:
                # Real offsets, invented wording: exactly the shape M4 rejects.
                blocks.append(
                    FakeTextBlock(
                        text="The regulator confirmed a markedly higher figure.",
                        citations=[
                            FakeCitationEntry(
                                document_index=index,
                                cited_text="a markedly higher figure was confirmed",
                                start_char_index=start,
                                end_char_index=start + len(sentence),
                            )
                        ],
                    )
                )
            else:
                blocks.append(
                    FakeTextBlock(
                        text=sentence,
                        citations=[
                            FakeCitationEntry(
                                document_index=index,
                                cited_text=sentence,
                                start_char_index=start,
                                end_char_index=start + len(sentence),
                            )
                        ],
                    )
                )
        return FakeCreateResponse(content=blocks)

    def _tool_runner(self, **kwargs: Any) -> _FakeToolRunner:
        return _FakeToolRunner(list(kwargs["tools"]), self._sub_question_ids, self._discovery_turns)


class _Messages:
    def __init__(self, owner: ScriptedLLM) -> None:
        self._owner = owner

    async def parse(self, **kwargs: Any) -> Any:
        return await self._owner._parse(**kwargs)

    async def create(self, **kwargs: Any) -> Any:
        return await self._owner._create(**kwargs)


class _BetaMessages:
    def __init__(self, owner: ScriptedLLM) -> None:
        self._owner = owner

    def tool_runner(self, **kwargs: Any) -> _FakeToolRunner:
        return self._owner._tool_runner(**kwargs)


class _Beta:
    def __init__(self, owner: ScriptedLLM) -> None:
        self.messages = _BetaMessages(owner)
