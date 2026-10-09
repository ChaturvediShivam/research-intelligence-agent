"""Stage 6 — which blocks of a cited response become claims.

The API returns a cited answer as a *sequence* of text blocks, split at
citation boundaries. Cited sentences, uncited connective prose, markdown
bullet labels and the model's own "the documents do not answer this" each
arrive as their own block.

Every offline fixture before this file emitted one cited block per document,
so the suite had never seen the shape production actually returns — which is
why turning every block into a claim passed 900+ tests and then reported 20
claims with 11 fake UNKNOWNs on the first real run (F-020). The blocks here
are modelled on that run's actual output.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from app.core.config import Settings
from app.llm.client import LLMClient
from app.pipeline.synthesize import (
    PROMPT_VERSION,
    run_synthesise_stage,
    synthesise_sub_question,
)
from app.schemas.evidence import Claim
from app.schemas.research import ResearchPlan, SourceType, SubQuestion
from app.schemas.source import FetchedSource
from tests.fixtures.fake_pipeline import (
    FakeCitationEntry,
    FakeCreateResponse,
    FakeTextBlock,
)

SOURCE_TEXT = (
    "Worldwide generative AI spending is expected to total $644 billion in "
    "2025, an increase of 76.4% from 2024. Spending will be driven largely "
    "by the integration of AI capabilities into hardware."
)
CITED_SENTENCE = "Worldwide generative AI spending is expected to total $644 billion in 2025"

# Verbatim shapes from the first production run, which the old parser turned
# into UNKNOWN claims with failure code `no_citation`.
UNCITED_PROSE = (
    "**The attached documents answer only a small part of this question.**",
    "- **Hardware share:**",
    "- **Drivers:**",
    "The document refers to Table 1 but does not include its contents.",
)


def _source(url: str = "https://www.hpcwire.com/report") -> FetchedSource:
    return FetchedSource(
        url=url,
        final_url=url,
        title="GenAI spending forecast",
        text=SOURCE_TEXT,
        content_hash=hashlib.sha256(SOURCE_TEXT.encode()).hexdigest(),
        status_code=200,
        byte_length=len(SOURCE_TEXT),
    )


def _cited_block(text: str = CITED_SENTENCE) -> FakeTextBlock:
    start = SOURCE_TEXT.index(CITED_SENTENCE)
    return FakeTextBlock(
        text=text,
        citations=[
            FakeCitationEntry(
                document_index=0,
                cited_text=CITED_SENTENCE,
                start_char_index=start,
                end_char_index=start + len(CITED_SENTENCE),
            )
        ],
    )


class BlockScriptedClient:
    """A raw-SDK stand-in returning exactly the blocks a test specifies."""

    def __init__(self, blocks: list[Any]) -> None:
        self._blocks = blocks
        self.calls: list[dict[str, Any]] = []
        self.messages = self._Messages(self)

    class _Messages:
        def __init__(self, owner: BlockScriptedClient) -> None:
            self._owner = owner

        async def create(self, **kwargs: Any) -> FakeCreateResponse:
            self._owner.calls.append(kwargs)
            return FakeCreateResponse(content=list(self._owner._blocks))


def _settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        environment="test", _env_file=None, anthropic_api_key="sk-ant-test"
    )


def _client(blocks: list[Any]) -> tuple[LLMClient, BlockScriptedClient]:
    raw = BlockScriptedClient(blocks)
    return LLMClient(_settings(), client=raw), raw


async def _synthesise(blocks: list[Any]) -> tuple[list[Claim], int]:
    client, _ = _client(blocks)
    claims, declined, _usage, _cost = await synthesise_sub_question(
        "SQ1", "How large is the market?", [_source()], client=client, settings=_settings()
    )
    return claims, declined


class TestOnlyCitedBlocksBecomeClaims:
    async def test_a_cited_block_becomes_a_claim(self) -> None:
        claims, declined = await _synthesise([_cited_block()])
        assert len(claims) == 1
        assert claims[0].citations
        assert declined == 0

    @pytest.mark.parametrize("prose", UNCITED_PROSE)
    async def test_an_uncited_block_does_not_become_a_claim(self, prose: str) -> None:
        """Each of these was an UNKNOWN claim in production. None is a claim."""
        claims, declined = await _synthesise([FakeTextBlock(text=prose)])
        assert claims == []
        assert declined == 1

    async def test_the_production_block_shape_yields_only_real_claims(self) -> None:
        """Interleaved, as the API actually returns it."""
        blocks = [
            FakeTextBlock(text=UNCITED_PROSE[0]),
            _cited_block(),
            FakeTextBlock(text=UNCITED_PROSE[1]),
            FakeTextBlock(text=UNCITED_PROSE[2]),
            _cited_block("Spending will be driven largely by hardware integration."),
            FakeTextBlock(text=UNCITED_PROSE[3]),
        ]
        claims, declined = await _synthesise(blocks)
        assert len(claims) == 2, [c.text for c in claims]
        assert declined == 4
        assert all(c.citations for c in claims)
        # No claim text is one of the fragments.
        assert not {c.text for c in claims} & set(UNCITED_PROSE)

    async def test_claim_ids_are_contiguous(self) -> None:
        blocks = [FakeTextBlock(text=UNCITED_PROSE[0]), _cited_block(), _cited_block("Second.")]
        claims, _ = await _synthesise(blocks)
        assert [c.claim_id for c in claims] == ["SQ1_C1", "SQ1_C2"]

    async def test_a_short_fragment_is_still_skipped(self) -> None:
        """The pre-existing length guard is unchanged."""
        claims, declined = await _synthesise([FakeTextBlock(text="ok")])
        assert claims == []
        assert declined == 0

    async def test_synthesis_still_cannot_mark_its_own_claim_verified(self) -> None:
        """The guarantee the filter must not weaken."""
        claims, _ = await _synthesise([_cited_block()])
        claim = claims[0]
        assert claim.status.value == "unknown"
        assert claim.confidence.value == "unknown"
        assert claim.verified_citations == 0


class TestDeclinedSubQuestionsBecomeGaps:
    def _plan(self) -> ResearchPlan:
        return ResearchPlan(
            restated_question="How large is the generative AI market in 2025?",
            sub_questions=[
                SubQuestion(
                    id="SQ1",
                    question="How large is the generative AI market?",
                    rationale="Needed for the overall answer.",
                    rank=1,
                    expected_source_types=[SourceType.INDUSTRY_REPORT],
                    answerable_if="A credible source sizes the market.",
                )
            ],
        )

    async def test_uncited_only_output_is_unanswered_not_an_unknown_claim(self) -> None:
        """The routing requirement: a gap, never a manufactured claim."""
        client, _ = _client([FakeTextBlock(text=UNCITED_PROSE[0])])
        result, _metric = await run_synthesise_stage(
            self._plan(), {"SQ1": [_source()]}, client=client, settings=_settings()
        )
        assert result.claims == []
        assert result.unanswered == ["SQ1"]
        assert result.declined == {"SQ1": 1}

    async def test_a_cited_sub_question_is_not_marked_unanswered(self) -> None:
        client, _ = _client([FakeTextBlock(text=UNCITED_PROSE[0]), _cited_block()])
        result, _metric = await run_synthesise_stage(
            self._plan(), {"SQ1": [_source()]}, client=client, settings=_settings()
        )
        assert len(result.claims) == 1
        assert result.unanswered == []

    async def test_claims_equals_cited_claims(self) -> None:
        """The invariant the filter establishes, stated through the existing property."""
        client, _ = _client([FakeTextBlock(text=UNCITED_PROSE[1]), _cited_block()])
        result, _metric = await run_synthesise_stage(
            self._plan(), {"SQ1": [_source()]}, client=client, settings=_settings()
        )
        assert result.claims == result.cited_claims

    async def test_a_sub_question_with_no_sources_is_still_unanswered(self) -> None:
        """Pre-existing behaviour, unchanged by the filter."""
        client, raw = _client([_cited_block()])
        result, _metric = await run_synthesise_stage(
            self._plan(), {}, client=client, settings=_settings()
        )
        assert result.unanswered == ["SQ1"]
        assert raw.calls == [], "no model call should be made without sources"


class TestCorroborationInstruction:
    async def test_the_request_asks_for_cross_source_citation(self) -> None:
        client, raw = _client([_cited_block()])
        await synthesise_sub_question(
            "SQ1", "How large is the market?", [_source()], client=client, settings=_settings()
        )
        sent = raw.calls[0]["messages"][0]["content"][-1]["text"]
        assert "cite each of them" in sent
        assert "same publisher are one source" in sent
        assert "must not present it as though it does" in sent
        assert "rests on a single source" in sent
        assert "Never cite a document for a point it does not actually make." in sent

    async def test_the_grounded_answer_guarantee_is_still_instructed(self) -> None:
        """The corroboration text must not have displaced the original rules."""
        client, raw = _client([_cited_block()])
        await synthesise_sub_question(
            "SQ1", "How large is the market?", [_source()], client=client, settings=_settings()
        )
        sent = raw.calls[0]["messages"][0]["content"][-1]["text"]
        assert "Answer only from the attached documents" in sent
        assert "do not answer the question, say exactly that and cite nothing" in sent

    def test_the_prompt_version_records_the_change(self) -> None:
        assert PROMPT_VERSION == "synthesize.v2"
