"""Source discovery over the server-side web search tool.

The important case here is the error shape: a failed server tool returns
HTTP 200 with a result block whose `content` is a single error *object*, not a
list. Indexing it as a list would raise an unrelated TypeError and hide the
real cause, so the branch is tested explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from app.core.config import Settings
from app.core.errors import UpstreamError
from app.schemas.source import SourceCandidate
from app.tools.search import (
    WEB_SEARCH_TOOL_TYPE,
    AnthropicSearchProvider,
    SourceProvider,
    _extract_candidates,
)


@dataclass
class FakeResult:
    url: str
    title: str = "A title"


@dataclass
class FakeSearchBlock:
    content: Any
    type: str = "web_search_tool_result"


@dataclass
class FakeTextBlock:
    text: str = "some prose"
    type: str = "text"


@dataclass
class FakeSearchError:
    error_code: str = "max_uses_exceeded"


@dataclass
class FakeResponse:
    content: list[Any] = field(default_factory=list)


class FakeMessages:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class FakeClient:
    def __init__(self, response: Any) -> None:
        self.messages = FakeMessages(response)


def _settings(**kw: object) -> Settings:
    return Settings(environment="test", _env_file=None, **kw)  # type: ignore[arg-type,call-arg]


class TestCandidateExtraction:
    def test_pulls_results_out_of_a_search_block(self) -> None:
        blocks = [
            FakeTextBlock(),
            FakeSearchBlock(
                content=[
                    FakeResult("https://gov.uk/stats", "Official statistics"),
                    FakeResult("https://fca.org.uk/data", "FCA data"),
                ]
            ),
        ]
        found = _extract_candidates(blocks, max_results=8, sub_question_id="SQ1")
        assert [str(c.url) for c in found] == [
            "https://gov.uk/stats",
            "https://fca.org.uk/data",
        ]
        assert all(c.sub_question_id == "SQ1" for c in found)

    def test_accepts_dict_shaped_results(self) -> None:
        """The SDK may return plain dicts depending on block type."""
        blocks = [FakeSearchBlock(content=[{"url": "https://gov.uk/a", "title": "A"}])]
        found = _extract_candidates(blocks, max_results=8, sub_question_id=None)
        assert len(found) == 1

    def test_deduplicates_by_url(self) -> None:
        blocks = [
            FakeSearchBlock(
                content=[
                    FakeResult("https://gov.uk/a"),
                    FakeResult("https://gov.uk/a"),
                    FakeResult("https://gov.uk/b"),
                ]
            )
        ]
        found = _extract_candidates(blocks, max_results=8, sub_question_id=None)
        assert len(found) == 2

    def test_respects_max_results(self) -> None:
        blocks = [FakeSearchBlock(content=[FakeResult(f"https://gov.uk/{i}") for i in range(20)])]
        assert len(_extract_candidates(blocks, max_results=3, sub_question_id=None)) == 3

    @pytest.mark.parametrize(
        "bad_url",
        ["file:///etc/passwd", "javascript:alert(1)", "not-a-url", "", "data:text/html,x"],
    )
    def test_malformed_result_is_skipped_not_fatal(self, bad_url: str) -> None:
        """One bad result from a search index must not fail discovery."""
        blocks = [FakeSearchBlock(content=[FakeResult(bad_url), FakeResult("https://gov.uk/good")])]
        found = _extract_candidates(blocks, max_results=8, sub_question_id=None)
        assert [str(c.url) for c in found] == ["https://gov.uk/good"]

    def test_no_search_block_yields_nothing(self) -> None:
        assert _extract_candidates([FakeTextBlock()], max_results=8, sub_question_id=None) == []

    def test_empty_content_list_yields_nothing(self) -> None:
        blocks = [FakeSearchBlock(content=[])]
        assert _extract_candidates(blocks, max_results=8, sub_question_id=None) == []


class TestServerToolErrorShape:
    def test_error_object_raises_rather_than_being_indexed(self) -> None:
        """Server-tool errors arrive as HTTP 200 with an object, not a list."""
        blocks = [FakeSearchBlock(content=FakeSearchError("max_uses_exceeded"))]
        with pytest.raises(UpstreamError, match="Web search failed"):
            _extract_candidates(blocks, max_results=8, sub_question_id=None)

    def test_error_code_is_carried_into_the_error_details(self) -> None:
        blocks = [FakeSearchBlock(content=FakeSearchError("query_too_long"))]
        with pytest.raises(UpstreamError) as info:
            _extract_candidates(blocks, max_results=8, sub_question_id=None)
        assert info.value.details["error_code"] == "query_too_long"

    def test_dict_shaped_error_also_handled(self) -> None:
        blocks = [FakeSearchBlock(content={"error_code": "unavailable"})]
        with pytest.raises(UpstreamError):
            _extract_candidates(blocks, max_results=8, sub_question_id=None)


class TestProviderRequestShape:
    async def test_declares_the_web_search_tool(self) -> None:
        client = FakeClient(
            FakeResponse(content=[FakeSearchBlock(content=[FakeResult("https://gov.uk/a")])])
        )
        provider = AnthropicSearchProvider(_settings(), client=client)
        await provider.search("UK pet insurance market size")

        call = client.messages.calls[0]
        tools = call["tools"]
        assert tools[0]["type"] == WEB_SEARCH_TOOL_TYPE
        assert tools[0]["name"] == "web_search"

    async def test_tool_choice_is_auto_not_forced(self) -> None:
        """Forced tool choice returns a 400 on Opus 5.5."""
        client = FakeClient(FakeResponse(content=[]))
        await AnthropicSearchProvider(_settings(), client=client).search("q")
        assert client.messages.calls[0]["tool_choice"] == {"type": "auto"}

    async def test_does_not_declare_code_execution(self) -> None:
        """web_search_20260209 runs code execution internally; a second
        declared execution environment confuses the model."""
        client = FakeClient(FakeResponse(content=[]))
        await AnthropicSearchProvider(_settings(), client=client).search("q")
        types = {t["type"] for t in client.messages.calls[0]["tools"]}
        assert not any("code_execution" in t for t in types)

    async def test_allowlist_is_passed_to_the_tool_as_an_enforced_control(self) -> None:
        client = FakeClient(FakeResponse(content=[]))
        provider = AnthropicSearchProvider(
            _settings(allowed_source_domains=("gov.uk", "fca.org.uk")), client=client
        )
        await provider.search("q")
        tool = client.messages.calls[0]["tools"][0]
        assert tool["allowed_domains"] == ["gov.uk", "fca.org.uk"]
        assert "blocked_domains" not in tool

    async def test_denylist_is_passed_when_no_allowlist(self) -> None:
        client = FakeClient(FakeResponse(content=[]))
        provider = AnthropicSearchProvider(
            _settings(blocked_source_domains=("spam.test",)), client=client
        )
        await provider.search("q")
        tool = client.messages.calls[0]["tools"][0]
        assert tool["blocked_domains"] == ["spam.test"]

    async def test_both_lists_configured_sends_only_the_allowlist(self) -> None:
        """The API rejects both lists on one tool."""
        client = FakeClient(FakeResponse(content=[]))
        provider = AnthropicSearchProvider(
            _settings(allowed_source_domains=("gov.uk",), blocked_source_domains=("spam.test",)),
            client=client,
        )
        await provider.search("q")
        tool = client.messages.calls[0]["tools"][0]
        assert "allowed_domains" in tool
        assert "blocked_domains" not in tool

    async def test_system_prompt_forbids_answering(self) -> None:
        client = FakeClient(FakeResponse(content=[]))
        await AnthropicSearchProvider(_settings(), client=client).search("q")
        system = client.messages.calls[0]["system"]
        assert "Do not answer the question yourself" in system


class TestProviderFailures:
    async def test_transport_failure_becomes_upstream_error(self) -> None:
        client = FakeClient(RuntimeError("socket died"))
        with pytest.raises(UpstreamError, match="discovery call failed"):
            await AnthropicSearchProvider(_settings(), client=client).search("q")

    async def test_search_tool_error_propagates_unwrapped(self) -> None:
        """An UpstreamError from the tool must not be re-wrapped and relabelled."""
        client = FakeClient(
            FakeResponse(content=[FakeSearchBlock(content=FakeSearchError("rate_limited"))])
        )
        with pytest.raises(UpstreamError, match="Web search failed"):
            await AnthropicSearchProvider(_settings(), client=client).search("q")


class TestProtocolConformance:
    def test_anthropic_provider_satisfies_the_protocol(self) -> None:
        """ADR-006's swap path is only real if the protocol is actually met."""
        provider = AnthropicSearchProvider(_settings(), client=FakeClient(None))
        assert isinstance(provider, SourceProvider)

    def test_candidate_domain_strips_www(self) -> None:
        c = SourceCandidate(url="https://www.gov.uk/a", title="t")
        assert c.domain == "gov.uk"
