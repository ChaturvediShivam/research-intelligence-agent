"""Source discovery.

`SourceProvider` is the seam ADR-006 promised: discovery goes through
Anthropic's server-side `web_search_20260209` tool today, and swapping in a
dedicated search API later is one implementation of this protocol rather than
a change to the pipeline.

Two API details that shape this code:

- **Forced tool choice returns a 400 on Opus 5.5.** `tool_choice: {"type":
  "any"}` is not available, so the search is steered with `auto` plus an
  explicit instruction naming the tool.
- **Server-tool errors do not raise.** A failed search comes back as HTTP 200
  with a `web_search_tool_result` block whose `content` is a single error
  object rather than a list. Branching on that shape is mandatory; indexing it
  as a list is the bug this would otherwise ship with.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import ValidationError

from app.core.config import Settings
from app.core.errors import UpstreamError
from app.llm.client import aclose_client
from app.schemas.source import SourceCandidate

logger = structlog.get_logger(__name__)

WEB_SEARCH_TOOL_TYPE = "web_search_20260209"

_SEARCH_SYSTEM = (
    "You are a source-discovery tool. Use the web_search tool to find "
    "primary and authoritative sources that would answer the user's research "
    "question. Prefer regulatory filings, official statistics, and primary "
    "company documents over commentary. Do not answer the question yourself "
    "and do not summarise what you find — searching is the whole task."
)


@runtime_checkable
class SourceProvider(Protocol):
    """Discovers candidate sources for a query."""

    async def search(
        self, query: str, *, max_results: int = 8, sub_question_id: str | None = None
    ) -> list[SourceCandidate]: ...


def _extract_candidates(
    content: list[Any], *, max_results: int, sub_question_id: str | None
) -> list[SourceCandidate]:
    """Pull search results out of a response's content blocks.

    Untrusted throughout: urls, titles and snippets come from the open web.
    They are validated into `SourceCandidate` (which rejects a non-HTTP url)
    and never interpreted as instruction.
    """
    candidates: list[SourceCandidate] = []
    seen: set[str] = set()

    for block in content:
        if getattr(block, "type", None) != "web_search_tool_result":
            continue

        results = getattr(block, "content", None)

        # The error shape: an object, not a list. Checked before indexing.
        if not isinstance(results, list):
            error_code = getattr(results, "error_code", None) or (
                results.get("error_code") if isinstance(results, dict) else None
            )
            logger.warning("web_search_error", error_code=error_code)
            raise UpstreamError("Web search failed.", details={"error_code": str(error_code)[:100]})

        for result in results:
            url = getattr(result, "url", None) or (
                result.get("url") if isinstance(result, dict) else None
            )
            if not url or url in seen:
                continue
            title = getattr(result, "title", None) or (
                result.get("title") if isinstance(result, dict) else ""
            )
            try:
                candidate = SourceCandidate(
                    url=url,
                    title=(title or "")[:500],
                    sub_question_id=sub_question_id,
                )
            except ValidationError:
                # A malformed or non-http url from a search index is skipped,
                # not fatal: one bad result should not fail the discovery step.
                logger.info("search_result_rejected", url=str(url)[:200])
                continue
            seen.add(url)
            candidates.append(candidate)
            if len(candidates) >= max_results:
                return candidates

    return candidates


class AnthropicSearchProvider:
    """Discovery via Anthropic's server-side web search tool."""

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self._settings.require_anthropic_key())
        return self._client

    def _tool_definition(self, max_uses: int) -> dict[str, Any]:
        """Build the server-tool definition, applying domain policy.

        `allowed_domains` and `blocked_domains` are an enforced control here,
        not a prompt request — which is the security point made in ADR-006.
        The API rejects both lists on one tool, so allow wins when both are
        configured.
        """
        tool: dict[str, Any] = {
            "type": WEB_SEARCH_TOOL_TYPE,
            "name": "web_search",
            "max_uses": max_uses,
        }
        if self._settings.allowed_source_domains:
            tool["allowed_domains"] = list(self._settings.allowed_source_domains)
        elif self._settings.blocked_source_domains:
            tool["blocked_domains"] = list(self._settings.blocked_source_domains)
        return tool

    async def aclose(self) -> None:
        """Close the underlying transport. See LLMClient.aclose."""
        await aclose_client(self._client)
        self._client = None

    async def search(
        self, query: str, *, max_results: int = 8, sub_question_id: str | None = None
    ) -> list[SourceCandidate]:
        """Discover candidate sources for one query."""
        try:
            response = await self.client.messages.create(
                model=self._settings.planning_model,
                max_tokens=4096,
                system=_SEARCH_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Use the web_search tool to find sources that would answer: {query}"
                        ),
                    }
                ],
                tools=[self._tool_definition(max_uses=3)],
                # Forced tool choice is a 400 on this model; `auto` plus the
                # instruction above is the supported way to steer it.
                tool_choice={"type": "auto"},
            )
        except UpstreamError:
            raise
        except Exception as exc:
            raise UpstreamError(
                "Source discovery call failed.",
                details={"error_type": type(exc).__name__},
            ) from exc

        candidates = _extract_candidates(
            list(getattr(response, "content", []) or []),
            max_results=max_results,
            sub_question_id=sub_question_id,
        )
        logger.info(
            "sources_discovered",
            query=query[:200],
            found=len(candidates),
            sub_question_id=sub_question_id,
        )
        return candidates
