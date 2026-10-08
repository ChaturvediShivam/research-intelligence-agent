"""Anthropic client wrapper.

Three responsibilities, deliberately narrow:

1. **Structured calls** — `structured()` constrains the response to a Pydantic
   model and returns a validated instance, so no call site parses JSON by hand.
2. **Usage capture** — every call returns its `TokenUsage` and computed cost,
   because a cost figure reconstructed later is an estimate, and this project
   does not report estimates.
3. **Error mapping** — provider exceptions become the application's typed
   errors, so pipeline code never imports `anthropic` to handle a failure.

This is not an abstraction layer over "LLM providers" — there is one provider
and adding a second is not a goal. It is the seam where usage accounting and
error translation happen exactly once.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Any, TypeVar

import anthropic
import structlog
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.errors import UpstreamError
from app.llm.pricing import cost_usd
from app.schemas.runs import TokenUsage

logger = structlog.get_logger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

# Retry only what is genuinely transient. A 400 or a schema failure will fail
# identically on retry and must surface immediately.
_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)


async def aclose_client(client: Any | None) -> None:
    """Close an async API client, whatever it calls its close method.

    `httpx.AsyncClient` exposes `aclose()`; `anthropic.AsyncAnthropic` exposes
    `close()`. A `getattr(client, "aclose", None)` guard silently did nothing
    for the latter, so the client was never closed and its transport was
    reaped by the garbage collector after the event loop had gone — surfacing
    as an intermittent "Event loop is closed". Both names are tried, and a
    client exposing neither is a programming error worth raising on.
    See docs/failure-analysis.md F-006.
    """
    if client is None:
        return
    for name in ("aclose", "close"):
        method = getattr(client, name, None)
        if method is None:
            continue
        result = method()
        if inspect.isawaitable(result):
            await result
        return
    raise TypeError(
        f"{type(client).__name__} exposes neither aclose() nor close(); its transport would leak."
    )


@dataclass(slots=True)
class LLMResult[T]:
    """A model response plus what it cost to produce."""

    value: T
    model: str
    usage: TokenUsage
    cost_usd: float
    duration_ms: int


def _usage_from_response(response: Any) -> TokenUsage:
    """Read usage off a response, tolerating absent cache fields.

    `cache_read_input_tokens` and `cache_creation_input_tokens` are absent (or
    None) when caching is not in play, so each is coerced rather than assumed.
    """
    usage = getattr(response, "usage", None)
    if usage is None:  # pragma: no cover - defensive
        return TokenUsage()
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


class LLMClient:
    """Thin wrapper around the Anthropic async client."""

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self._settings = settings
        # An injected client is how tests exercise this without a key or a
        # network call; `None` means construct a real one, which requires a key.
        self._client = client
        self._max_attempts = 3

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=self._settings.require_anthropic_key(),
                max_retries=0,  # retries are handled here, with logging
            )
        return self._client

    async def structured(
        self,
        *,
        model: str,
        output_model: type[ModelT],
        system: str,
        user_content: str,
        effort: str = "high",
        max_tokens: int = 8192,
        cache_system: bool = True,
    ) -> LLMResult[ModelT]:
        """Make a call whose response is constrained to `output_model`.

        `system` is sent as a cacheable block and `user_content` after it, so
        the stable prefix stays byte-identical across calls (ADR-008). Effort
        is always explicit: Claude Opus 5.5 defaults to `medium`, so omitting
        it would silently pick a level.
        """
        system_blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
        if cache_system:
            system_blocks[0]["cache_control"] = {"type": "ephemeral"}

        started = time.perf_counter()
        response = await self._call_with_retry(
            model=model,
            output_format=output_model,
            system=system_blocks,
            messages=[{"role": "user", "content": user_content}],
            output_config={"effort": effort},
            max_tokens=max_tokens,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            # The API guarantees schema-valid output under output_format, so
            # this means a refusal or a truncation — both worth naming.
            raise UpstreamError(
                "Model returned no parsed output.",
                details={
                    "model": model,
                    "stop_reason": getattr(response, "stop_reason", None),
                },
            )
        if not isinstance(parsed, output_model):
            try:
                parsed = output_model.model_validate(
                    parsed if isinstance(parsed, dict) else parsed.model_dump()
                )
            except (ValidationError, AttributeError) as exc:
                raise UpstreamError(
                    "Model output did not validate against the expected schema.",
                    details={"model": model, "expected": output_model.__name__},
                ) from exc

        usage = _usage_from_response(response)
        cost = cost_usd(model, usage)
        logger.info(
            "llm_call",
            model=model,
            output_model=output_model.__name__,
            effort=effort,
            duration_ms=duration_ms,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cost_usd=cost,
        )
        return LLMResult(
            value=parsed,
            model=model,
            usage=usage,
            cost_usd=cost,
            duration_ms=duration_ms,
        )

    async def _call_with_retry(self, **kwargs: Any) -> Any:
        """Call `messages.parse`, retrying only transient failures.

        Backoff is exponential and respects a `retry-after` when the provider
        supplies one. Non-retryable errors are mapped and raised at once.
        """
        last: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await self.client.messages.parse(**kwargs)
            except _RETRYABLE as exc:
                last = exc
                if attempt == self._max_attempts:
                    break
                delay = self._retry_delay(exc, attempt)
                logger.warning(
                    "llm_retry",
                    attempt=attempt,
                    max_attempts=self._max_attempts,
                    delay_seconds=delay,
                    error_type=type(exc).__name__,
                )
                await asyncio.sleep(delay)
            except anthropic.AuthenticationError as exc:
                raise UpstreamError(
                    "Anthropic rejected the API key.", details={"status": 401}
                ) from exc
            except anthropic.BadRequestError as exc:
                # A 400 is our bug — a malformed request or an unsupported
                # parameter combination. Surface it, never retry it.
                raise UpstreamError(
                    "Anthropic rejected the request as invalid.",
                    details={"status": 400, "provider_message": str(exc)[:500]},
                ) from exc
            except anthropic.APIStatusError as exc:
                raise UpstreamError(
                    "Anthropic returned an error status.",
                    details={"status": getattr(exc, "status_code", None)},
                ) from exc

        raise UpstreamError(
            f"Anthropic call failed after {self._max_attempts} attempts.",
            details={"error_type": type(last).__name__ if last else None},
        ) from last

    async def aclose(self) -> None:
        """Close the underlying transport.

        An AsyncAnthropic owns an httpx client; letting the garbage collector
        reap it after the event loop has closed raises "Event loop is closed"
        from the transport's destructor. Observed in M2 live verification.
        """
        await aclose_client(self._client)
        self._client = None

    @staticmethod
    def _retry_delay(exc: Exception, attempt: int) -> float:
        """Exponential backoff, overridden by a provider-supplied retry-after."""
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            raw = headers.get("retry-after")
            if raw:
                try:
                    return min(float(raw), 30.0)
                except (TypeError, ValueError):
                    pass
        return float(min(2**attempt, 8))
