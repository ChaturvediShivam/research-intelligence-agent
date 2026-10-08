"""LLM client behaviour: usage capture, retry policy, error mapping.

No test here touches the network. The fake asserts the request shape as well
as the response handling, because a silently wrong request shape is the
failure mode that only shows up on a live call.
"""

from __future__ import annotations

import anthropic
import httpx
import pytest
from pydantic import BaseModel

from app.core.config import Settings
from app.core.errors import UpstreamError
from app.llm.client import LLMClient
from tests.fixtures.fake_anthropic import FakeAnthropic, FakeResponse, FakeUsage


class Answer(BaseModel):
    value: str


def _settings() -> Settings:
    return Settings(environment="test", _env_file=None)  # type: ignore[call-arg]


def _client(outcomes: list[object]) -> tuple[LLMClient, FakeAnthropic]:
    fake = FakeAnthropic(outcomes)
    return LLMClient(_settings(), client=fake), fake


def _rate_limit() -> anthropic.RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request, headers={"retry-after": "0"})
    return anthropic.RateLimitError("slow down", response=response, body=None)


def _bad_request() -> anthropic.BadRequestError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(400, request=request)
    return anthropic.BadRequestError("bad params", response=response, body=None)


class TestRequestShape:
    async def test_effort_is_always_explicit(self) -> None:
        """Opus 5.5 defaults to `medium`; omitting effort would pick silently."""
        client, fake = _client([FakeResponse(parsed_output=Answer(value="x"))])
        await client.structured(
            model="claude-opus-5-5",
            output_model=Answer,
            system="sys",
            user_content="user",
            effort="high",
        )
        assert fake.messages.calls[0]["output_config"] == {"effort": "high"}

    async def test_system_is_sent_as_a_cacheable_block(self) -> None:
        client, fake = _client([FakeResponse(parsed_output=Answer(value="x"))])
        await client.structured(
            model="claude-opus-5-5",
            output_model=Answer,
            system="stable prefix",
            user_content="volatile question",
        )
        call = fake.messages.calls[0]
        assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert call["system"][0]["text"] == "stable prefix"
        # Volatile content goes after the cached prefix, never inside it.
        assert call["messages"][0]["content"] == "volatile question"

    async def test_caching_can_be_disabled(self) -> None:
        client, fake = _client([FakeResponse(parsed_output=Answer(value="x"))])
        await client.structured(
            model="claude-opus-5-5",
            output_model=Answer,
            system="sys",
            user_content="u",
            cache_system=False,
        )
        assert "cache_control" not in fake.messages.calls[0]["system"][0]

    async def test_output_format_is_the_pydantic_type(self) -> None:
        client, fake = _client([FakeResponse(parsed_output=Answer(value="x"))])
        await client.structured(
            model="claude-opus-5-5", output_model=Answer, system="s", user_content="u"
        )
        assert fake.messages.calls[0]["output_format"] is Answer


class TestUsageAndCost:
    async def test_usage_and_cost_are_captured_from_the_response(self) -> None:
        client, _ = _client(
            [
                FakeResponse(
                    parsed_output=Answer(value="ok"),
                    usage=FakeUsage(
                        input_tokens=1_000_000,
                        output_tokens=0,
                        cache_read_input_tokens=0,
                    ),
                )
            ]
        )
        result = await client.structured(
            model="claude-opus-5-5", output_model=Answer, system="s", user_content="u"
        )
        assert result.value.value == "ok"
        assert result.usage.input_tokens == 1_000_000
        assert result.cost_usd == pytest.approx(4.0)
        assert result.duration_ms >= 0

    async def test_cache_tokens_are_recorded_separately(self) -> None:
        client, _ = _client(
            [
                FakeResponse(
                    parsed_output=Answer(value="ok"),
                    usage=FakeUsage(
                        input_tokens=0,
                        output_tokens=0,
                        cache_read_input_tokens=1_000_000,
                    ),
                )
            ]
        )
        result = await client.structured(
            model="claude-opus-5-5", output_model=Answer, system="s", user_content="u"
        )
        assert result.usage.cache_read_input_tokens == 1_000_000
        assert result.cost_usd == pytest.approx(0.20)


class TestRetryPolicy:
    async def test_retries_rate_limit_then_succeeds(self) -> None:
        client, fake = _client(
            [_rate_limit(), FakeResponse(parsed_output=Answer(value="recovered"))]
        )
        result = await client.structured(
            model="claude-opus-5-5", output_model=Answer, system="s", user_content="u"
        )
        assert result.value.value == "recovered"
        assert len(fake.messages.calls) == 2

    async def test_gives_up_after_max_attempts(self) -> None:
        client, fake = _client([_rate_limit(), _rate_limit(), _rate_limit()])
        with pytest.raises(UpstreamError, match="after 3 attempts"):
            await client.structured(
                model="claude-opus-5-5",
                output_model=Answer,
                system="s",
                user_content="u",
            )
        assert len(fake.messages.calls) == 3

    async def test_bad_request_is_not_retried(self) -> None:
        """A 400 is our bug; retrying it wastes money and hides the cause."""
        client, fake = _client([_bad_request()])
        with pytest.raises(UpstreamError, match="invalid"):
            await client.structured(
                model="claude-opus-5-5",
                output_model=Answer,
                system="s",
                user_content="u",
            )
        assert len(fake.messages.calls) == 1


class TestErrorMapping:
    async def test_auth_error_becomes_upstream_error(self) -> None:
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        exc = anthropic.AuthenticationError(
            "bad key", response=httpx.Response(401, request=request), body=None
        )
        client, _ = _client([exc])
        with pytest.raises(UpstreamError, match="rejected the API key"):
            await client.structured(
                model="claude-opus-5-5",
                output_model=Answer,
                system="s",
                user_content="u",
            )

    async def test_missing_parsed_output_is_reported_not_silently_empty(self) -> None:
        client, _ = _client([FakeResponse(parsed_output=None, stop_reason="refusal")])
        with pytest.raises(UpstreamError, match="no parsed output"):
            await client.structured(
                model="claude-opus-5-5",
                output_model=Answer,
                system="s",
                user_content="u",
            )

    async def test_provider_detail_is_bounded_in_error_details(self) -> None:
        """Provider messages are truncated so a huge body cannot flood logs."""
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        exc = anthropic.BadRequestError(
            "x" * 5000, response=httpx.Response(400, request=request), body=None
        )
        client, _ = _client([exc])
        with pytest.raises(UpstreamError) as info:
            await client.structured(
                model="claude-opus-5-5",
                output_model=Answer,
                system="s",
                user_content="u",
            )
        assert len(info.value.details["provider_message"]) <= 500


class TestKeyRequirement:
    async def test_real_client_construction_requires_a_key(self) -> None:
        """Without an injected client, the key is required at point of use."""
        from app.core.config import MissingConfigurationError

        client = LLMClient(_settings())  # no injected client, no key configured
        with pytest.raises(MissingConfigurationError, match="ANTHROPIC_API_KEY"):
            await client.structured(
                model="claude-opus-5-5",
                output_model=Answer,
                system="s",
                user_content="u",
            )
