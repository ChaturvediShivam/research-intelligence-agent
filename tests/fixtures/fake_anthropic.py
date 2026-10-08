"""A fake Anthropic async client for tests.

Mirrors only the surface `LLMClient` actually uses: `messages.parse(**kwargs)`
returning an object with `parsed_output` and `usage`. Keeping the fake this
narrow means a change in the real surface shows up as a failure here rather
than being papered over by an over-permissive mock.

No test makes a billable call; `-m live` tests use the real client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeUsage:
    input_tokens: int = 1000
    output_tokens: int = 500
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class FakeResponse:
    parsed_output: Any
    usage: FakeUsage = field(default_factory=FakeUsage)
    stop_reason: str = "end_turn"


class FakeMessages:
    """Records calls and replays a scripted sequence of outcomes."""

    def __init__(self, outcomes: list[Any]) -> None:
        # Each outcome is either a FakeResponse to return or an Exception to raise.
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._outcomes:
            raise AssertionError("FakeMessages ran out of scripted outcomes")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeAnthropic:
    """Stands in for `anthropic.AsyncAnthropic`."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.messages = FakeMessages(outcomes)
