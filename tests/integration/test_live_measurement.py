"""Live proof that cost and latency are measured, not fixture values (M9).

Deliberately the smallest call that settles the question. The measurement
path does not vary by stage: `LLMClient` reads `response.usage` and prices it
through `app.llm.pricing` identically wherever it is called from, so proving
it once on the cheapest model proves it everywhere.

The full-pipeline figures in `docs/cost-latency.md` come from the M5 live
end-to-end run, which is already committed. This test exists to prove the
numbers originate at the API rather than in a fixture — something no offline
test can establish, since an offline test's tokens are whatever the fake
returned.

One Haiku call. Marked `live`, so it is deselected by default.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.llm.client import LLMClient
from app.llm.pricing import cost_usd
from app.observability.trace import Measurement
from app.schemas.runs import TokenUsage


def _key_available() -> bool:
    try:
        Settings(environment="local").require_anthropic_key()
    except Exception:
        return False
    return True


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _key_available(), reason="ANTHROPIC_API_KEY not resolvable"),
]

ARTIFACT = Path("evals/results/live_measurement_m9.json")


def _write_artifact(payload: str) -> str:
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(payload + "\n")
    return ARTIFACT.read_text()


class Answer(BaseModel):
    """Trivial schema: the content is irrelevant, the usage is the point."""

    country: str = Field(max_length=100)


@pytest.mark.asyncio
async def test_tokens_and_latency_come_from_the_api() -> None:
    settings = Settings(environment="local")
    client = LLMClient(settings)
    try:
        result = await client.structured(
            model="claude-haiku-4-5",
            output_model=Answer,
            system="Answer with the country named. One word.",
            user_content="Which country does the Financial Conduct Authority regulate?",
            effort=None,
            max_tokens=64,
        )
    finally:
        await client.aclose()

    # Measured: the API reported these. A fixture cannot produce a plausible
    # input count for a prompt it never saw.
    assert result.usage.input_tokens > 0, "input tokens must come from the API"
    assert result.usage.output_tokens > 0
    assert result.duration_ms > 0, "latency is wall-clock measured"

    # Derived: recomputing from the measured usage and the price table must
    # reproduce the reported cost exactly. That is what "derived" means.
    recomputed = cost_usd(result.model, result.usage)
    assert result.cost_usd == recomputed
    assert result.cost_usd > 0

    # And the provenance labels the trace exporter would attach.
    assert Measurement.MEASURED == "measured"
    assert Measurement.DERIVED == "derived"

    payload = json.dumps(
        {
            "purpose": "M9 proof that tokens and latency originate at the API",
            "model": result.model,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "cache_read_input_tokens": result.usage.cache_read_input_tokens,
            "cache_creation_input_tokens": result.usage.cache_creation_input_tokens,
            "duration_ms": result.duration_ms,
            "cost_usd": result.cost_usd,
            "cost_provenance": "derived from measured tokens x app/llm/pricing.py",
            "latency_provenance": "measured (perf_counter around the call)",
            "tokens_provenance": "measured (response.usage)",
            "calls": 1,
        },
        indent=2,
        sort_keys=True,
    )
    written = await asyncio.to_thread(_write_artifact, payload)

    # The artifact must carry no secret. Checked against the resolved key
    # rather than os.environ, since the key may come from .env — and against
    # a sentinel that cannot occur, not a one-character default.
    assert settings.require_anthropic_key() not in written
    assert "sk-ant" not in written


def test_an_absent_measurement_is_not_reported_as_zero() -> None:
    """Offline companion: the distinction the exporter has to preserve."""
    empty = TokenUsage()
    assert empty.cache_hit_rate is None, "no input is unknown, not a rate of 0.0"
