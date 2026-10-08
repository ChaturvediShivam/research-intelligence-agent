"""Cost computation. Every published cost figure derives from this table."""

from __future__ import annotations

import pytest

from app.llm.pricing import PRICES, UnknownModelError, cost_usd, is_priced
from app.schemas.runs import TokenUsage


class TestCostComputation:
    def test_known_model_input_and_output(self) -> None:
        # 1M input + 1M output on Opus 5.5 = $4 + $20.
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost_usd("claude-opus-5-5", usage) == pytest.approx(24.0)

    def test_cache_read_is_twenty_times_cheaper_than_input(self) -> None:
        """The reason cache categories are tracked separately at all."""
        uncached = cost_usd("claude-opus-5-5", TokenUsage(input_tokens=1_000_000))
        cached = cost_usd("claude-opus-5-5", TokenUsage(cache_read_input_tokens=1_000_000))
        assert uncached == pytest.approx(4.0)
        assert cached == pytest.approx(0.20)
        assert uncached / cached == pytest.approx(20.0)

    def test_cache_write_costs_more_than_plain_input(self) -> None:
        write = cost_usd("claude-opus-5-5", TokenUsage(cache_creation_input_tokens=1_000_000))
        assert write == pytest.approx(5.0)
        assert write > 4.0

    def test_haiku_is_cheaper_than_opus_on_identical_usage(self) -> None:
        """Underpins the stage-routing decision in ADR-007."""
        usage = TokenUsage(input_tokens=500_000, output_tokens=100_000)
        assert cost_usd("claude-haiku-4-5", usage) < cost_usd("claude-opus-5-5", usage)

    def test_zero_usage_costs_nothing(self) -> None:
        assert cost_usd("claude-opus-5-5", TokenUsage()) == 0.0


class TestUnknownModel:
    def test_raises_rather_than_reporting_zero(self) -> None:
        """A silently free model would corrupt every cost figure published."""
        with pytest.raises(UnknownModelError, match="No price entry"):
            cost_usd("claude-does-not-exist", TokenUsage(input_tokens=100))

    def test_is_priced_reports_membership(self) -> None:
        assert is_priced("claude-opus-5-5")
        assert not is_priced("gpt-nope")


class TestTokenUsage:
    def test_addition_sums_every_category(self) -> None:
        a = TokenUsage(input_tokens=1, output_tokens=2, cache_read_input_tokens=3)
        b = TokenUsage(input_tokens=10, output_tokens=20, cache_creation_input_tokens=5)
        total = a + b
        assert (total.input_tokens, total.output_tokens) == (11, 22)
        assert total.cache_read_input_tokens == 3
        assert total.cache_creation_input_tokens == 5

    def test_cache_hit_rate_is_none_before_any_input(self) -> None:
        assert TokenUsage().cache_hit_rate is None

    def test_cache_hit_rate_computed_over_all_input_categories(self) -> None:
        usage = TokenUsage(input_tokens=250, cache_read_input_tokens=750)
        assert usage.total_input == 1000
        assert usage.cache_hit_rate == pytest.approx(0.75)

    def test_every_configured_model_has_a_price(self) -> None:
        """Guards against a default in config.py drifting from the price table."""
        from app.core.config import Settings

        s = Settings(_env_file=None)  # type: ignore[call-arg]
        for model in (s.planning_model, s.synthesis_model, s.extraction_model):
            assert model in PRICES, f"{model} is configured but unpriced"
