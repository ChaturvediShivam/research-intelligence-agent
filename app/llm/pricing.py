"""Model price table and cost computation.

Prices are USD per million tokens, current as of 2026-10-08. They are data,
not estimates: every cost figure this project reports is derived from a real
`usage` object multiplied by this table. Nothing is approximated.

Cache economics are the reason the categories are tracked separately:
a cache read on Opus 5.5 is 20x cheaper than uncached input.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.schemas.runs import TokenUsage


class ModelPrice(BaseModel):
    """Per-million-token prices for one model."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_per_mtok: float


# Source: Anthropic published pricing. Cache writes bill at ~1.25x input.
PRICES: dict[str, ModelPrice] = {
    "claude-opus-5-5": ModelPrice(
        input_per_mtok=4.00,
        output_per_mtok=20.00,
        cache_read_per_mtok=0.20,
        cache_write_per_mtok=5.00,
    ),
    "claude-haiku-4-5": ModelPrice(
        input_per_mtok=1.00,
        output_per_mtok=5.00,
        cache_read_per_mtok=0.10,
        cache_write_per_mtok=1.25,
    ),
}

_MILLION = 1_000_000


class UnknownModelError(KeyError):
    """Raised when a cost is requested for a model with no price entry.

    Deliberately an error rather than a zero: a silently free model would
    under-report every cost figure in the project, which is worse than a crash.
    """


def cost_usd(model: str, usage: TokenUsage) -> float:
    """Compute the USD cost of one call from its reported usage."""
    try:
        price = PRICES[model]
    except KeyError as exc:
        raise UnknownModelError(
            f"No price entry for model {model!r}. Add it to app/llm/pricing.py "
            "rather than letting its cost be reported as zero."
        ) from exc

    total = (
        usage.input_tokens * price.input_per_mtok
        + usage.output_tokens * price.output_per_mtok
        + usage.cache_read_input_tokens * price.cache_read_per_mtok
        + usage.cache_creation_input_tokens * price.cache_write_per_mtok
    ) / _MILLION
    return round(total, 6)


def is_priced(model: str) -> bool:
    """Whether a price entry exists, for pre-flight configuration checks."""
    return model in PRICES
