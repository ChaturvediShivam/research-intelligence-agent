"""Observability schemas: token usage, cost, and per-stage timing.

Token categories are kept separate rather than summed. On Claude Opus 5.5 a
cache read costs $0.20/MTok against $4.00/MTok uncached — a 20x difference.
Collapsing them into one "input_tokens" number would hide the single most
useful cost signal in the system.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Stage(StrEnum):
    """The ten pipeline stages, named so metrics are attributable."""

    PLAN = "plan"
    DISCOVER = "discover"
    PROCESS = "process"
    RETRIEVE = "retrieve"
    EXTRACT = "extract"
    SYNTHESISE = "synthesise"
    VALIDATE = "validate"
    ASSESS = "assess"
    REPORT = "report"
    METRICS = "metrics"


class TokenUsage(BaseModel):
    """Token counts as reported by the API, by billing category."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
        )

    @property
    def total_input(self) -> int:
        """All input tokens regardless of billing category."""
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens

    @property
    def cache_hit_rate(self) -> float | None:
        """Share of input tokens served from cache, or None if no input yet.

        The number to watch: a silent cache invalidator shows up here as a
        sustained zero, with no error anywhere else.
        """
        if self.total_input == 0:
            return None
        return self.cache_read_input_tokens / self.total_input


class StageMetric(BaseModel):
    """One stage's measured cost and duration."""

    stage: Stage
    model: str | None = None
    duration_ms: int
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    # Number of model calls the stage made; EXTRACT makes one per chunk.
    calls: int = 0


class RunTrace(BaseModel):
    """Per-run measurement record. Written as the pipeline progresses."""

    run_id: str
    stages: list[StageMetric] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    @property
    def total_usage(self) -> TokenUsage:
        total = TokenUsage()
        for stage in self.stages:
            total = total + stage.usage
        return total

    @property
    def total_cost_usd(self) -> float:
        return round(sum(stage.cost_usd for stage in self.stages), 6)

    @property
    def total_duration_ms(self) -> int:
        return sum(stage.duration_ms for stage in self.stages)

    def cost_by_stage(self) -> dict[str, float]:
        """Cost attribution per stage — what makes routing decisions reviewable."""
        out: dict[str, float] = {}
        for stage in self.stages:
            out[stage.stage.value] = round(out.get(stage.stage.value, 0.0) + stage.cost_usd, 6)
        return out
