"""Stage 1 (PLAN) behaviour."""

from __future__ import annotations

import anthropic
import httpx
import pytest

from app.core.config import Settings
from app.core.errors import PipelineStageError
from app.llm.client import LLMClient
from app.pipeline.plan import PROMPT_VERSION, run_plan_stage
from app.schemas.research import ResearchPlan, ResearchRequest
from app.schemas.runs import Stage
from tests.fixtures.fake_anthropic import FakeAnthropic, FakeResponse, FakeUsage


def _settings(**kw: object) -> Settings:
    return Settings(environment="test", _env_file=None, **kw)  # type: ignore[arg-type,call-arg]


class TestPlanStage:
    async def test_returns_plan_and_measured_metric(self, sample_plan: ResearchPlan) -> None:
        settings = _settings()
        fake = FakeAnthropic(
            [
                FakeResponse(
                    parsed_output=sample_plan,
                    usage=FakeUsage(input_tokens=2000, output_tokens=800),
                )
            ]
        )
        plan, metric = await run_plan_stage(
            ResearchRequest(question="How large is the UK pet insurance market?"),
            client=LLMClient(settings, client=fake),
            settings=settings,
        )
        assert plan.sub_questions[0].id == "SQ1"
        assert metric.stage is Stage.PLAN
        assert metric.model == settings.planning_model
        assert metric.calls == 1
        assert metric.usage.input_tokens == 2000
        assert metric.cost_usd > 0
        assert metric.duration_ms >= 0

    async def test_uses_the_configured_planning_model_at_high_effort(
        self, sample_plan: ResearchPlan
    ) -> None:
        """ADR-007: planning gets the capable model, explicitly at high effort."""
        settings = _settings()
        fake = FakeAnthropic([FakeResponse(parsed_output=sample_plan)])
        await run_plan_stage(
            ResearchRequest(question="How large is the UK pet insurance market?"),
            client=LLMClient(settings, client=fake),
            settings=settings,
        )
        call = fake.messages.calls[0]
        assert call["model"] == "claude-opus-5-5"
        assert call["output_config"] == {"effort": "high"}
        assert call["output_format"] is ResearchPlan

    async def test_sends_the_versioned_prompt_as_the_cached_prefix(
        self, sample_plan: ResearchPlan
    ) -> None:
        settings = _settings()
        fake = FakeAnthropic([FakeResponse(parsed_output=sample_plan)])
        await run_plan_stage(
            ResearchRequest(question="How large is the UK pet insurance market?"),
            client=LLMClient(settings, client=fake),
            settings=settings,
        )
        system = fake.messages.calls[0]["system"][0]
        assert "research planner" in system["text"].lower()
        assert system["cache_control"] == {"type": "ephemeral"}
        assert PROMPT_VERSION == "plan.v1"

    async def test_caller_context_reaches_the_volatile_half(
        self, sample_plan: ResearchPlan
    ) -> None:
        settings = _settings()
        fake = FakeAnthropic([FakeResponse(parsed_output=sample_plan)])
        await run_plan_stage(
            ResearchRequest(
                question="How large is the UK pet insurance market?",
                context="Only consider retail policies.",
            ),
            client=LLMClient(settings, client=fake),
            settings=settings,
        )
        content = fake.messages.calls[0]["messages"][0]["content"]
        assert "Only consider retail policies." in content
        # It must not be welded into the cached system prefix.
        assert "Only consider retail policies." not in fake.messages.calls[0]["system"][0]["text"]

    async def test_failure_is_wrapped_with_the_stage_name(self) -> None:
        settings = _settings()
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        exc = anthropic.BadRequestError(
            "nope", response=httpx.Response(400, request=request), body=None
        )
        with pytest.raises(PipelineStageError) as info:
            await run_plan_stage(
                ResearchRequest(question="How large is the UK pet insurance market?"),
                client=LLMClient(settings, client=FakeAnthropic([exc])),
                settings=settings,
            )
        assert info.value.stage == Stage.PLAN.value
        assert info.value.details["stage"] == "plan"
        assert info.value.cause is not None
