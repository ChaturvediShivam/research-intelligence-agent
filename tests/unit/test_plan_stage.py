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
        assert PROMPT_VERSION == "plan.v2"

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


class TestPlannerIsBudgetAware:
    """The planner produced 6 sub-questions against a 4-source budget.

    Six sub-questions cannot be answered, let alone corroborated, from four
    sources — the arithmetic forecloses the run before discovery starts. The
    budget is stated to the planner so the decomposition can respect it.
    """

    @pytest.mark.parametrize("budget", [2, 4, 12])
    async def test_the_runtime_budget_reaches_the_planner(
        self, sample_plan: ResearchPlan, budget: int
    ) -> None:
        """Read from settings, never hard-coded."""
        settings = _settings(max_sources_per_run=budget)
        fake = FakeAnthropic([FakeResponse(parsed_output=sample_plan)])
        await run_plan_stage(
            ResearchRequest(question="How large is the UK pet insurance market?"),
            client=LLMClient(settings, client=fake),
            settings=settings,
        )
        user_content = fake.messages.calls[0]["messages"][0]["content"]
        assert f"Source budget for this run: {budget} source(s)" in user_content
        assert "shared across all sub-questions" in user_content

    async def test_a_per_request_override_is_what_the_planner_sees(
        self, sample_plan: ResearchPlan
    ) -> None:
        """The HTTP and MCP layers override the budget per request.

        They do it by copying settings, so the planner must read the copy —
        otherwise a caller asking for 4 sources is planned against 12.
        """
        settings = _settings(max_sources_per_run=12).model_copy(update={"max_sources_per_run": 4})
        fake = FakeAnthropic([FakeResponse(parsed_output=sample_plan)])
        await run_plan_stage(
            ResearchRequest(question="How large is the UK pet insurance market?", max_sources=4),
            client=LLMClient(settings, client=fake),
            settings=settings,
        )
        content = fake.messages.calls[0]["messages"][0]["content"]
        assert "Source budget for this run: 4 source(s)" in content

    async def test_the_budget_is_not_in_the_cached_prefix(self, sample_plan: ResearchPlan) -> None:
        """A per-request value in the system prompt would break the cache.

        `max_sources` varies per request, so putting it in the cached prefix
        would invalidate the prompt cache on every differing run (ADR-008,
        context assembly rule 1).
        """
        settings = _settings(max_sources_per_run=4)
        fake = FakeAnthropic([FakeResponse(parsed_output=sample_plan)])
        await run_plan_stage(
            ResearchRequest(question="How large is the UK pet insurance market?"),
            client=LLMClient(settings, client=fake),
            settings=settings,
        )
        system = fake.messages.calls[0]["system"][0]["text"]
        assert "Source budget for this run" not in system
        assert "4 source(s)" not in system

    async def test_an_over_budget_plan_is_flagged_not_failed(
        self, sample_plan: ResearchPlan
    ) -> None:
        """A prompt rule is a request, not a guarantee.

        Failing the run would trade a usable result for a quality preference,
        so the mismatch is logged and the extra sub-questions are reported as
        gaps by stage 8 exactly as before.
        """
        settings = _settings(max_sources_per_run=1)
        assert len(sample_plan.sub_questions) > 1
        fake = FakeAnthropic([FakeResponse(parsed_output=sample_plan)])
        plan, metric = await run_plan_stage(
            ResearchRequest(question="How large is the UK pet insurance market?"),
            client=LLMClient(settings, client=fake),
            settings=settings,
        )
        assert plan.sub_questions == sample_plan.sub_questions
        assert metric.calls == 1

    def test_the_prompt_states_the_budget_rule(self) -> None:
        from app.llm.context import load_prompt

        text = load_prompt(PROMPT_VERSION)
        assert "never produce more sub-questions than the source budget" in text.lower()
        assert "source budget" in text.lower()
        # The old unconditional range must now defer to the budget.
        assert "overrides this range" in text
