"""Shared fixtures.

Tests never read the developer's `.env`: every fixture builds `Settings`
explicitly, so a locally exported ANTHROPIC_API_KEY cannot change an outcome
and no test can make a billable call. The database is per-test and temporary.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.llm.client import LLMClient
from app.main import create_app
from app.schemas.research import ResearchPlan, SourceType, SubQuestion
from tests.fixtures.fake_anthropic import FakeAnthropic, FakeResponse


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Deterministic settings: no credentials, temporary database."""
    return Settings(
        environment="test",
        log_level="DEBUG",
        anthropic_api_key=None,
        postgres_dsn=None,
        vector_backend="sqlite",
        database_path=tmp_path / "test_runs.db",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def sample_plan() -> ResearchPlan:
    """A valid plan, used wherever a stage output is needed without a model."""
    return ResearchPlan(
        restated_question=("Gross written premium of the UK pet insurance market, 2024-2026."),
        sub_questions=[
            SubQuestion(
                id="SQ1",
                question="What was UK pet insurance GWP in each year 2024-2026?",
                rationale="The headline figure the question asks for.",
                rank=1,
                expected_source_types=[SourceType.OFFICIAL_STATISTICS],
                answerable_if="A regulator or trade body publishes annual GWP.",
            ),
            SubQuestion(
                id="SQ2",
                question="Which insurers hold the largest shares of that market?",
                rationale="Concentration changes how the headline should be read.",
                rank=2,
                expected_source_types=[SourceType.REGULATORY_FILING],
                answerable_if="Filings or a regulator report per-insurer shares.",
            ),
        ],
        out_of_scope=["Pet health outcomes unrelated to insurance pricing."],
        assumptions=["'UK' includes Northern Ireland."],
    )


@pytest.fixture
def client(test_settings: Settings) -> Iterator[TestClient]:
    """App with no LLM available — for routing, validation and error paths."""
    with TestClient(create_app(test_settings)) as c:
        yield c


@pytest.fixture
def client_with_fake_llm(
    test_settings: Settings, sample_plan: ResearchPlan
) -> Iterator[tuple[TestClient, FakeAnthropic]]:
    """App whose LLM transport is a fake returning `sample_plan`.

    The background task runs for real, so this exercises the full M1 path
    (route → background → stage → persistence) without a billable call.
    """
    app = create_app(test_settings)
    fake = FakeAnthropic([FakeResponse(parsed_output=sample_plan)])
    app.state.llm_client = LLMClient(test_settings, client=fake)
    with TestClient(app) as c:
        yield c, fake


@pytest.fixture
def client_with_failing_llm(test_settings: Settings) -> Iterator[TestClient]:
    """App whose LLM transport always fails, to exercise the failure path."""
    app = create_app(test_settings)

    class AlwaysFails:
        async def parse(self, **_: Any) -> Any:
            raise RuntimeError("transport exploded")

    class FailingClient:
        def __init__(self) -> None:
            self.messages = AlwaysFails()

    app.state.llm_client = LLMClient(test_settings, client=FailingClient())
    with TestClient(app) as c:
        yield c
