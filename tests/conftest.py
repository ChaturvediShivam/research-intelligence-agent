"""Shared fixtures.

Tests never read the developer's real `.env`: every fixture builds `Settings`
explicitly so a locally-exported ANTHROPIC_API_KEY cannot change test
outcomes, and so no test can accidentally make a billable call.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def test_settings() -> Settings:
    """Deterministic settings with no credentials configured."""
    return Settings(
        environment="test",
        log_level="DEBUG",
        anthropic_api_key=None,
        postgres_dsn=None,
        vector_backend="sqlite",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def client(test_settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(test_settings)) as c:
        yield c
