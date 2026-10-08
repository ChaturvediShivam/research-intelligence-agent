"""Health and readiness endpoint behaviour."""

from __future__ import annotations

from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import Settings
from app.main import create_app


class TestLiveness:
    def test_health_is_ok_without_any_credentials(self, client: TestClient) -> None:
        """Liveness must not depend on external configuration."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "version": "0.1.0"}


class TestReadiness:
    def test_not_ready_and_names_missing_key(self, client: TestClient) -> None:
        response = client.get("/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is False
        assert "ANTHROPIC_API_KEY" in body["missing"]
        assert body["anthropic_key_configured"] is False

    def test_ready_when_key_configured(self) -> None:
        settings = Settings(
            environment="test",
            anthropic_api_key=SecretStr("sk-ant-test"),
            _env_file=None,  # type: ignore[call-arg]
        )
        with TestClient(create_app(settings)) as c:
            body = c.get("/ready").json()
        assert body["ready"] is True
        assert body["missing"] == []
        assert body["anthropic_key_configured"] is True

    def test_postgres_backend_requires_dsn(self) -> None:
        settings = Settings(
            environment="test",
            anthropic_api_key=SecretStr("sk-ant-test"),
            vector_backend="postgres",
            postgres_dsn=None,
            _env_file=None,  # type: ignore[call-arg]
        )
        with TestClient(create_app(settings)) as c:
            body = c.get("/ready").json()
        assert body["ready"] is False
        assert "POSTGRES_DSN" in body["missing"]

    def test_readiness_never_returns_the_key_itself(self) -> None:
        settings = Settings(
            environment="test",
            anthropic_api_key=SecretStr("sk-ant-do-not-leak"),
            _env_file=None,  # type: ignore[call-arg]
        )
        with TestClient(create_app(settings)) as c:
            assert "sk-ant-do-not-leak" not in c.get("/ready").text
