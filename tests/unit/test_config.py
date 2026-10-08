"""Configuration validation and secret-handling behaviour."""

from __future__ import annotations

import pytest
from pydantic import SecretStr
from pydantic import ValidationError as PydanticValidationError

from app.core.config import MissingConfigurationError, Settings, get_settings


def _settings(**kwargs: object) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[arg-type,call-arg]


class TestLogLevel:
    def test_accepts_lowercase_and_normalises(self) -> None:
        assert _settings(log_level="debug").log_level == "DEBUG"

    def test_rejects_unknown_level(self) -> None:
        with pytest.raises(PydanticValidationError, match="log_level must be one of"):
            _settings(log_level="verbose")


class TestGuardrails:
    @pytest.mark.parametrize("bad", [0, -1.5])
    def test_cost_ceiling_must_be_positive(self, bad: float) -> None:
        with pytest.raises(PydanticValidationError, match="greater than 0"):
            _settings(max_cost_usd_per_run=bad)

    def test_negative_chunk_overlap_rejected(self) -> None:
        with pytest.raises(PydanticValidationError, match="cannot be negative"):
            _settings(chunk_overlap_tokens=-1)

    def test_defaults_are_conservative(self) -> None:
        s = _settings()
        assert s.max_cost_usd_per_run > 0
        assert s.max_sources_per_run > 0
        assert s.vector_backend == "sqlite"


class TestSecretHandling:
    def test_missing_anthropic_key_raises_actionable_error(self) -> None:
        s = _settings(anthropic_api_key=None)
        with pytest.raises(MissingConfigurationError, match="ANTHROPIC_API_KEY is not set"):
            s.require_anthropic_key()

    def test_present_key_is_returned_only_via_explicit_call(self) -> None:
        s = _settings(anthropic_api_key=SecretStr("sk-ant-test-value"))
        assert s.require_anthropic_key() == "sk-ant-test-value"

    def test_secret_absent_from_repr_and_str(self) -> None:
        """A key must not appear in a traceback, log line, or debug print."""
        s = _settings(anthropic_api_key=SecretStr("sk-ant-super-secret"))
        assert "sk-ant-super-secret" not in repr(s)
        assert "sk-ant-super-secret" not in str(s)
        assert "sk-ant-super-secret" not in str(s.model_dump())

    def test_missing_postgres_dsn_raises(self) -> None:
        s = _settings(vector_backend="postgres", postgres_dsn=None)
        with pytest.raises(MissingConfigurationError, match="POSTGRES_DSN is required"):
            s.require_postgres_dsn()

    def test_whitespace_stripped_from_key(self) -> None:
        """A trailing newline in .env must not reach the Authorization header."""
        s = _settings(anthropic_api_key=SecretStr("  sk-ant-padded  "))
        assert s.require_anthropic_key() == "sk-ant-padded"


class TestSingleton:
    def test_get_settings_is_cached(self) -> None:
        get_settings.cache_clear()
        assert get_settings() is get_settings()
        get_settings.cache_clear()
