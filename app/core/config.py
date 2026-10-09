"""Application configuration.

Every secret is read from the environment (or a gitignored `.env`) and never
from code. `Settings` is constructed once per process and injected via
`app.api.deps`, so no module reaches for `os.environ` directly — that keeps the
set of required environment variables discoverable in exactly one place.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "production"]
VectorBackend = Literal["sqlite", "postgres"]


class Settings(BaseSettings):
    """Runtime configuration, validated at startup.

    Secrets use `SecretStr` so that an accidental `repr()`, log line, or
    exception payload prints `**********` instead of the credential. The raw
    value is only reachable through an explicit `.get_secret_value()` call,
    which makes every read of a secret greppable in review.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # A stray `ANTHROPIC_API_KEY=` with trailing whitespace is a real and
        # annoying failure mode; strip it rather than send a malformed header.
        str_strip_whitespace=True,
    )

    environment: Environment = "local"
    log_level: str = "INFO"
    # JSON logs in production (machine-parseable); console renderer locally.
    log_json: bool = False

    # --- Anthropic ---------------------------------------------------------
    # Optional at import time so the process can boot, lint, and run unit tests
    # without a key. Code paths that genuinely need it call
    # `require_anthropic_key()` and fail loudly at the point of use instead of
    # silently sending an unauthenticated request.
    anthropic_api_key: SecretStr | None = None
    planning_model: str = "claude-opus-5-5"
    synthesis_model: str = "claude-opus-5-5"
    extraction_model: str = "claude-haiku-4-5"

    # --- Storage -----------------------------------------------------------
    database_path: Path = Path("data/runs.db")
    vector_backend: VectorBackend = "sqlite"
    postgres_dsn: SecretStr | None = None

    # --- Retrieval ---------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Chunks per ONNX forward pass. A memory bound set by the instance size,
    # not a throughput preference: see DEFAULT_BATCH_SIZE in
    # app/retrieval/embeddings.py for the measurements behind the default.
    # Raise it only with headroom to spare; the default fits 512 MB.
    embedding_batch_size: int = 4
    chunk_tokens: int = 512
    chunk_overlap_tokens: int = 64
    retrieval_top_k: int = 12

    # --- Guardrails --------------------------------------------------------
    # A hard ceiling per research run. The pipeline aborts rather than
    # overspending on a pathological question.
    max_cost_usd_per_run: float = 2.0
    max_sources_per_run: int = 12
    fetch_timeout_seconds: float = 20.0
    # Empty means "allow any public host"; SSRF defence is enforced separately
    # in app.core.security and is not optional.
    allowed_source_domains: tuple[str, ...] = ()
    blocked_source_domains: tuple[str, ...] = ()

    # --- SEC EDGAR ---------------------------------------------------------
    # sec.gov rejects automated requests that do not declare a contact
    # address, returning 403 (sec.gov/os/webmaster-faq). Environment-driven
    # (SEC_CONTACT_EMAIL) rather than committed, because it is a real personal
    # or role address and does not belong in the repository. Unset means SEC
    # sources stay unreadable, which is the honest default: fabricating a
    # contact address to satisfy a policy would defeat the policy.
    sec_contact_email: str | None = None

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {value!r}")
        return upper

    @field_validator("sec_contact_email")
    @classmethod
    def _validate_sec_contact(cls, value: str | None) -> str | None:
        """Reject a malformed contact rather than let SEC reject every fetch.

        A typo here fails at startup. Without this it fails much later, as a
        403 indistinguishable from the one this setting exists to fix.
        """
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if "@" not in cleaned or any(c.isspace() for c in cleaned):
            raise ValueError(
                "sec_contact_email must be a single email address, e.g. "
                "name@example.com; SEC requires a contactable address."
            )
        return cleaned

    @field_validator("max_cost_usd_per_run")
    @classmethod
    def _validate_cost_ceiling(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("max_cost_usd_per_run must be greater than 0")
        return value

    @field_validator("embedding_batch_size")
    @classmethod
    def _batch_size_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("embedding_batch_size must be at least 1")
        return value

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def _validate_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError("chunk_overlap_tokens cannot be negative")
        return value

    def require_anthropic_key(self) -> str:
        """Return the API key, or fail with an actionable message.

        Called at the point of use so that a missing key surfaces as a clear
        configuration error rather than a 401 from the provider.
        """
        if self.anthropic_api_key is None:
            raise MissingConfigurationError(
                "ANTHROPIC_API_KEY is not set. Export it, or add it to .env "
                "(see .env.example). No LLM call can be made without it."
            )
        # str_strip_whitespace does not apply inside SecretStr, so strip here:
        # a trailing newline in .env must never reach the Authorization header.
        return self.anthropic_api_key.get_secret_value().strip()

    def require_postgres_dsn(self) -> str:
        """Return the Postgres DSN for the pgvector backend (ADR-004)."""
        if self.postgres_dsn is None:
            raise MissingConfigurationError(
                "POSTGRES_DSN is required when VECTOR_BACKEND=postgres."
            )
        return self.postgres_dsn.get_secret_value().strip()


class MissingConfigurationError(RuntimeError):
    """Raised when a required setting is absent at the point it is needed.

    Defined here rather than in `app.core.errors` to keep `config` free of
    internal imports — it is the lowest layer in the application and must stay
    importable on its own.
    """


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached so that `.env` is read once. Tests that need different values
    construct `Settings(...)` directly or call `get_settings.cache_clear()`.
    """
    return Settings()
