"""Liveness and readiness endpoints.

The split matters for deployment: `/health` answers "is the process up" and
must never depend on anything external, while `/ready` answers "can this
instance actually serve traffic" and reports which capabilities are
configured. A deploy platform polls the former; an operator reads the latter.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.deps import SettingsDep

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class ReadinessResponse(BaseModel):
    """Capability report. `ready` is false when a required dependency is absent."""

    ready: bool
    environment: str
    vector_backend: str
    # Reported as booleans, never as values — this response is public.
    anthropic_key_configured: bool
    missing: list[str]


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness: the process is running and can serve a request."""
    return HealthResponse(status="ok", version="0.1.0")


@router.get("/ready", response_model=ReadinessResponse)
async def ready(settings: SettingsDep) -> ReadinessResponse:
    """Readiness: every dependency needed to run research is configured."""
    missing: list[str] = []
    if settings.anthropic_api_key is None:
        missing.append("ANTHROPIC_API_KEY")
    if settings.vector_backend == "postgres" and settings.postgres_dsn is None:
        missing.append("POSTGRES_DSN")

    return ReadinessResponse(
        ready=not missing,
        environment=settings.environment,
        vector_backend=settings.vector_backend,
        anthropic_key_configured=settings.anthropic_api_key is not None,
        missing=missing,
    )
