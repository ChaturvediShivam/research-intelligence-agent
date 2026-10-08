"""FastAPI application factory.

A factory rather than a module-level `app` singleton so that tests can build
an instance with overridden settings, and so import of this module has no side
effects beyond defining functions.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.api import routes_health
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown. Logs the resolved configuration without secrets."""
    settings: Settings = app.state.settings
    logger.info(
        "application_start",
        environment=settings.environment,
        vector_backend=settings.vector_backend,
        anthropic_key_configured=settings.anthropic_api_key is not None,
    )
    yield
    logger.info("application_stop")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Pass `settings` to override configuration."""
    resolved = settings or get_settings()
    configure_logging(level=resolved.log_level, json_output=resolved.log_json)

    app = FastAPI(
        title="Research Intelligence Agent",
        description=(
            "Evidence-first research and due-diligence agent. Source-grounded "
            "synthesis with deterministic citation verification."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = resolved
    register_exception_handlers(app)
    app.include_router(routes_health.router)
    return app


app = create_app()
