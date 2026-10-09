"""FastAPI application factory.

A factory rather than a module-level singleton so tests can build an instance
with overridden settings, and so importing this module has no side effects
beyond defining functions.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.api import routes_health, routes_research
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.llm.client import LLMClient
from app.llm.pricing import is_priced
from app.storage.runs import RunRepository
from app.tools.registry import ToolContext

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown. Logs resolved configuration without secrets."""
    settings: Settings = app.state.settings

    # Fail loudly at startup if a configured model has no price entry. The
    # alternative is reporting that model's cost as zero for the life of the
    # process, which would silently corrupt every figure the project publishes.
    unpriced = [
        model
        for model in (
            settings.planning_model,
            settings.synthesis_model,
            settings.extraction_model,
        )
        if not is_priced(model)
    ]
    if unpriced:
        raise RuntimeError(
            f"No price entry for configured model(s): {sorted(set(unpriced))}. "
            "Add them to app/llm/pricing.py — an unpriced model would report "
            "zero cost."
        )

    # Any run still mid-pipeline belongs to a process that no longer exists:
    # this one has not started executing anything yet. Failing them here is
    # what keeps "a run always reaches a terminal state" true across a
    # restart, deploy or OOM kill (F-019).
    repo: RunRepository = app.state.run_repository
    interrupted = await repo.fail_interrupted()

    logger.info(
        "application_start",
        environment=settings.environment,
        interrupted_runs_failed=len(interrupted),
        vector_backend=settings.vector_backend,
        anthropic_key_configured=settings.anthropic_api_key is not None,
        planning_model=settings.planning_model,
        extraction_model=settings.extraction_model,
    )
    try:
        yield
    finally:
        repo.close()
        client: LLMClient = app.state.llm_client
        await client.aclose()
        components: ToolContext = app.state.components
        await components.aclose()
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
    app.state.run_repository = RunRepository(resolved.database_path)
    app.state.llm_client = LLMClient(resolved)
    # Search provider, fetcher and embedder for the research pipeline. Built
    # lazily inside ToolContext, so constructing the app stays cheap and an
    # instance that never runs research never loads the embedding model.
    # Plain attributes rather than a `yield` dependency on purpose: a
    # generator dependency is torn down when the response is sent, and the
    # pipeline runs in a BackgroundTask *after* that — it would be handed
    # closed clients.
    app.state.components = ToolContext(settings=resolved)

    register_exception_handlers(app)
    app.include_router(routes_health.router)
    app.include_router(routes_research.router)
    return app


app = create_app()
