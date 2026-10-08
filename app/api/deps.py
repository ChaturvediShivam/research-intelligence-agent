"""FastAPI dependency providers.

Settings, the run repository and the LLM client are all resolved from
`request.app.state`, not from module-level singletons. `create_app(settings)`
must actually govern the instance it builds — resolving the cached
`get_settings()` singleton in a route was a real bug (see
docs/failure-analysis.md F-001).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings
from app.llm.client import LLMClient
from app.storage.runs import RunRepository
from app.tools.registry import ToolContext


def get_app_settings(request: Request) -> Settings:
    """Return the `Settings` the running application was built with."""
    settings = request.app.state.settings
    if not isinstance(settings, Settings):  # pragma: no cover - defensive
        raise RuntimeError("app.state.settings is not configured; use create_app().")
    return settings


def get_run_repository(request: Request) -> RunRepository:
    """Return the process-wide run repository."""
    repo = request.app.state.run_repository
    if not isinstance(repo, RunRepository):  # pragma: no cover - defensive
        raise RuntimeError("app.state.run_repository is not configured.")
    return repo


def get_llm_client(request: Request) -> LLMClient:
    """Return the shared LLM client.

    Constructed without touching the API key, so an app with no key configured
    still starts; the key is required at the point of the first call.
    """
    client = request.app.state.llm_client
    if not isinstance(client, LLMClient):  # pragma: no cover - defensive
        raise RuntimeError("app.state.llm_client is not configured.")
    return client


def get_components(request: Request) -> ToolContext:
    """Return the shared pipeline components.

    `ToolContext` already builds the search provider, fetcher and embedder
    lazily and closes whatever it built, which is exactly what the HTTP layer
    needs — so it is reused rather than reimplemented. Lazy matters here: the
    embedding model takes about 14 seconds to load on first use, and an app
    that only ever serves /health must not pay for it.
    """
    components = request.app.state.components
    if not isinstance(components, ToolContext):  # pragma: no cover - defensive
        raise RuntimeError("app.state.components is not configured; use create_app().")
    return components


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
RunRepositoryDep = Annotated[RunRepository, Depends(get_run_repository)]
LLMClientDep = Annotated[LLMClient, Depends(get_llm_client)]
ComponentsDep = Annotated[ToolContext, Depends(get_components)]
