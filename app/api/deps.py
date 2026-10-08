"""FastAPI dependency providers.

Settings are resolved from `request.app.state.settings`, not from the cached
`get_settings()` singleton. That distinction matters: `create_app(settings)`
must actually govern the instance it builds, otherwise a test (or a second app
in one process) silently reads whatever the module-level cache holds. This was
a real bug caught by `test_ready_when_key_configured`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings


def get_app_settings(request: Request) -> Settings:
    """Return the `Settings` the running application was built with."""
    settings = request.app.state.settings
    if not isinstance(settings, Settings):  # pragma: no cover - defensive
        raise RuntimeError("app.state.settings is not configured; use create_app().")
    return settings


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
