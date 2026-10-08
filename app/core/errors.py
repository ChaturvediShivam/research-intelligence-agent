"""Typed error hierarchy and FastAPI exception handlers.

One base class, one HTTP status per subclass, and a single response shape. The
point is that a caller can branch on `code` programmatically, and that an
unexpected exception never leaks a stack trace or a provider message to the
client — it is logged server-side and returned as an opaque 500.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = structlog.get_logger(__name__)


class AppError(Exception):
    """Base class for every error this application raises deliberately.

    `status_code` and `code` are class attributes so handlers can map any
    subclass without a registry. `details` carries structured, client-safe
    context — never a secret, never a raw upstream payload.
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


# --- Client errors ---------------------------------------------------------


class ValidationError(AppError):
    """Input failed a domain rule that Pydantic alone cannot express."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "validation_error"


class NotFoundError(AppError):
    """A referenced resource (usually a research run) does not exist."""

    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class UnsafeURLError(AppError):
    """A URL was rejected by the SSRF guard (private range, scheme, or denylist).

    A client error rather than a server error: the request named a target the
    service will not fetch, and the reason is safe to return.
    """

    status_code = status.HTTP_400_BAD_REQUEST
    code = "unsafe_url"


class RateLimitedError(AppError):
    """The caller exceeded the local request budget."""

    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"


# --- Server / dependency errors --------------------------------------------


class CostCeilingExceededError(AppError):
    """A run was aborted because it would exceed `max_cost_usd_per_run`.

    Surfaced explicitly instead of silently truncating the research, so the
    report never looks complete when it was cut short for budget.
    """

    status_code = status.HTTP_402_PAYMENT_REQUIRED
    code = "cost_ceiling_exceeded"


class UpstreamError(AppError):
    """A dependency (LLM provider, remote source) failed in a retryable way."""

    status_code = status.HTTP_502_BAD_GATEWAY
    code = "upstream_error"


class PipelineStageError(AppError):
    """A named pipeline stage failed; `details["stage"]` identifies which."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "pipeline_stage_error"

    def __init__(self, stage: str, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message, details={"stage": stage})
        self.stage = stage
        self.cause = cause


def _error_body(code: str, message: str, details: dict[str, Any]) -> dict[str, Any]:
    """The single response shape for every error this service returns."""
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return body


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers so error responses are uniform across the API."""

    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        # Deliberate errors are expected; log at warning without a traceback.
        logger.warning(
            "app_error",
            code=exc.code,
            message=exc.message,
            details=exc.details,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic's own errors, normalised into our envelope. `errors()` can
        # contain non-JSON-serialisable values, so coerce to str.
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=_error_body(
                "validation_error",
                "Request body failed validation.",
                {"fields": [{k: str(v) for k, v in e.items()} for e in exc.errors()]},
            ),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Anything reaching here is a bug. Log it with the traceback, return
        # nothing about it — an upstream provider message could carry detail
        # the client should not see.
        logger.exception("unhandled_exception", exc_type=type(exc).__name__)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body("internal_error", "An internal error occurred.", {}),
        )
