"""Structured logging with mandatory secret redaction.

Two independent layers, because either one alone fails in practice:

1. **Key-name redaction** — any event key that looks like a credential
   (`api_key`, `authorization`, `dsn`, ...) has its value replaced.
2. **Value-pattern redaction** — any string anywhere in the event that matches
   a known credential shape (`sk-ant-...`, `Bearer ...`, a DSN with inline
   credentials) is rewritten, even under an innocuous key.

Layer 1 misses `logger.info("calling", url="https://u:p@host/db")`.
Layer 2 misses nothing shaped like a secret but catches it late. Together they
cover both the careless key and the careless value.

The redaction is enforced by `tests/security/test_log_redaction.py`, which is
part of the Definition of Done — not a convention.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

import structlog
from structlog.typing import EventDict, WrappedLogger

REDACTED = "***REDACTED***"

# Event keys whose values are always credentials, matched case-insensitively
# on substring so `anthropic_api_key` and `x-api-key` are both covered.
SENSITIVE_KEY_PARTS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "access_token",
    "refresh_token",
    "secret",
    "password",
    "passwd",
    "credential",
    "dsn",
    "cookie",
    "session_token",
)

# Credential shapes. Kept deliberately broad: a false positive costs a
# redacted log line, a false negative leaks a key.
SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Anthropic keys (sk-ant-...) and admin keys (sk-ant-admin...).
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{8,}"),
    # Generic provider-style keys.
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    # HTTP Authorization header values.
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]{10,}=*"),
    # Credentials embedded in a URL or DSN: scheme://user:password@host
    re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^:/\s@]+:[^@/\s]+@"),
)


def _redact_string(value: str) -> str:
    """Rewrite any credential-shaped substring inside `value`."""
    redacted = value
    for pattern in SECRET_VALUE_PATTERNS:
        if pattern is SECRET_VALUE_PATTERNS[-1]:
            # Preserve the scheme so the log still says *what* was contacted,
            # just not with whose credentials.
            redacted = pattern.sub(rf"\1{REDACTED}@", redacted)
        else:
            redacted = pattern.sub(REDACTED, redacted)
    return redacted


def _redact_value(value: Any) -> Any:
    """Recursively redact strings inside nested containers."""
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        rebuilt = [_redact_value(v) for v in value]
        return tuple(rebuilt) if isinstance(value, tuple) else rebuilt
    return value


def _is_sensitive_key(key: str) -> bool:
    """Match header-style and snake_case names alike.

    Hyphens are normalised to underscores so that `x-api-key`, `X-Api-Key`
    and `anthropic_api_key` all match the single `api_key` pattern. Caught by
    the `x-api-key` case in tests/security/test_log_redaction.py.
    """
    lowered = key.lower().replace("-", "_")
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def redact_secrets(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """structlog processor applying both redaction layers to every event."""
    for key, value in list(event_dict.items()):
        if _is_sensitive_key(key):
            event_dict[key] = REDACTED
        else:
            event_dict[key] = _redact_value(value)
    return event_dict


def configure_logging(*, level: str = "INFO", json_output: bool = False) -> None:
    """Configure structlog once per process.

    `redact_secrets` is positioned last in the processor chain so it also sees
    keys added by earlier processors.
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
        force=True,
    )

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            redact_secrets,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
