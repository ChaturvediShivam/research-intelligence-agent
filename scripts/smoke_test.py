"""Deterministic production smoke test.

Checks that a deployed instance is actually serving, without spending a cent:
every endpoint it touches is free. Research endpoints are deliberately not
exercised — proving the HTTP server works should not cost money.

    uv run python scripts/smoke_test.py http://127.0.0.1:9137
    uv run python scripts/smoke_test.py https://<service>.onrender.com

Exit code 0 if every check passes, 1 otherwise, so it is usable in CI or as a
post-deploy gate.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

TIMEOUT = 15
_UA = {"User-Agent": "ria-smoke/1.0"}
# Secret *values* and failure signatures that must never appear in a public
# response. Deliberately not the string "ANTHROPIC_API_KEY": /ready reports
# that name in its `missing` list by design, and flagging it would mean
# flagging correct behaviour. A variable's name is public; its value is not.
SECRET_MARKERS = ("sk-ant-", "Traceback (most recent call last)", "sk-proj-")


class CheckFailed(Exception):
    pass


def _get(url: str) -> tuple[int, dict[str, str], str]:
    # S310: the scheme is validated in main() before any request is made.
    request = urllib.request.Request(url, headers=_UA)  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310  # noqa: S310
            return response.status, _headers(response.headers), response.read().decode()
    except urllib.error.HTTPError as exc:  # a status is still a result
        return (
            exc.code,
            _headers(exc.headers),
            exc.read().decode(errors="replace"),
        )


def _headers(raw: Any) -> dict[str, str]:
    """Lower-case header names: HTTP header casing is not guaranteed."""
    return {str(k).lower(): str(v) for k, v in dict(raw or {}).items()}


def check_health(base: str) -> dict[str, Any]:
    status, headers, body = _get(f"{base}/health")
    if status != 200:
        raise CheckFailed(f"/health returned {status}, expected 200")
    content_type = headers.get("content-type", "")
    if "application/json" not in content_type:
        raise CheckFailed(f"/health content-type was {content_type!r}")
    payload = json.loads(body)
    if payload.get("status") != "ok":
        raise CheckFailed(f"/health status was {payload.get('status')!r}")
    return payload


def check_ready(base: str) -> dict[str, Any]:
    """Readiness is reported, not required.

    An instance with no key configured is honestly not ready, and that is an
    operator problem. The check asserts the endpoint answers truthfully and
    leaks nothing, not that it says yes.
    """
    status, _, body = _get(f"{base}/ready")
    if status != 200:
        raise CheckFailed(f"/ready returned {status}, expected 200")
    payload = json.loads(body)
    for field in ("ready", "environment", "anthropic_key_configured", "missing"):
        if field not in payload:
            raise CheckFailed(f"/ready is missing {field!r}")
    if not isinstance(payload["anthropic_key_configured"], bool):
        raise CheckFailed("/ready must report key configuration as a boolean")
    return payload


def check_no_secret_leak(base: str) -> None:
    """No public response may carry a secret or a traceback."""
    for path in ("/health", "/ready", "/openapi.json", "/nonexistent-path-xyz"):
        _, _, body = _get(f"{base}{path}")
        for marker in SECRET_MARKERS:
            if marker in body:
                raise CheckFailed(f"{path} response contained {marker!r}")


def check_unknown_route_is_handled(base: str) -> None:
    """A missing route must 404 cleanly, not 500 with a stack trace."""
    status, _, body = _get(f"{base}/nonexistent-path-xyz")
    if status != 404:
        raise CheckFailed(f"unknown route returned {status}, expected 404")
    if "Traceback" in body:
        raise CheckFailed("unknown route leaked a traceback")


def main(base: str) -> int:
    # Validated once, here: the URL comes from argv, and `file:` would turn a
    # smoke test into a local file read.
    if urllib.parse.urlsplit(base).scheme not in {"http", "https"}:
        print(f"refusing non-HTTP(S) URL: {base!r}")
        return 2
    base = base.rstrip("/")
    print(f"smoke test against {base}\n")

    checks = (
        ("health endpoint", lambda: check_health(base)),
        ("readiness endpoint", lambda: check_ready(base)),
        ("unknown route handled", lambda: check_unknown_route_is_handled(base)),
        ("no secret or traceback leaked", lambda: check_no_secret_leak(base)),
    )

    failures = 0
    for name, check in checks:
        try:
            result = check()
        except (CheckFailed, OSError, ValueError) as exc:
            print(f"  FAIL  {name}: {exc}")
            failures += 1
        else:
            detail = f" {result}" if isinstance(result, dict) else ""
            print(f"  ok    {name}{detail}")

    print()
    if failures:
        print(f"{failures} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
