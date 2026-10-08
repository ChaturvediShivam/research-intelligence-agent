"""Secret redaction in logs.

Part of the Definition of Done: "no secret appears in any log line".
These tests capture real rendered output and assert the credential is absent
from the emitted bytes, rather than inspecting the processor in isolation.
"""

from __future__ import annotations

import json

import pytest
import structlog

from app.core.logging import REDACTED, configure_logging, redact_secrets

ANTHROPIC_KEY = "sk-ant-api03-AAAAbbbbCCCCddddEEEEffffGGGG"


def _process(**event: object) -> dict[str, object]:
    """Run the processor directly on an event dict."""
    return dict(redact_secrets(None, "info", dict(event)))  # type: ignore[arg-type]


class TestKeyNameRedaction:
    @pytest.mark.parametrize(
        "key",
        [
            "api_key",
            "API_KEY",
            "anthropic_api_key",
            "x-api-key",
            "authorization",
            "Authorization",
            "access_token",
            "refresh_token",
            "password",
            "client_secret",
            "postgres_dsn",
            "cookie",
        ],
    )
    def test_sensitive_key_value_is_replaced(self, key: str) -> None:
        out = _process(**{key: "super-secret-value"})
        assert out[key] == REDACTED
        assert "super-secret-value" not in json.dumps(out)

    def test_ordinary_keys_pass_through(self) -> None:
        out = _process(url="https://example.com/a", status=200, stage="plan")
        assert out["url"] == "https://example.com/a"
        assert out["status"] == 200


class TestValuePatternRedaction:
    """Layer 2: a credential under an innocuous key must still be caught."""

    def test_anthropic_key_under_harmless_key(self) -> None:
        out = _process(note=f"calling with {ANTHROPIC_KEY} now")
        assert ANTHROPIC_KEY not in json.dumps(out)
        assert REDACTED in str(out["note"])

    def test_bearer_token(self) -> None:
        out = _process(header="Bearer abcdefghijklmnopqrstuvwxyz123456")
        assert "abcdefghijklmnopqrstuvwxyz123456" not in json.dumps(out)

    def test_dsn_credentials_removed_but_scheme_kept(self) -> None:
        """The log should still say what was contacted, not with whose password."""
        out = _process(target="postgresql://admin:hunter2@db.internal:5432/research")
        rendered = str(out["target"])
        assert "hunter2" not in rendered
        assert "admin" not in rendered
        assert rendered.startswith("postgresql://")
        assert "db.internal" in rendered

    def test_nested_dict_and_list_are_redacted(self) -> None:
        out = _process(
            payload={"inner": {"note": ANTHROPIC_KEY}},
            items=[f"x {ANTHROPIC_KEY}", "clean"],
            pair=("a", ANTHROPIC_KEY),
        )
        blob = json.dumps(out, default=str)
        assert ANTHROPIC_KEY not in blob
        assert "clean" in blob

    def test_non_string_values_survive(self) -> None:
        out = _process(count=3, ratio=0.5, flag=True, nothing=None)
        assert out == {"count": 3, "ratio": 0.5, "flag": True, "nothing": None}


class TestEndToEndRenderedOutput:
    """The real test: does the credential reach stdout?"""

    def test_key_absent_from_rendered_json_log(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="INFO", json_output=True)
        log = structlog.get_logger("test")
        log.info(
            "llm_request",
            anthropic_api_key=ANTHROPIC_KEY,
            note=f"inline {ANTHROPIC_KEY}",
            dsn="postgresql://u:p@h/db",
            url="https://api.anthropic.com/v1/messages",
        )
        captured = capsys.readouterr().out
        assert ANTHROPIC_KEY not in captured
        assert "hunter2" not in captured
        assert REDACTED in captured
        # The useful, non-secret context is still there.
        assert "llm_request" in captured
        assert "api.anthropic.com" in captured

    def test_key_absent_from_rendered_console_log(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="INFO", json_output=False)
        structlog.get_logger("test").warning("retry", api_key=ANTHROPIC_KEY)
        assert ANTHROPIC_KEY not in capsys.readouterr().out
