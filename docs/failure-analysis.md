# Failure analysis

Real failures found while building, and what changed as a result. Added as they
happen — not reconstructed at the end.

---

## F-001 · Dependency injection silently ignored the injected settings

**Milestone:** M0
**Found by:** `tests/integration/test_health.py::TestReadiness::test_ready_when_key_configured`

**Symptom.** `create_app(settings)` was passed settings with an API key
configured. The startup log correctly reported `anthropic_key_configured=True`,
but `GET /ready` returned `ready: false` and listed `ANTHROPIC_API_KEY` as
missing.

**Cause.** The route dependency resolved `get_settings()` — the
`lru_cache`-backed module singleton — rather than the settings the application
was actually constructed with. The factory argument governed the app object but
not the request path.

**Why it nearly passed unnoticed.** The first two readiness tests asserted
`ready is false` with no key in the environment. They passed for the wrong
reason: the global cache also had no key. Only the positive-case test, which
required the injected value to be honoured, exposed it.

**Fix.** `app/api/deps.py` now reads `request.app.state.settings`. The cached
singleton is used only as the default when `create_app()` is called without
arguments.

**Lesson recorded.** A test that asserts the *absence* of something can pass
for the wrong reason. Each configuration-dependent behaviour needs a positive
case as well as a negative one.

---

## F-002 · Header-style secret keys escaped log redaction

**Milestone:** M0
**Found by:** `tests/security/test_log_redaction.py` — the `x-api-key` case

**Symptom.** `logger.info("req", **{"x-api-key": "..."})` emitted the value in
clear text.

**Cause.** The sensitive-key patterns were snake_case (`api_key`), and
substring matching against `x-api-key` failed on the hyphen. HTTP header names
use hyphens, and headers are exactly what gets logged around an API call.

**Fix.** Key names are normalised (`-` → `_`) before matching, so `x-api-key`,
`X-Api-Key` and `anthropic_api_key` all match one pattern.

**Lesson recorded.** The value-pattern layer would have caught a real
`sk-ant-…` string anyway. Two independent redaction layers are what turned a
near-miss into a caught bug — neither layer alone was sufficient.

---

## F-003 · A trailing newline in `.env` would have reached the auth header

**Milestone:** M0
**Found by:** `tests/unit/test_config.py::test_whitespace_stripped_from_key`

**Symptom.** `Settings(anthropic_api_key=SecretStr("  sk-ant-x  "))` returned
the padded value from `require_anthropic_key()`.

**Cause.** `str_strip_whitespace=True` on the settings model does not reach
inside a `SecretStr`. A key pasted into `.env` with a trailing newline would
have been sent in the `x-api-key` header verbatim, producing a confusing 401
rather than a configuration error.

**Fix.** Strip at the point of use, in `require_anthropic_key()` and
`require_postgres_dsn()`.
