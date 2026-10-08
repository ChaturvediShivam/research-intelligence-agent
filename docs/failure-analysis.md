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

---

## F-004 · A guessed schema bound rejected a correct model response

**Milestone:** M1 (live verification)
**Found by:** `tests/integration/test_live_plan.py` — the first real API call

**Symptom.** The API returned HTTP 200 and a well-formed plan. The pipeline
then failed with `PipelineStageError: Planning failed: 1 validation error for
ResearchPlan — restated_question: String should have at most 600 characters`.

**Cause.** My own schema, not the model. `restated_question` carried
`max_length=600`, a number I guessed. The prompt instructs the planner to name
the entity, geographic and sector scope, time period, and unit of measurement
explicitly — a restatement that does all four runs longer than that. The
constraint was fighting the instruction. The real response measured **671
characters**.

**Why no offline test caught it.** Every fixture was one I wrote, and I wrote
them inside my own bounds. A fake transport can only prove the code handles a
response shaped the way I imagined. This is precisely the class of defect the
live milestone gate exists to catch, and the reason M1 was not marked complete
on the strength of 121 passing offline tests.

**Fix.** Bounds raised from evidence, not from a second guess:
`restated_question` 600 → 2000; `question` / `rationale` / `answerable_if`
500 → 1000. `out_of_scope` and `assumptions` had **no per-item cap at all** —
only a list-length cap — so each item gained a 1000-character bound while I
was there.

**Also changed.** The live test now prints the observed length of every text
field, so these bounds stay evidence-backed rather than being re-guessed next
time. Measured on the passing run: restated_question 671; sub-question text
158–267; rationale 177–283; answerable_if 246–366. The new limits have
headroom without being unbounded.

**Cost of the lesson.** Two billable calls whose cost is unmeasurable: the
validation exception discarded the response before `usage` was read, so the
spend on a schema-rejected call is not captured. Noted rather than fixed —
capturing usage on a validation failure is a real gap, logged here as a known
limitation rather than silently absorbed.
