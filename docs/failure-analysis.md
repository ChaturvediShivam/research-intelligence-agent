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

---

## F-005 · The SSRF allowlist was an enumeration, and enumerations go stale

**Milestone:** M2
**Found by:** `tests/security/test_ssrf.py` — the `100.64.0.1` case

**Symptom.** `100.64.0.1` passed every check in the SSRF guard and would have
been fetched.

**Cause.** `_classify` enumerated the conditions it knew about —
`is_loopback`, `is_private`, `is_link_local`, `is_multicast`, `is_reserved`,
`is_unspecified`. `100.64.0.0/10` is RFC 6598 carrier-grade NAT space, and
Python reports it as **neither private nor reserved**:

    >>> ipaddress.ip_address("100.64.0.1").is_private
    False
    >>> ipaddress.ip_address("100.64.0.1").is_reserved
    False
    >>> ipaddress.ip_address("100.64.0.1").is_global
    False

Only `is_global` tells the truth about it.

**Fix.** `not ip.is_global` added as a catch-all **after** the specific checks.
The specific checks stay because they produce a useful reason string for the
log and the error; the catch-all is what makes the guard sound. Regression
cases added for CGNAT, TEST-NET-1/2/3, the benchmarking range and the
broadcast address.

**Lesson recorded.** A security check built as a list of known-bad cases is
only as current as the list. Where an authoritative predicate exists, the
enumeration should narrow the message, not define the policy.

---

## F-006 · A defensive `getattr` turned a resource leak into a silent no-op

**Milestone:** M2 (live verification)
**Found by:** intermittent `RuntimeError: Event loop is closed` during live
test teardown — passing on one run and failing on the next

**Symptom.** `uv run pytest -m live` emitted a stray task error:

    Task finished coro=<AsyncClient.aclose()> exception=RuntimeError('Event loop is closed')

It failed the run once and passed the next. A rerun being green was not a fix.

**Cause, in two layers.** The first attempt at a fix was:

    closer = getattr(client, "aclose", None)
    if closer is not None:
        await closer()

`httpx.AsyncClient` exposes `aclose()`. **`anthropic.AsyncAnthropic` exposes
`close()`.** So for the real client the guard found nothing, did nothing, and
reported success. The client was never closed; the garbage collector reaped
its transport wrapper after the event loop had already gone, and the
transport's own cleanup scheduled a task on a dead loop.

The defensive `getattr` did not prevent a failure — it converted a loud one
into an invisible one, and I then shipped a "fix" that fixed nothing.

**Fix.** An explicit `aclose_client()` helper that tries `aclose` then
`close`, awaits whichever is awaitable, and **raises `TypeError` if a client
exposes neither** — because a client whose transport cannot be closed is a
programming error, not a condition to tolerate. Wired into the application
lifespan, which had never closed the LLM client at all, and into the live test
fixtures.

**Pinned by** `tests/unit/test_llm_client.py::TestTransportClosing`, including
a regression test against the real `anthropic.AsyncAnthropic` asserting
`is_closed()` flips to `True`.

**Second-order finding.** Repairing this surfaced that `FakeAnthropic` had no
close method either, so the fakes had a *more forgiving* surface than the real
object — which is how the gap survived 121 offline tests. The fake now mirrors
the real client and names the method `close()`.

---

## F-007 · Live, billable tests ran on every plain `pytest`

**Milestone:** M2
**Found by:** a full suite run taking 191 seconds instead of under one, after
`ANTHROPIC_API_KEY` became available

**Symptom.** `uv run pytest --cov` made real API calls and took over three
minutes.

**Cause.** The `live` marker was registered and the live tests were marked,
but nothing deselected them. They had only ever *appeared* deselected because
no key was configured and their `skipif` guard fired. The moment a key
existed, every ordinary test run started spending money — while the README and
the live-test docstrings both stated they were "deselected by default".

**Fix.** `-m 'not live'` added to `addopts`, so live is genuinely opt-in and
`-m live` on the command line selects it. Verified both directions with
`--collect-only`: default collects 321 and deselects 7; `-m live` collects
exactly the 7.

**Lesson recorded.** A guard that happens to produce the right behaviour for
the wrong reason is not a guard. The skip was environmental; the policy needed
to be explicit.

---

## F-008 · My first eval fixture measured nothing

**Milestone:** M3
**Found by:** reading the first eval output instead of accepting it

**Symptom.** The harness ran, produced numbers, and reported
`recall@10 = 1.000` for hybrid, dense and lexical alike. Taken at face value,
a perfect retrieval baseline.

**Cause.** The fixture had **10 documents and the cutoff was k=10.** Retrieving
ten documents from a corpus of ten necessarily finds every relevant one, so
recall@10 was 1.000 by construction, for any retriever, including a random
one. The metric carried no information.

Two further defects in the same design:

- **No hard negatives.** Every document was relevant to some query, so there
  was nothing for a retriever to be wrong about and precision could not
  discriminate.
- **Documents shorter than one chunk.** Each ~120-word document produced
  exactly one chunk at the shipped 512-token budget, so the chunk→document
  collapse in the harness was never exercised, and fusion had nothing to
  reorder.

**Fix.** Ten hard-negative documents added — same domain, overlapping
vocabulary, relevant to no query — taking the corpus to 20. Cutoffs changed to
report k ∈ {1, 3, 5, 10}, and the harness now prints a warning when a
requested cutoff is at or above the corpus size. recall@1 is 0.800, which
discriminates; the k=10 row is retained only to show that it does not.

**Lesson recorded.** An eval that reports a perfect score on its first run has
almost certainly not been given anything to fail at. The first question to ask
of a new metric is not "is the number good" but "what number would a broken
implementation produce" — and if the answer is the same number, the metric is
decorative. Publishing that 1.000 as an M3 baseline would have been the single
most misleading thing in this repository.

---

## F-009 · The apex domain of a primary TLD was classified UNVETTED

**Milestone:** M4
**Found by:** `tests/security/test_citation_verification.py` — the `www.gov.uk`
credibility case

**Symptom.** `classify_credibility("www.gov.uk")` returned `UNVETTED`.

**Cause.** The primary-tier list holds suffixes with a leading dot
(`".gov.uk"`), matched with `host.endswith(...)`. After stripping `www.`, the
host is `gov.uk` — which does not end with `.gov.uk`, because the apex has no
leading dot. So the single most credible source class in a UK regulatory
corpus fell through to the most cautious tier.

The direction of the error matters: it would have **depressed** confidence on
good sources rather than inflating it on bad ones. Safer, but still wrong, and
it would have made `HIGH` confidence nearly unreachable.

**Fix.** `_matches_suffix()` checks both `host == bare` and
`host.endswith("." + bare)`. Regression cases added for `gov.uk`,
`www.gov.uk`, `europa.eu` and `ac.uk`, plus negative cases (`notgov.uk`,
`fakeac.uk`) so the fix cannot be loosened into a substring match.

---

## F-010 · Haiku 4.5 rejects `output_config.effort`, and the client always sent it

**Milestone:** M4 (live verification)
**Found by:** the first live extraction call — HTTP 400

**Symptom.** Every stage 5 call failed with
`PipelineStageError: Evidence extraction failed: Anthropic rejected the
request as invalid.` Isolated with a two-line probe:

    haiku WITH effort:    400 — "This model does not support the effort parameter."
    haiku WITHOUT effort: OK

**Cause.** `LLMClient.structured` sent `output_config={"effort": effort}`
unconditionally, because ADR-007 says effort must always be explicit. That
rule is right for the Opus-tier models it was written against and wrong for
Haiku 4.5, which rejects the parameter outright. Stage 5 is the first stage to
route to Haiku, so nothing before M4 could have hit it.

No offline test could have caught this. The fake transport accepts any
keyword argument — a fake is, by construction, more permissive than the API,
which is the same shape of blind spot as F-006.

**Fix.** Capability moved into the model table beside the prices
(`ModelPrice.supports_effort`), so the client omits `output_config` for models
that reject it and no call site has to remember. Call sites still state the
effort they want; it is applied where it can be. Pinned by tests at both the
client and the stage, including one asserting that pointing
`EXTRACTION_MODEL` at an effort-capable model does send it.

**ADR-007 amended** rather than abandoned — see that file.

**Lesson recorded.** "Always set X explicitly" is a rule about intent, not
about the wire format. Per-model request capability belongs in data next to
the other per-model facts, not in a convention each call site re-applies.

---

## F-011 · One rich chunk killed the whole run

**Milestone:** M5 (live end-to-end verification)
**Found by:** the first live end-to-end run

**Symptom.** PLAN passed, DISCOVER passed, PROCESS partial, RETRIEVE passed,
then the run died:

    extract: Evidence extraction failed: 1 validation error for ChunkExtraction
    items: List should have at most 3 items after validation, not 4

**Cause, in two layers.**

The surface cause is another guessed bound, the same shape as F-004.
`ChunkExtraction.items` carried `max_length=3`. A passage dense with figures
legitimately yields more, Haiku returned four, and Pydantic rejected the
response — losing all four findings. The constrained decoder did not enforce
the `maxItems` in the schema, so the model was never prevented from exceeding
it.

The deeper cause is the one that mattered: `run_extract_stage` gathered its
per-chunk tasks with a bare `asyncio.gather`, so **one malformed response out
of twenty-four aborted the entire research run**. The architecture is explicit
that a source-level failure must not end a run; extraction had no equivalent
protection, and nothing before M5 ran enough chunks concurrently for it to
show.

**Fix, both layers.**

- Cap raised 3 → 8. Bounded still, but no longer tighter than a good answer.
- `asyncio.gather(..., return_exceptions=True)`. A failing chunk is recorded
  in `ExtractionResult.failures`, costs that chunk's evidence, and nothing
  more. The stage only raises when **every** chunk failed — an empty result
  must not be mistaken for "no evidence found".
- The orchestrator surfaces chunk failures as stage warnings, so the run
  reports PARTIAL rather than hiding them.

**Regression tests.** One failing chunk among two leaves the healthy chunk's
evidence intact with `chunk_failure_rate == 0.5`; all chunks failing raises
with "all 2 chunks" in the message; the cap accepts 4 and still rejects 9.

**Lesson recorded.** Fan-out needs a failure policy decided deliberately, not
inherited from `gather`'s default. Every other fan-out in this project
(`run_process_stage`) already used `return_exceptions=True`; extraction was
written later and did not, and no offline test ran enough chunks to notice.
