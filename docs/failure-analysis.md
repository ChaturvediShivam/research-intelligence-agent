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

---

## F-012 · The golden set measured alphabetical position, not retrieval

**Milestone:** M7
**Found by:** reading a baseline number that looked impossible — G03's
`source_coverage` was 0.00 for a case whose single relevant document was in
the corpus

**Symptom.** Baseline `retrieval_relevance` 0.583, with recall 0.00 on four
cases whose relevant documents plainly existed.

**Cause.** `build_golden_v1.py` set `available_docs` to the whole 20-document
corpus, **sorted alphabetically**. The discovery stage asks its search tool
for at most 8 results per query. So every case discovered the same
alphabetically-first eight documents, and `pra-solvency`, `hmrc-ipt`,
`insurtech-funding` and others could never be discovered at all. Recall was
measuring where a document's name fell in the alphabet.

**Fix.** Each case now gets its judged documents plus distractors drawn in
fixed order from the hard-negative pool, sized to fit inside the discovery
cap. **No judgment was altered** — only which documents are made available.
Re-measured: `retrieval_relevance` 0.583 → **0.833**, `source_coverage`
0.750 → **1.000**.

**Lesson recorded.** The first baseline number to distrust is the one that
disagrees with something you can check by hand. A metric that silently
depends on an unrelated implementation bound measures that bound.

---

## F-012b · The golden-set validator rejected a legitimate case shape

Found immediately after: the validator refused any case that both judged a
document relevant and expected UNKNOWN. That is exactly the shape of a
paywall or fetch-failure case — the relevant source exists and cannot be
read. The rule now applies only to `healthy` cases, and cases that make every
source unreadable are instead *required* to expect UNKNOWN.

---

## F-013 · A guessed bound rejected a correct response, for the third time

**Milestone:** M7
**Found by:** the first judge call

**Symptom.** `ValidationError: reason — String should have at most 600
characters`. The judge's explanation of a five-criterion rubric score was
longer than 600 characters, so the whole response was rejected.

**Cause.** The same defect as F-004 (`restated_question`, 600) and F-011
(`ChunkExtraction.items`, 3): an arbitrary `max_length` on a field a model
writes, tighter than a correct answer needs.

**Fix.** 600 → 2000.

**Lesson recorded, three occurrences in.** My habit of bounding
model-generated text fields tightly is a recurring source of defects, and in
every case the bound was invented rather than measured. The bound should come
from observed output — as the chunking and citation code already does — or be
generous enough that only a runaway response hits it. Worth a standing check
on any new schema field a model fills.

---

## F-014 · A successful run produced a near-empty report

**Milestone:** M7 (the improvement cycle)
**Found by:** baseline `report_quality` = 0.183, the weakest of the seven
metrics

**Symptom.** On a run with no source failures and no information gaps, the
report collapsed to one generic limitation ("Citation verification proves a
quote is present…") and one generic next step ("Corroborate the findings
above…"). The rubric criteria "limitations specific to this run rather than
generic boilerplate" and "next steps actionable and tied to a stated gap"
both failed.

**Cause.** `_limitations` and `_next_steps` were driven entirely by
*failures*. A run where nothing failed therefore had nothing to say — which
is precisely when a reader most needs to know what the findings still do not
establish. Meanwhile the run held unused measured facts: on G02 all four
supported claims had `corroboration == 1`, two were LOW confidence, and six
of eight sources read were unvetted.

**Fix.** Both functions now also report evidence-level limitations derived
from existing verified output: uncorroborated claim count, LOW-confidence
claim count, and unvetted-source composition, with next steps tied to those
counts. No new model call, no schema change.

**Outcome — and this is the part worth reading.** `report_quality` moved
0.183 → 0.208. But `answer_relevance` moved 0.583 → 0.542 in the same run,
and **my change cannot affect answer generation at all.** So I re-ran the
evaluation on identical code to measure the judge's noise floor:

| Run | answer_relevance | report_quality |
|---|---|---|
| baseline | 0.583 | 0.183 |
| after change | 0.542 | 0.208 |
| identical code, re-run | 0.583 | 0.200 |

The judge varies by **±0.04** on 6 cases with no code change — larger than
the +0.025 the change produced. **The improvement is therefore not
demonstrated by the metric.** The change is kept because it is independently
correct (it surfaces measured facts that were being discarded) and regresses
none of the five deterministic metrics, but no claim is made that it improved
report quality.

**Lesson recorded.** A single before/after pair on a non-deterministic metric
is not evidence. Measuring the noise floor cost one extra run and converted a
tempting claim into an honest one.

---

## F-015 · A stage that raised left no trace record at all

**Milestone:** M9
**Found by:** writing the per-stage trace assertion M9 requires — "exceptions
do not silently disappear from traces"

**Symptom.** `KeyError: <Stage.PLAN: 'plan'>`. A run whose PLAN stage raised
reported `status=failed` with a correct `error` string, but its exported trace
contained **no stage records whatsoever** — not a FAILED one for the stage
that raised, and not SKIPPED ones for the stages that never ran.

**Cause.** Each stage records its own outcome *after* completing. A stage that
raises therefore never reaches its own `_record` call, and the exception
unwinds to the handler in `run()`, which set `status` and `error` but recorded
nothing. The run knew what failed; the trace could not say.

This had been true since M5 and was invisible because nothing had asked the
trace to attribute a failure to a stage before. The log line
(`run_failed stage=plan`) carried the information, so an operator reading logs
would have been fine — but an operator reading the trace would have seen a
failed run with no failed stage.

**Fix.** `_record_raised_stage` in the `PipelineStageError` handler — the one
place that already knows which stage raised, from `exc.stage`. It records that
stage FAILED with the message, then marks the rest SKIPPED. It is deliberately
narrow: it declines to guess when the stage name is unrecognised, and it will
not overwrite an outcome the stage already recorded.

**Regression test.** `test_an_exception_does_not_vanish_from_the_trace` and
`test_stages_that_never_ran_are_still_listed`.

**One existing test conflicted**, and the resolution is worth recording
because "a test went red" is where corners get cut. `TestPlannerFailure`
asserted `result.stages == [] or all(s.stage is Stage.PLAN ...)` — written
when a raising stage produced nothing, so it could only check that nothing
*unexpected* was present. Its stated intent, "no stage after PLAN may have
run", is still satisfied: SKIPPED records that they did not run. The
assertion was therefore **tightened**, not relaxed — it now requires PLAN to
be FAILED and every other stage to be SKIPPED, which is strictly stronger
than what it checked before.

**Lesson recorded.** The gap existed because success and failure were
instrumented asymmetrically: the happy path recorded itself, the error path
recorded only at the top. Worth checking the same asymmetry anywhere else
observability is bolted on after the fact.

---

## F-016 · A code fence labelled as a role escaped injection telemetry

**Milestone:** M9

**Symptom.** Corpus case INJ07 — a ```` ```system ```` block — produced no
injection signal, while every other marker-bearing case did.

**Cause.** The `fake_role` patterns all required a colon (`system:`,
`developer:`). A fenced block labels its role without one.

**Fix.** Added ```` ```system ````, `~~~system` and `<system` to the pattern
set. Still telemetry, not a filter.

**Worth noting:** writing this corpus also established that **2 of 15 cases
are undetectable by design** — INJ10 (social engineering) and INJ14
(pseudo-configuration) contain no marker word at all. Rather than stretch the
word list until it caught them and started flagging real documents, both are
documented as the reason the defence has to be structural. The test asserts
the exact set `{INJ10, INJ14}`, so the limitation cannot drift unnoticed.

---

## F-017 · The HTTP API never left the planning stage, and a test confirmed it

**Milestone:** found in M10, introduced in M1

**Symptom.** The deployed service accepted a research question, returned
`status: "completed"`, and produced a plan and nothing else — no sources, no
evidence, no citations, no report. The live run that exposed it reported
`cost.by_stage = {"plan": 0.064917}`: one stage billed, where a complete run
bills eight.

**Cause.** `app/api/routes_research.py` was the M1 implementation. Its
background task called `run_plan_stage` and then set `COMPLETED`, with the
comment *"M1 ends after planning. Stages 2-10 extend this in later
milestones."* Those milestones extended `ResearchOrchestrator` and the MCP
tool registry; the HTTP route was never migrated. The pipeline was complete
and fully tested — it simply had no caller on the REST surface.

**Why it survived nine milestones.** `tests/integration/test_research_api.py`
asserted `detail["status"] == RunStatus.COMPLETED.value` after planning
alone. The bug was encoded as the expected contract, so every suite run since
M1 **actively confirmed it**. A scan for `TODO|FIXME|stub|NotImplementedError`
across `app/` returned nothing relevant: the only marker was a prose comment
in a docstring, which no tool treats as incomplete work.

**Fix.** The route now builds the real `ResearchOrchestrator`, adopts the
`run_id` already created by the repository, applies `request.max_sources`
through `Settings.model_copy`, persists plan, report and trace, and takes its
terminal status from `result.status` rather than hard-coding one. A `report`
column was added as an additive migration, because the production database on
Render's mounted disk already existed without it.

**Worth noting:** this is the same class as F-012 — a test that measured the
wrong thing and passed. Both were found by checking an *external* artifact
(there, the golden set's discoverability; here, a live run's per-stage cost)
rather than by reading code or trusting a green suite. A passing test suite
is evidence that the code matches its assertions, not that the assertions
describe the intended product. The cheapest guard found so far is asserting
on a measured side effect — cost attributed across stages — rather than on a
status string the code is free to set.

---

## F-018 · A failure code was counted as a rejected citation

**Milestone:** M10, found by the first production research run

**Symptom.** Production run `run_a16334dd6722474c` returned a report whose
sub-questions reported `rejected_citation_count: 1` for SQ1 and SQ3, while
the report's own `rejected_citations` total said `0`. The gap reason read
*"1 citation(s) were produced and none survived verification against the
stored source text."* **No citation had been produced at all** — both claims
carried `failures: ['no_citation']` and an empty `citations` list.

**Cause.** Two different quantities shared a name. At report level,
`rejected_citations = claim_validation.rejected_count` counts citation
*verdicts* that failed; a claim with no citation produces no verdict, so `0`
was right. At sub-question level it was `sum(len(c.failures) for c in
unknown)` — a count of failure *codes*. A claim rejected for carrying no
citation has exactly one code, `no_citation`, so it was counted as one
rejected citation. The two numbers could never agree, and the count had no
relationship to citations at all: a claim with two failure codes would have
reported two rejected citations.

**Fix.** Count citations that failed verification:
`sum(len(c.citations) - c.verified_citations for c in (*supported, *unknown))`.
A claim with no citation contributes `0`. Supported claims are included
because a claim can be supported by two citations and still have had a third
rejected. With the count corrected, `_diagnose` no longer reaches
`CITATIONS_REJECTED` for this run and reports `INSUFFICIENT_EVIDENCE` —
*"evidence item(s) were extracted but none supported a citable claim"* —
which is what actually happened.

**Worth noting:** the bug inverted the meaning of the project's central
guarantee. It told a reader that the stored source text had failed to support
a quote, when the truth was that the model had declined to answer and offered
no quote. The synthesis step behaved *correctly* — faced with one unvetted
source that did not address the question, it said so instead of fabricating —
and the reporting layer then misattributed that honesty to a verification
failure. Three milestones of offline tests never caught it because no offline
case produced a claim with zero citations alongside a sub-question that had
retrieved evidence; the real web produced that combination on the first run.

**Also observed in the same run, not bugs.** Two SEC EDGAR filings returned
HTTP 403 and one primary source was a PDF, which the fetcher rejects by
documented design (`Unsupported content type 'application/pdf'`). Both are
real capability limits for a due-diligence agent whose preferred sources are
regulatory filings, and both are recorded in docs/deployment.md rather than
fixed here.

## F-019 · An OOM kill looked exactly like a slow run

**Milestone:** M10, found by investigating a production run that never finished

**Symptom.** `GET /research/run_b1c8bb35235c4275` returned, and kept
returning indefinitely:

```json
{"status": "planning", "plan": null, "report": null,
 "sources": null, "citations": [], "stages": [], "error": null, "cost": null}
```

while the Render logs for the *same* run showed `plan` passed (26.3 s),
`discover` passed (33.3 s) and `process` partial (14.0 s) — then stopped.
No `run_complete`, no `run_failed`, no `pipeline_failed`, no traceback.
Anthropic's logs showed HTTP 200 throughout. The API and the logs appeared
to contradict each other, and the PARTIAL process stage looked like the
cause.

**Cause — three defects, one visible symptom.**

*1. The pipeline was OOM-killed in stage 4, and this is the actual cause.*
`FastEmbedEmbedder.embed_documents` called `model.embed(texts)` without
`batch_size`, so fastembed applied its own default of **256**. Stage 4
embeds every chunk of every source in a single call, and transformer
attention allocates activations proportional to *batch × sequence²* — at
512-token chunks that asks ONNX for gigabytes at once. Measured in a 512 MB
container, which is the Render plan the service runs on:

| chunks | batch | peak RSS |
|---|---|---|
| 250 | 256 (shipped default) | **OOM-killed** — also OOM at a 1 GB ceiling |
| 250 | 16 | **OOM-killed** |
| 250 | 8 | 409 MB |
| 250 | 4 | 322 MB |
| 1000 | 4 | 317 MB |

A real run produces hundreds of chunks (one earlier live run produced 248),
so **every** real run was killed in stage 4. A `SIGKILL` from the kernel runs
no `except` block, no `finally`, and no log handler, which is precisely why
the logs end mid-pipeline with no error: there was no exception to catch.
The container restarted, `/health` went green, and the service looked fine.

*2. Nothing was persisted between the first status write and the end of the
run.* The route wrote `planning`, ran the orchestrator, and only then wrote
the plan, report, trace and final status. The orchestrator does advance
`result.status` through `discovering`, `processing` and so on, and passes it
to the stage observer — which **logged it and persisted nothing**. So the API
was not stale or reading a different database; it was reporting the last
thing anybody had written. That is the whole of the apparent contradiction,
and it is why a four-minute run was indistinguishable from a stuck one.

*3. Nothing reconciled a run whose process had died.* The route's
`try/except` guarantees a terminal state only while the process lives. After
a `SIGKILL` the run kept `planning` forever — and over HTTP a stranded run
and a working one are the same response.

**Fix.** Three changes, matching the three defects:

- `DEFAULT_BATCH_SIZE = 4` in `app/retrieval/embeddings.py`, passed
  explicitly to `embed`, configurable as `EMBEDDING_BATCH_SIZE`. Peak memory
  is now **flat in chunk count** — 341 MB at 250 chunks and 351 MB at 1000,
  measured through the real `run_retrieve_stage` in a 512 MB container. That
  flatness is the property that matters: chunk count follows source length
  and is unbounded, so memory must not scale with it.
- The stage observer now persists each stage's status, so `GET` reflects the
  stage actually reached. Writes are scheduled (the observer must not block
  the orchestrator) and drained before the terminal status, so a late
  progress write cannot land on top of the verdict.
- `RunRepository.fail_interrupted()`, called once from the lifespan on
  startup, fails every non-terminal run. This is the only thing that makes
  "a run always reaches a terminal state" true across a process boundary.
  It is correct *because* the service is single-process and single-instance:
  a non-terminal run at startup cannot be one this process is executing.
  Running a second instance, or uvicorn `--workers`, against one database
  would break that and fail live runs.

**Verified** by SIGKILLing a container holding a `planning` run and
restarting it: the run came back `failed` with *"Run was interrupted while
planning; the process did not survive to finish it."*

**Not the cause, despite appearances.** `stage_status=partial` on PROCESS is
a recorded gap, not a halt — the orchestrator only stops when
`processed.sources` is *empty*. PARTIAL meant "some sources fetched, some
failed", the run continued into stage 4 exactly as designed, and that is
where it died. A test now pins this so the next investigation does not spend
time on it. Also ruled out, and all of them wrong: stale repository objects,
two databases, a relative `database_path` resolving differently, a swallowed
exception, and the API reading in-memory state. The earlier migration
evidence already disproved the database theories — a run persisted before a
redeploy survived it, which only happens if the mounted disk and the path
are both right.

**Worth noting.** The memory figure previously recorded in `render.yaml` and
`docs/deployment.md` — "221 MB peak while fastembed loads its ONNX model and
embeds a batch" — was measured embedding a trivially small batch. It was a
real measurement of the wrong thing, and it is what made a 512 MB plan look
like it had headroom. The lesson is the same one as F-012 and F-017: the
measurement has to exercise the production path at production scale, or it
certifies something nobody is going to run. The ONNX session alone costs
~150 MB on top of an 86 MB interpreter, which no measurement here had
isolated before.

**Also fixed, same area.** `FASTEMBED_CACHE_PATH` was never set, so fastembed
fell back to `tempfile.gettempdir()` and re-downloaded the model from the
HuggingFace Hub on every cold start — while the comment on the persistent
disk in `render.yaml` claimed that disk existed to prevent exactly that. It
now points at `/app/data/fastembed`, verified writable by `appuser` on the
mount. A download inside stage 4 is also a failure mode: it makes the stage
depend on the Hub being reachable.
