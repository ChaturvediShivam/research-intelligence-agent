# Deployment

**Status summary, stated before any detail, because the distinction matters:**

| Item | Status |
|---|---|
| Production `Dockerfile` | ✅ **VERIFIED** — builds and runs |
| Image build | ✅ **VERIFIED** — Podman 6.0.2, `localhost/ria:m10` |
| Container runtime | ✅ **VERIFIED** — starts clean, non-root, honours `$PORT` |
| Health / readiness | ✅ **VERIFIED** — HTTP 200 from a running container |
| Smoke test | ✅ **VERIFIED** — 4/4 checks against the live container |
| Memory against the 512 MB plan | ✅ **VERIFIED** — 351 MB peak through the real stage 4, measured under a 512 MB cap, flat in chunk count (was OOM-killed before F-019) |
| `render.yaml` | 📄 **DOCUMENTED** — validated and deployment-ready, never applied |
| Render deployment | 🛑 **BLOCKED** — needs a GitHub repository and a Render login, neither available here |
| Live public URL | ❌ **DOES NOT EXIST** |

Nothing below claims a running public service. There isn't one.

**One correction worth recording:** `render.yaml` originally declared
`runtime: image`. That value means "pull a prebuilt image from a registry",
requires an `image:` block and ignores `dockerfilePath` entirely — so the
blueprint would have been rejected before anything was built. A blueprint
that is never applied is never contradicted, which is precisely how that
survived into a committed file. It is now `runtime: docker`.

## Architecture

Two entry points, and only one is meant to be public:

```
                    ┌──────────────────────────────┐
  public HTTP  ───►  │ app/main.py  (FastAPI)       │  ← deployed
                    │  /health  /ready  /research  │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │ app/pipeline/orchestrator.py │
                    └──────────────┬───────────────┘
                                   │
  local MCP client ─(stdio)─►  app/mcp/server.py ──┘
```

`app/mcp/server.py` is **stdio only** and is deliberately not deployed. An MCP
client runs it as a local subprocess; exposing it over HTTP would mean
inventing a transport and an auth story that nothing has asked for, so M10
deploys the HTTP service that already exists.

## Local development

```bash
uv sync --extra retrieval --extra mcp
cp .env.example .env            # then add your key
uv run uvicorn app.main:app --reload
uv run pytest                   # 898 tests, offline
```

## Docker build — verified

```bash
podman build -t ria:m10 -f Dockerfile .      # or: docker build -t ria:m10 .
```

Multi-stage: dependencies resolve in the builder and only the virtualenv and
`app/` ship. The runtime image installs no compiler and runs as uid 10001.

Verified with Podman 6.0.2 on macOS/arm64. The Dockerfile is standard and
contains nothing Podman-specific, but **it has not been built with Docker
itself** — Docker is not installed in this environment.

One Podman-specific note: Podman's default OCI image format ignores
`HEALTHCHECK` and warns about it at build time. Docker and Render honour it.
Not a defect in the Dockerfile.

## Docker run — verified

```bash
podman run -d --name ria -e PORT=8000 -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-... ria:m10
```

The container binds `$PORT`, falling back to 8000. That matters for Render,
which injects `PORT` and expects the process to use it — see the bug note
below.

Verified behaviour, from a real run:

```
$ podman run -d --name ria-m10 -e PORT=9137 -p 9137:9137 ria:m10
$ curl -s http://127.0.0.1:9137/health
{"status":"ok","version":"0.1.0"}                                    # HTTP 200

$ curl -s http://127.0.0.1:9137/ready
{"ready":false,"environment":"production","vector_backend":"sqlite",
 "anthropic_key_configured":false,"missing":["ANTHROPIC_API_KEY"]}    # HTTP 200

$ podman exec ria-m10 id
uid=10001(appuser) gid=10001(appuser) groups=10001(appuser)
```

With a key supplied, `/ready` returns `{"ready":true, ..., "missing":[]}`.
Startup logs are JSON, carry no secret, and shutdown is clean (exit code 0).

## Environment variables

Every variable below exists in `app/core/config.py`. **None is required for
the process to start** — the service starts and serves `/health` without a
key and reports itself not-ready, which is the correct behaviour for a
liveness probe.

| Variable | Required for research | Default | Notes |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | **yes** | – | The only secret. Set it in Render's dashboard, never in a file. |
| `SEC_CONTACT_EMAIL` | no, but SEC sources fail without it | – | A contact address sec.gov requires automated clients to declare, or it returns **403**. Sent only to `sec.gov` hosts. See below. |
| `PORT` | no | `8000` | Injected by Render; the container binds it. |
| `ENVIRONMENT` | no | `local` | Set to `production`. |
| `LOG_JSON` | no | `false` | Set to `true` in production. |
| `LOG_LEVEL` | no | `INFO` | |
| `MAX_COST_USD_PER_RUN` | no | `2.0` | Per-run ceiling, enforced in the orchestrator. |
| `MAX_SOURCES_PER_RUN` | no | `12` | Bounds cost and latency. |
| `PLANNING_MODEL` | no | `claude-opus-5-5` | |
| `EXTRACTION_MODEL` | no | `claude-haiku-4-5` | |
| `SYNTHESIS_MODEL` | no | `claude-opus-5-5` | |
| `RETRIEVAL_TOP_K` | no | `12` | |
| `CHUNK_TOKENS` / `CHUNK_OVERLAP_TOKENS` | no | `512` / `64` | |
| `EMBEDDING_MODEL` | no | `BAAI/bge-small-en-v1.5` | Downloaded on first use. |
| `VECTOR_BACKEND` | no | `sqlite` | `postgres` additionally needs `POSTGRES_DSN`. |
| `POSTGRES_DSN` | only if `VECTOR_BACKEND=postgres` | – | Secret. |
| `DATABASE_PATH` | no | `data/runs.db` | Put on the mounted disk. |
| `FETCH_TIMEOUT_SECONDS` | no | `20.0` | |
| `ALLOWED_SOURCE_DOMAINS` / `BLOCKED_SOURCE_DOMAINS` | no | empty | Domain policy. |

## Render deployment — NOT VERIFIED

`render.yaml` is a complete blueprint: Docker runtime, `/health` as the health
check path, `ANTHROPIC_API_KEY` as `sync: false` so Render prompts for it
rather than reading it from the repository, and a 1 GB disk at `/app/data` so
the SQLite index and the embedding-model cache survive a restart.

**It has never been applied.** The blocking facts, checked rather than
assumed:

| Prerequisite | State |
|---|---|
| GitHub push authentication | ✅ works — `ssh -T git@github.com` authenticates as `ChaturvediShivam` |
| Remote repository | ❌ `ChaturvediShivam/research-intelligence-agent` does not exist (`git ls-remote` → *Repository not found*) |
| A way to create it from here | ❌ no `gh` CLI, no GitHub token in the environment |
| Render CLI | ❌ not installed |
| Render credentials / API key | ❌ not present in the environment |

Render deploys from a connected Git repository, so with no remote repository
there is nothing for it to deploy from. Rather than simulate a deployment,
here is exactly what remains — **two manual actions**, both requiring a
browser login this environment does not have:

1. **Create the repository** (manual). At <https://github.com/new>, name it
   `research-intelligence-agent`. Then, locally:

   ```bash
   git push -u origin main    # the `origin` remote is already configured
   ```

   SSH authentication is already working, so the push itself needs no setup.

2. **Create the Blueprint** (manual). In Render, **New → Blueprint**, select
   the repository; Render reads `render.yaml`. When prompted, paste
   `ANTHROPIC_API_KEY` — the only value it asks for, because everything else
   is in the blueprint.

Then wait for the first build. It is slow: the image installs `fastembed`,
and the embedding model is downloaded on first use (~14 s, measured).

3. Confirm the service is live:

```bash
curl -s https://<service>.onrender.com/health
curl -s https://<service>.onrender.com/ready
uv run python scripts/smoke_test.py https://<service>.onrender.com
```

`/ready` should report `"ready": true`. If it reports
`"missing": ["ANTHROPIC_API_KEY"]`, the key did not reach the service.

A note on the `starter` plan: it idles after inactivity, so the first request
after an idle period will be slow, and research runs are long-lived requests.

### Memory

Measured under `podman run --memory=512m` — the same 512 MB the `starter`
plan (`0.5c-512mb`) provides — driving the **real `run_retrieve_stage`**:

| | RSS |
|---|---|
| Interpreter and imports | 86 MB |
| ONNX session loaded | 237 MB |
| Peak, 250 chunks embedded | 341 MB |
| Peak, 1000 chunks embedded | 351 MB |

Peak is **flat in chunk count**, which is the property that matters: chunk
count follows source length and is unbounded, so memory must not scale with
it. That flatness comes from `EMBEDDING_BATCH_SIZE`, and it is the whole
reason the plan fits.

**This file previously claimed 221 MB peak and "307 MB worst case". That was
wrong, and wrong in the direction that hid a production outage.** The figure
was taken embedding a 32-passage batch, which no real run ever does: stage 4
embeds every chunk of every source in one call. At the batch size that
shipped, 250 chunks OOM-killed a 512 MB container — and OOM-killed a 1 GB
one. Every real research run died in stage 4 with no error, because a
`SIGKILL` raises nothing. See F-019 in
[`docs/failure-analysis.md`](failure-analysis.md).

Raising the instance size is therefore two edits, not one: the plan **and**
`EMBEDDING_BATCH_SIZE`. Leaving the batch size alone is always safe; raising
it without headroom is what caused the outage.

Latency is the cost of that headroom. On the measurement host, 1000 chunks
took 86 s to embed. Render's `starter` plan provides 0.5 CPU, so expect
materially longer there, and stage 4 has no timeout.

### Model cache

`FASTEMBED_CACHE_PATH=/app/data/fastembed` puts the embedding model on the
persistent disk. Without it fastembed falls back to
`tempfile.gettempdir()` and re-downloads the model from the HuggingFace Hub
on every cold start — which also makes stage 4 fail whenever the Hub is
unreachable. Verified writable by `appuser` (uid 10001) on the mount.

### Interrupted runs

A research run executes in a `BackgroundTask` inside the API process, so a
deploy, restart or OOM kill takes it with it and runs no exception handler.
On startup the service fails every run left in a non-terminal state:

```
{"event": "run_interrupted", "run_id": "run_...", "status": "failed"}
{"event": "application_start", "interrupted_runs_failed": 1, ...}
```

The run then reports `failed` with *"Run was interrupted while planning; the
process did not survive to finish it."* — naming the stage it died at.

This is correct **only** because the service is single-process and
single-instance. Do not set uvicorn `--workers`, and do not scale this
service past one instance, without replacing the reaper with something that
can tell a dead process's runs from a live one's — otherwise startup will
fail runs that are still executing elsewhere.

## SEC EDGAR access

`sec.gov` refuses automated requests that do not declare a contact address,
returning HTTP 403. Production run `run_67498d61c5794183` hit this on two
NVIDIA 10-K filings — the two best primary sources discovery had found.

Set `SEC_CONTACT_EMAIL` to a contactable address. `app/tools/fetch.py` then
sends the format sec.gov documents:

```
User-Agent: ResearchIntelligenceAgent/0.1 <SEC_CONTACT_EMAIL>
Accept-Encoding: gzip, deflate
```

Three properties worth knowing:

- **It is sent only to `sec.gov` and its subdomains.** The header is set
  per-request, not on the shared client, so the address is never disclosed to
  other sites the agent fetches. Host matching reuses the same subdomain-aware
  comparison the domain policy uses, so `sec.gov.example.com` and
  `notsec.gov` do not qualify.
- **It is recomputed on every redirect hop.** Entering `sec.gov` adds it;
  being redirected off `sec.gov` drops it.
- **Leaving it unset changes nothing else.** The fetch still happens, SEC
  still refuses it, and a warning naming the variable is logged so the 403 is
  distinguishable from a genuine access restriction. A 403 is never treated
  as success either way.

In Render the variable is declared `sync: false`, so it is prompted for
rather than committed. SEC also caps automated access at 10 requests/second;
this service fetches at most `MAX_SOURCES_PER_RUN` documents per run, well
inside that.

PDFs remain unsupported, so a filing served as `application/pdf` is still
refused with `Unsupported content type`. That is a separate limitation.

## Production smoke test

```bash
uv run python scripts/smoke_test.py https://<service>.onrender.com
```

Four checks, all free — no research endpoint is touched, because proving the
HTTP server works should not cost money:

1. `/health` returns 200 with JSON and `status: "ok"`
2. `/ready` returns 200 and reports key configuration **as a boolean**
3. an unknown route returns 404 rather than a 500 with a stack trace
4. no response carries a secret value or a traceback

Exit code 0 on success, so it works as a post-deploy gate.

Verified against the local container: **4/4 passed.**

## Security considerations

- **No secret is in the image.** The Dockerfile copies only
  `pyproject.toml`, `uv.lock`, `README.md` and `app/` — never `.`. Verified:
  `find / -name .env` inside the running container returns nothing.
- `.dockerignore` excludes `.env`, `.env.*`, `.git`, caches, `tests/`,
  `evals/` and local data as defence in depth and to shrink the build context.
- `.env` is gitignored and untracked. Verified with `git check-ignore`.
- **Runs as non-root**, uid 10001.
- `/ready` reports key presence as a **boolean**, never a value — it is a
  public endpoint.
- Logs are JSON in production and pass through the two-layer redaction from
  M0. Verified: a dummy key passed as an env var appears zero times in logs.
- The M9 controls are untouched by this milestone: SSRF guard, untrusted-
  content boundary, injection telemetry, citation verification.

## Limitations

- **No live public deployment exists.** Everything Render-related is
  configuration and instructions.
- Built and run with **Podman, not Docker**. The Dockerfile is standard, but
  Docker itself was unavailable here.
- `HEALTHCHECK` could not be exercised, because Podman's OCI format ignores it.
- The smoke test deliberately never calls a research endpoint, so **no
  end-to-end research run has been verified through a deployed HTTP
  interface.** The pipeline itself is live-verified in M5; the gap is
  specifically "via a deployed container".
- SQLite on a single mounted disk means one instance. Horizontal scaling would
  need `VECTOR_BACKEND=postgres`, which exists in config but is not exercised
  in a deployment.
- A research request is a long-lived HTTP request. There is no job queue, so a
  platform request timeout will cut off a long run.
