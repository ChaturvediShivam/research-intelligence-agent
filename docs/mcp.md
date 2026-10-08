# MCP interface

An external MCP client can discover and invoke this system's four approved
tools (architecture §8). The MCP layer is an adapter: it translates protocol,
validates arguments and shapes errors. It holds no research logic of its own.

```
MCP client --(JSON-RPC)--> app/mcp/server.py --> app/tools/registry.py --> existing pipeline code
```

## Tools

| Tool | Does | Billable |
|---|---|---|
| `search_sources` | Finds candidate sources for a query. Returns URLs, titles, domains. Does not fetch them. | yes |
| `fetch_and_index` | Fetches one URL, extracts and chunks its text, returns provenance. | no |
| `retrieve_evidence` | Fetches and indexes one URL, then returns the passages most relevant to a query, with offsets. | no |
| `run_research` | Runs the full pipeline and returns the M6 report. | yes |

`billable` is advertised in each tool's MCP `meta`, so a client can see which
calls cost money before making one.

### Input contracts

Schemas are generated from the Pydantic models in `app/tools/registry.py`, so
the schema a client discovers is the schema that validates its call — they
cannot drift. All four set `additionalProperties: false`.

| Tool | Required | Optional |
|---|---|---|
| `search_sources` | `query` (3–500 chars) | `max_results` (1–20, default 6) |
| `fetch_and_index` | `url` (8–2000 chars) | — |
| `retrieve_evidence` | `url`, `query` | `k` (1–25, default 5) |
| `run_research` | `question` (12–2000 chars) | `max_sources` (1–12, default 5) |

### Output contracts

Every result is structured JSON. `fetch_and_index` and `retrieve_evidence`
carry the provenance a caller needs to verify a later citation: `source_id`,
`content_hash`, credibility tier, redirect chain, and character offsets into
the canonical source text.

`run_research` returns the `ResearchReport` **verbatim** — not a flattened
summary — so sub-question ANSWERED/PARTIAL/UNKNOWN status, information gaps
with their causes, derived confidence and verified citations all survive the
boundary. Alongside it are the run's measured totals (sources fetched and
failed, citations verified and rejected, cost, duration) and the per-stage
outcome map.

### Errors

Failures are structured results, not transport exceptions, so a client can
branch on them:

```json
{
  "status": "error",
  "code": "invalid_input",
  "message": "Arguments do not match the schema for fetch_and_index.",
  "details": {"errors": [{"field": "url", "problem": "String should have at least 8 characters"}]}
}
```

`code` is the application's own error vocabulary (`invalid_input`,
`unsafe_url`, `upstream_error`, `rate_limited`, `cost_ceiling_exceeded`,
`pipeline_stage_error`, `internal_error`). An unexpected exception becomes
`internal_error` with the exception type — never a traceback.

## Safety boundaries

- **Validation happens before any work.** Arguments are parsed against the
  tool's model first, so a malformed call cannot reach the network or the
  model. This matters for the two billable tools: an invalid call costs
  nothing. Tested.
- **The SSRF guard applies**, because MCP uses the same fetcher. Private,
  loopback, link-local and CGNAT addresses and non-HTTP schemes are refused
  at the boundary. Tested, including `169.254.169.254` and `file://`.
- **No stage is individually exposed.** There is no `synthesise` or `extract`
  tool, so an MCP caller cannot assemble prose from unverified input. The only
  research entry point is `run_research`, which goes through
  `ResearchOrchestrator`. A regression test asserts the orchestrator is
  actually called and that every stage ran.
- **Returned source text is untrusted data.** The server's `instructions`
  say so to the client. Internally the untrusted-content fence still applies.
- **Cost ceilings survive.** `max_sources` is bounded by the schema at 12, so
  an MCP caller cannot raise it past the ceiling.

## Connecting a client

Run over stdio, which is what a local MCP client expects:

```bash
uv run python -m app.mcp.server
```

Claude Desktop / Claude Code configuration:

```json
{
  "mcpServers": {
    "research-intelligence-agent": {
      "command": "uv",
      "args": ["run", "python", "-m", "app.mcp.server"],
      "cwd": "/absolute/path/to/research-intelligence-agent",
      "env": {"ANTHROPIC_API_KEY": "sk-ant-..."}
    }
  }
}
```

The key is read from the environment by `Settings`, server-side. It is never
accepted as a tool argument.

**Tested clients:** `fastmcp.Client` over FastMCP's in-memory transport, which
is a real MCP client speaking the real protocol (`initialize`, `tools/list`,
`tools/call`). The stdio entry point above has not been exercised against
Claude Desktop, and no other client has been tested — the configuration block
is the documented shape, not a verified integration.

## Local development

```bash
uv run pytest tests/integration/test_mcp_server.py -v   # 36 tests, offline, no API calls
```

The tests stand in only the three outbound edges — search, HTTP (respx) and
the model (`ScriptedLLM`). Everything between is production code, so a test
that passes means the real handler worked.

One fixture caveat worth knowing: `FakeResolver` answers for a literal IP as
though it were an unknown hostname, which would mask an SSRF block. The SSRF
tests therefore use the real resolver (a literal IP needs no DNS anyway).

## Limitations

- `retrieve_evidence` is stateless: it re-fetches and re-indexes on every
  call. Holding an index between calls would need a session store, which
  nothing currently requires.
- No authentication. The server is intended for a local, trusted client over
  stdio. Exposing it over HTTP would need auth first.
- No rate limiting at the MCP layer; the existing cost ceiling is the only
  spend bound.
- `search_sources` and `run_research` spend money when called. The `billable`
  flag advertises this but nothing enforces a budget per client.
