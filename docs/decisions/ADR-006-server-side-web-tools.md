# ADR-006: Anthropic server-side web search/fetch over a dedicated search vendor

**Status:** Accepted · **Date:** 2026-10-08

## Context
Stage 2 needs to discover sources on the open web, and stage 3 needs to fetch
them. The usual approach is a search API (Tavily, Exa, Brave, SerpAPI) plus an
HTTP fetcher.

## Decision
Use Anthropic's server-side tools `web_search_20260209` and
`web_fetch_20260209`, declared in `tools`, behind a `SourceProvider` protocol
in `app/tools/search.py`.

## Alternatives considered
- **Tavily / Exa.** Good research-oriented APIs, but another vendor, another
  key, another bill, and no citation integration.
- **Raw Google/Bing + own scraper.** More control, considerably more
  maintenance (rate limits, parsing, robots handling) for no demonstration
  value this project needs.

## Consequences
**Accepted cost:** source discovery is coupled to one provider's index and its
ranking, and the search runs server-side where this application cannot inspect
intermediate ranking. The `SourceProvider` protocol keeps a vendor swap to one
file if measured recall proves insufficient.

**Security benefit worth stating:** both tools accept `allowed_domains` /
`blocked_domains`, which turns source selection into an enforceable control
surface rather than a prompt instruction. That composes with the SSRF guard in
`app/core/security.py` for any URL the user supplies directly.

**Note:** `web_fetch` only fetches URLs already present in the conversation,
which is itself a useful constraint against a model inventing a target.

**Revisit when:** measured source coverage on the golden set is the limiting
factor.


---

## Amendment, 2026-10-08 (M2 implementation)

**Discovery uses the server-side tool as decided. Fetching does not.**

Implementing stage 3 surfaced a conflict between this ADR and
[ADR-002](ADR-002-deterministic-citation-verification.md). Deterministic
citation verification requires this service to hold the exact canonical text
that citation offsets index into, byte for byte. A server-side `web_fetch`
hands a document to the model; it does not hand this application a string it
can slice at `[start_char:end_char]` later and compare.

The architecture document already implied the resolution: stage 3 is specified
as "fetch → SSRF guard → extract → sanitise → chunk", and an SSRF guard only
has meaning if the outbound request is ours to refuse.

**Resolved as:**

- **`web_search_20260209`** for discovery — the vendor, key and ranking
  argument in this ADR stands, as does `allowed_domains` / `blocked_domains`
  as an enforced control.
- **Local `httpx` fetch** for retrieval of text that will be cited, in
  `app/tools/fetch.py`, behind `app/core/security.py`.

**Consequences of the amendment.** Fetching locally means owning timeouts,
size caps, content-type policy, redirect re-validation and encoding — roughly
90 lines in `fetch.py`. In exchange: character offsets are ours, citation
verification is possible at all, and the SSRF guard is enforceable. Both
halves are covered by `tests/unit/test_fetch.py` and
`tests/security/test_ssrf.py`.
