# ADR-004: SQLite + sqlite-vec locally; pgvector when deployed

**Status:** Accepted · **Date:** 2026-10-08

## Context
The architecturally important fact: **the corpus is per-question and
ephemeral.** Each research run discovers its own sources, indexes them, answers
the question, and discards the index. This is not a persistent knowledge base
accumulating documents over time.

That removes the usual justification for a dedicated vector database — there
is no large, long-lived, concurrently-queried index to serve.

## Decision
A `VectorStore` protocol with two real implementations:

- **sqlite-vec** — local development and every evaluation run. Zero infra, one
  file, and BM25 available for free through SQLite's built-in FTS5.
- **pgvector** — the deployed instance, on managed Postgres.

## Alternatives considered
- **Qdrant / Weaviate / Pinecone.** Rejected for V1: a service to run and pay
  for, serving an index that lives for the duration of one HTTP request.
- **FAISS or Chroma in-process.** FAISS gives no lexical search, so BM25 would
  need a second component; FTS5 in the same SQLite file is strictly simpler.
- **sqlite-vec only, everywhere.** Tempting and nearly chosen. Rejected because
  the deployed environment genuinely differs: a container with an ephemeral
  filesystem is a poor host for a file-backed store, and managed Postgres is
  already available on the deployment target.

## Consequences
**Accepted cost:** two implementations of one interface — the one place in this
project where that is accepted. Justified because the two environments really
differ, and because it forces the protocol to be a genuine seam rather than a
single-implementation abstraction. Both are covered by the same test suite,
parameterised over the backend.

**Revisit when:** a shared or persistent corpus is introduced, at which point
`docs/scaling.md` covers the path to a dedicated vector service.
