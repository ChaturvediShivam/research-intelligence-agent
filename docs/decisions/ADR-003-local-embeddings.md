# ADR-003: Local ONNX embeddings rather than a hosted embedding API

**Status:** Accepted · **Date:** 2026-10-08

## Context
RAG needs an embedding model. Anthropic does not offer an embeddings endpoint,
so this is a separate vendor decision. The evaluation harness re-embeds the
corpus on every run, and the eval is meant to run often — cost per eval run
directly determines how habitual measurement is.

## Decision
Use `fastembed` with `BAAI/bge-small-en-v1.5` (ONNX, runs locally on arm64 and
x86) behind an `Embedder` protocol in `app/retrieval/embeddings.py`.

## Alternatives considered
- **Voyage AI** (Anthropic's recommended pairing). Better retrieval quality,
  but a per-call cost on every eval run and another key to manage. Kept as the
  documented upgrade path behind the same protocol.
- **OpenAI `text-embedding-3-small`.** Introduces a second LLM vendor for no
  architectural gain.
- **sentence-transformers.** Pulls in torch — a ~2 GB dependency for a model
  ONNX runs in megabytes.

## Consequences
**Accepted cost:** retrieval quality is likely below a frontier hosted
embedder. This is measurable rather than assumed: M7 includes an optional
measured fastembed-vs-Voyage comparison on the same golden set, and if
retrieval is shown to be the binding constraint the protocol makes the swap a
one-file change.

**Benefit:** $0 marginal cost per eval run and no network dependency in tests.

**Revisit when:** measured `recall@k` is the limiting factor on report quality.
