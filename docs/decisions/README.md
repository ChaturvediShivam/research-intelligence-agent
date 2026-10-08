# Architecture Decision Records

One file per decision that was not obvious. Each records the decision, the
alternatives actually considered, and the cost accepted — because the cost is
the part that makes a decision real.

| ADR | Decision | Status |
|---|---|---|
| [001](ADR-001-native-citations.md) | Anthropic native document citations as the grounding backbone | Accepted |
| [002](ADR-002-deterministic-citation-verification.md) | Citation correctness verified in code, not by an LLM judge | Accepted |
| [003](ADR-003-local-embeddings.md) | Local ONNX embeddings rather than a hosted embedding API | Accepted |
| [004](ADR-004-vector-store.md) | SQLite + sqlite-vec locally; pgvector when deployed | Accepted |
| [005](ADR-005-no-agent-framework.md) | No agent framework | Accepted |
| [006](ADR-006-server-side-web-tools.md) | Anthropic server-side web search/fetch over a search vendor | Accepted |
| [007](ADR-007-model-routing.md) | Model routing by pipeline stage | Accepted |
| [008](ADR-008-prompt-caching.md) | Prompt caching strategy and its verification | Accepted |
| [009](ADR-009-deterministic-pipeline.md) | Deterministic pipeline with exactly one model-driven step | Accepted |
| [010](ADR-010-intelligence-methodology.md) | Intelligence-analysis methodology encoded in the domain model | Accepted |

Template: [ADR-000-template.md](ADR-000-template.md)
