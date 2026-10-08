# ADR-010: Intelligence-analysis methodology encoded in the domain model

**Status:** Accepted · **Date:** 2026-10-08

## Context
A generic RAG system returns an answer plus citations. Professional research
practice — competitive intelligence, due diligence, AML/KYC — does not stop
there. It grades sources by provenance, requires corroboration across
independent sources before treating a claim as established, states confidence
with its basis, and names what could not be determined.

A system that skips those steps produces output that reads authoritative and
cannot be audited.

## Decision
Encode four analyst primitives in the schema, each **computed by code from
verifiable facts** rather than asserted by a model:

| Primitive | Where | How it is derived |
|---|---|---|
| `SourceRef.credibility` | `app/schemas/evidence.py` | Rule-based tier: primary/regulatory/filing → established secondary → unvetted/UGC |
| `Claim.corroboration` | `app/pipeline/validate.py` | Count of **independent** supporting sources (distinct registrable domain *and* distinct publisher) |
| `Claim.confidence` | `app/pipeline/validate.py` | Derived from corroboration × credibility × citation-verification result. Ordinal (`high`/`moderate`/`low`/`UNKNOWN`), never a percentage |
| `Report.information_gaps` | `app/pipeline/assess.py` | Sub-questions from stage 1 with no verified supporting evidence |

## Alternatives considered
- **Ask the model to self-report confidence.** Rejected: an LLM's stated
  confidence is not calibrated, and it is precisely the thing under audit.
- **A numeric confidence score (0–100).** Rejected: fake precision. An ordinal
  scale with a printed basis is honest; "73% confident" is not.
- **Omit all of it and return answer + citations.** Rejected: that is the
  generic design this project exists to improve on.

## Consequences
**Accepted cost:** more schema surface and more deterministic logic to test.
Independence detection is a heuristic — two outlets syndicating one wire story
can register as two sources. That limitation is documented in the README and
is a named candidate for improvement via the failure analysis.

**Benefit:** the output is auditable. A reader can see why a claim is rated as
it is, and disagree with a specific rule rather than with a vibe.

**Revisit when:** syndication-driven false corroboration shows up in measured
results.
