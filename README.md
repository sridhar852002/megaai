# Mega AI — Multi-Agent Orchestration & Evaluation

Production-style FastAPI service with explicit orchestration, PostgreSQL + pgvector, Redis-backed Dramatiq workers, SSE streaming, and a hand-rolled evaluation harness.

## Quick start (Docker)

1. Copy `cp .env.example .env` and set `OPENAI_API_KEY` if using live models.
2. For fully local deterministic runs set `LLM_MODE=mock`.
3. Ensure `DATABASE_URL` uses `postgresql+asyncpg://` and `DATABASE_SYNC_URL` uses `postgresql+psycopg://` (see `.env.example` for compose hostnames).
4. `docker compose up --build`

Services: **api** (:8000), **worker**, **postgres** (pgvector), **redis**, **log-viewer** (:8080).

## Local Python

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export DATABASE_URL=postgresql+asyncpg://mega:mega@localhost:5432/mega_ai
export DATABASE_SYNC_URL=postgresql+psycopg://mega:mega@localhost:5432/mega_ai
export REDIS_URL=redis://localhost:6379/0
alembic upgrade head
uvicorn mega_ai.api.app:app --reload --port 8000
dramatiq mega_ai.workers.tasks
```

Run evaluation suite: `make eval` (requires DB).

## API (exactly five routes)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/query` | SSE stream of agent tokens, tool events, routing, budgets |
| GET | `/trace/{job_id}` | Full persisted trace + shared context snapshot |
| GET | `/evals/latest` | Latest eval aggregates + per-case scores |
| POST | `/prompt-review` | Approve/reject meta-agent prompt proposals |
| POST | `/reeval` | Enqueue targeted re-eval Dramatiq job on failed cases |

OpenAPI UI is **disabled** intentionally so only these five routes exist on the main app; this README is the contract reference.

### Example: SSE query

```bash
curl -N -H "Content-Type: application/json" \
  -d '{"query":"What is the capital of France?"}' \
  http://localhost:8000/query
```

Errors use `{"error":{"code","message","job_id"}}`.

## Architecture

See `docs/architecture.md` for ASCII diagram and data flow.

## Agents & boundaries

- **Orchestrator** — single routing brain; logs structured decisions with reason/conf confidence; invokes tools with explicit failure policies (never silently substitutes answers).
- **Decomposition** — emits JSON task DAG; violations logged on cycles.
- **Retrieval** — two-hop vector retrieval over pgvector; claims must cite ≥2 chunk IDs when corpus allows.
- **Critique** — span-level confidence/disagreement suggestions.
- **Synthesis** — merges outputs, emits sentence-level provenance.
- **Compression** — lossy on conversational filler only; structured payloads preserved verbatim.

Agents never call each other; they only read/write `SharedContext` mediated by the orchestrator.

## Evaluation

Fifteen cases in `mega_ai/evals/cases.py` (5 baseline / 5 ambiguous / 5 adversarial). Scoring is custom multi-dimensional with textual justifications in `mega_ai/evals/scoring.py`. Each run stores prompts, routing, tool calls, and scores for diff-friendly regression review.

### Self-improving loop (bounded)

After a failing suite, `maybe_propose_prompt_update` records a **pending** proposal. Humans approve via `/prompt-review`. Nothing auto-applies without approval. `/reeval` recomputes failed cases and persists `performance_delta` on the new eval row.

## AI collaboration disclosure

This repository was implemented with AI coding assistance (architecture, scaffolding, and iterative fixes). Human review is expected before production use; prompts and evaluator weights remain explicit in code for audit.

## Known limitations

- Token budgets use heuristic estimation when provider usage is partial; violations still surface explicitly.
- NL2SQL targets a bundled SQLite demo catalog (not Postgres) to keep the stack self-contained.
- Second-pass orchestrator routing always executes; it is a deliberate hook for dynamic replanning but adds latency.
- Streaming of intermediate JSON agents is chunked by whitespace rather than model tokenizer streams for non-synthesis phases.
- The `/query` handler keeps one DB transaction open for the entire SSE stream; shorten that window or offload trace writes for heavy traffic.

## Next steps

- Pluggable vector corpora with tenant isolation.
- Stronger sandboxing (gVisor/Firecracker) for code execution.
- Human-in-the-loop approvals surfaced through a minimal dashboard on top of the same five APIs.
