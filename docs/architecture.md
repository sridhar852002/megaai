# Mega AI Architecture (text diagram)

```text
                +------------------+
                |     Client       |
                +--------+---------+
                         | POST /query (SSE)
                         v
                +------------------+
                |   FastAPI API    |
                |  (5 routes only)|
                +--------+---------+
                         |
         SharedContext   |   LiteLLM (OpenAI / Ollama / mock)
         +---------------v-----------------------------+
         |            Orchestrator                      |
         |  - routing JSON + logged justification       |
         |  - tool gateway (retries + failure codes)    |
         |  - context budget + compression trigger      |
         +----+------+-------+-------+---------+--------+
              |      |       |       |         |
              v      v       v       v         v
         Decomp  Retriev  Critique  Synth   Compress
              \      |_______|_______/
               \            |
                v           v
            pgvector RAG   PostgreSQL (jobs, evals, logs)
                ^           ^
                |           |
        +-------+-----------+-----------+
        | Dramatiq worker (Redis broker)|
        |  - targeted re-eval           |
        |  - optional full eval actor   |
        +-------------------------------+

Log viewer (separate FastAPI) ---> reads structured_logs (read-only SQL)
```

## Trace storage

Each job stores:

- orchestration routing decisions with timestamps
- tool call records including retry index and acceptance bit
- full `SharedContext` snapshot for reproducibility
- critique spans and synthesis provenance map

## Evaluation artifacts

`eval_runs` aggregate dimension scores; `eval_case_results` retain per-case payloads; `prompt_rewrite_proposals` retain meta-agent diffs pending approval.
