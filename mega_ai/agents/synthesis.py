"""Synthesis merges agent outputs with an explicit provenance map."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from mega_ai.core.context import AgentMessage, PolicyViolation, ProvenanceEntry, SharedContext
from mega_ai.core.hashing import sha256_json
from mega_ai.core.structured_logging import StructuredLogEntry
from mega_ai.db.log_service import persist_structured_log
from mega_ai.llm.client import LLMClient
from mega_ai.orchestrator.context_budget import ContextBudgetManager, estimate_tokens


async def prepare_synthesis(
    *,
    session: AsyncSession,
    job_uuid: uuid.UUID,
    context: SharedContext,
    llm: LLMClient,
    system_prompt: str,
    budget: ContextBudgetManager,
    turn_id: str,
    budget_cap: int,
) -> dict[str, Any]:
    bundle = {
        "user_query": context.user_query,
        "decomposition": context.decomposition_graph,
        "retrieval": context.retrieval_scratchpad,
        "critiques": [c.model_dump() for c in context.critique_results],
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(bundle)},
    ]
    est = sum(estimate_tokens(m["content"]) for m in messages)
    if not budget.check_can_add(agent_id="synthesis", turn_id=turn_id, budget_tokens=budget_cap, addition_tokens=est):
        context.add_violation(
            PolicyViolation(
                kind="context_budget_overflow",
                agent_id="synthesis",
                detail="estimated prompt tokens exceed remaining budget before call",
            )
        )
    data, pt, ct = await llm.complete_json(messages=messages, agent_id="synthesis")
    budget.record_usage(agent_id="synthesis", turn_id=turn_id, tokens=pt + ct)
    snap = budget.snapshot_report(agent_id="synthesis", turn_id=turn_id, budget_tokens=budget_cap)
    if snap.used_tokens > budget_cap:
        context.add_violation(
            PolicyViolation(
                kind="context_budget_overflow",
                agent_id="synthesis",
                detail="post-call token usage exceeds declared cap",
            )
        )
    for row in data.get("provenance", []):
        context.provenance.append(
            ProvenanceEntry(
                sentence=str(row.get("sentence", "")),
                source_agent=str(row.get("source_agent", "")),
                source_chunk_id=row.get("source_chunk_id"),
                source_tool_call_id=row.get("source_tool_call_id"),
            )
        )
    context.append_message(
        AgentMessage(
            role="assistant",
            agent_id="synthesis",
            content=json.dumps(data),
            metadata={"structured": True},
        )
    )
    await persist_structured_log(
        session,
        StructuredLogEntry(
            job_id=str(job_uuid),
            agent_id="synthesis",
            event_type="agent_end",
            input_hash=sha256_json(messages),
            output_hash=sha256_json(data),
            token_count=pt + ct,
            payload={"phase": "synthesis_prepare"},
        ),
    )
    return data


async def stream_answer_text(answer: str) -> AsyncIterator[str]:
    """Character streaming keeps SSE dense while remaining deterministic for mock mode."""
    for ch in answer:
        yield ch
        await asyncio.sleep(0)
