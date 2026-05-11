"""Lossy compression of conversational filler while preserving structured facts."""

from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from mega_ai.core.context import AgentMessage, PolicyViolation, SharedContext
from mega_ai.core.hashing import sha256_json
from mega_ai.core.structured_logging import StructuredLogEntry
from mega_ai.db.log_service import persist_structured_log
from mega_ai.llm.client import LLMClient
from mega_ai.orchestrator.context_budget import ContextBudgetManager, estimate_tokens


async def run_compression(
    *,
    session: AsyncSession,
    job_uuid: uuid.UUID,
    context: SharedContext,
    llm: LLMClient,
    system_prompt: str,
    budget: ContextBudgetManager,
    turn_id: str,
    budget_cap: int,
    conversational_text: str,
    structured_facts: dict[str, Any],
) -> str:
    payload = {
        "structured_facts": structured_facts,
        "conversational_text": conversational_text,
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload)},
    ]
    est = sum(estimate_tokens(m["content"]) for m in messages)
    if not budget.check_can_add(agent_id="compression", turn_id=turn_id, budget_tokens=budget_cap, addition_tokens=est):
        context.add_violation(
            PolicyViolation(
                kind="context_budget_overflow",
                agent_id="compression",
                detail="estimated prompt tokens exceed remaining budget before call",
            )
        )
    data, pt, ct = await llm.complete_json(messages=messages, agent_id="compression")
    budget.record_usage(agent_id="compression", turn_id=turn_id, tokens=pt + ct)
    snap = budget.snapshot_report(agent_id="compression", turn_id=turn_id, budget_tokens=budget_cap)
    if snap.used_tokens > budget_cap:
        context.add_violation(
            PolicyViolation(
                kind="context_budget_overflow",
                agent_id="compression",
                detail="post-call token usage exceeds declared cap",
            )
        )
    compressed = str(data.get("compressed_filler", ""))
    context.append_message(
        AgentMessage(
            role="assistant",
            agent_id="compression",
            content=json.dumps({"compressed_filler": compressed, "structured_preserved": True}),
            metadata={"structured": True},
        )
    )
    await persist_structured_log(
        session,
        StructuredLogEntry(
            job_id=str(job_uuid),
            agent_id="compression",
            event_type="compression",
            input_hash=sha256_json(messages),
            output_hash=sha256_json(data),
            token_count=pt + ct,
            payload={},
        ),
    )
    return compressed
