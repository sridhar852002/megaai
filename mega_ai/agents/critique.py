"""Span-level critique of peer agent outputs."""

from __future__ import annotations

import json
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from mega_ai.core.context import (
    AgentMessage,
    CritiqueResult,
    CritiqueSpan,
    PolicyViolation,
    SharedContext,
)
from mega_ai.core.hashing import sha256_json
from mega_ai.core.structured_logging import StructuredLogEntry
from mega_ai.db.log_service import persist_structured_log
from mega_ai.llm.client import LLMClient
from mega_ai.orchestrator.context_budget import ContextBudgetManager, estimate_tokens


def _latest_agent_payload(context: SharedContext, agent_id: str) -> str | None:
    for msg in reversed(context.messages):
        if msg.agent_id == agent_id:
            return msg.content
    return None


async def run_critique(
    *,
    session: AsyncSession,
    job_uuid: uuid.UUID,
    context: SharedContext,
    llm: LLMClient,
    system_prompt: str,
    budget: ContextBudgetManager,
    turn_id: str,
    budget_cap: int,
    target_agents: list[str],
) -> CritiqueResult:
    targets = {agent: _latest_agent_payload(context, agent) for agent in target_agents if agent != "critique"}
    user = json.dumps({"targets": targets})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
    est = sum(estimate_tokens(m["content"]) for m in messages)
    if not budget.check_can_add(agent_id="critique", turn_id=turn_id, budget_tokens=budget_cap, addition_tokens=est):
        context.add_violation(
            PolicyViolation(
                kind="context_budget_overflow",
                agent_id="critique",
                detail="estimated prompt tokens exceed remaining budget before call",
            )
        )
    data, pt, ct = await llm.complete_json(messages=messages, agent_id="critique")
    budget.record_usage(agent_id="critique", turn_id=turn_id, tokens=pt + ct)
    snap = budget.snapshot_report(agent_id="critique", turn_id=turn_id, budget_tokens=budget_cap)
    if snap.used_tokens > budget_cap:
        context.add_violation(
            PolicyViolation(
                kind="context_budget_overflow",
                agent_id="critique",
                detail="post-call token usage exceeds declared cap",
            )
        )

    spans = [
        CritiqueSpan(
            target_agent=str(item.get("target_agent", "retrieval")),
            claim_span=str(item.get("claim_span", "")),
            confidence=float(item.get("confidence", 0.0)),
            disagree=bool(item.get("disagree", False)),
            suggested_correction=item.get("suggested_correction"),
        )
        for item in data.get("spans", [])
    ]
    result = CritiqueResult(
        critique_id=str(uuid.uuid4()),
        source_output_agent="multi",
        spans=spans,
        overall_notes=str(data.get("overall_notes", "")),
    )
    context.critique_results.append(result)
    context.append_message(
        AgentMessage(
            role="assistant",
            agent_id="critique",
            content=json.dumps(data),
            metadata={"structured": True},
        )
    )
    await persist_structured_log(
        session,
        StructuredLogEntry(
            job_id=str(job_uuid),
            agent_id="critique",
            event_type="agent_end",
            input_hash=sha256_json(messages),
            output_hash=sha256_json(data),
            token_count=pt + ct,
            payload={"spans": len(spans)},
        ),
    )
    return result
