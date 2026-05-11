"""LLM-driven routing with logged justifications — not a hardcoded LangGraph."""

from __future__ import annotations

import json
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from mega_ai.core.context import AgentMessage, RoutingDecision, SharedContext, TokenUsage
from mega_ai.core.hashing import sha256_json
from mega_ai.core.structured_logging import StructuredLogEntry
from mega_ai.db.log_service import persist_structured_log
from mega_ai.llm.client import LLMClient


async def decide_next_route(
    *,
    session: AsyncSession,
    job_uuid: uuid.UUID,
    context: SharedContext,
    llm: LLMClient,
    orchestrator_prompt: str,
    prior_outputs: dict[str, str],
) -> RoutingDecision:
    router_payload = {
        "user_query": context.user_query,
        "prior_outputs": prior_outputs,
        "policy": (
            "Choose agents and explicit per-agent token budgets. "
            "You may reorder; do not assume a fixed pipeline if the query is narrow."
        ),
    }
    messages = [
        {"role": "system", "content": orchestrator_prompt},
        {"role": "user", "content": json.dumps(router_payload)},
    ]
    data, pt, ct = await llm.complete_json(messages=messages, agent_id="orchestrator_router")
    decision = RoutingDecision(
        decision_id=str(uuid.uuid4()),
        chosen_agents=list(data.get("chosen_agents", [])),
        order=list(data.get("order", [])),
        reason=str(data.get("reason", "")),
        confidence=float(data.get("confidence", 0.5)),
        context_budget_tokens={k: int(v) for k, v in dict(data.get("context_budget_tokens", {})).items()},
    )
    context.record_routing(decision)
    context.token_usage.append(
        TokenUsage(
            agent_id="orchestrator",
            turn_id=context.job_id,
            prompt_tokens=pt,
            completion_tokens=ct,
        )
    )
    await persist_structured_log(
        session,
        StructuredLogEntry(
            job_id=str(job_uuid),
            agent_id="orchestrator",
            event_type="routing",
            input_hash=sha256_json(messages),
            output_hash=sha256_json(decision.model_dump(mode="json")),
            latency_ms=None,
            token_count=pt + ct,
            policy_violations=[],
            payload={"decision_id": decision.decision_id},
        ),
    )
    context.append_message(
        AgentMessage(
            role="assistant",
            agent_id="orchestrator",
            content=f"routing:{decision.model_dump_json()}",
            metadata={"structured": True},
        )
    )
    return decision
