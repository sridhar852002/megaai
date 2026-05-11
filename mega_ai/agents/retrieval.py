"""Multi-hop retrieval agent — plans two hops and cites chunk IDs per claim."""

from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from mega_ai.core.context import AgentMessage, CitationRecord, PolicyViolation, SharedContext
from mega_ai.core.hashing import sha256_json
from mega_ai.core.structured_logging import StructuredLogEntry
from mega_ai.db.log_service import persist_structured_log
from mega_ai.llm.client import LLMClient
from mega_ai.orchestrator.context_budget import ContextBudgetManager, estimate_tokens
from mega_ai.rag.service import multi_hop_retrieve


async def run_retrieval(
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
    user = json.dumps(
        {
            "user_query": context.user_query,
            "decomposition": context.decomposition_graph,
        }
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
    est = sum(estimate_tokens(m["content"]) for m in messages)
    if not budget.check_can_add(agent_id="retrieval", turn_id=turn_id, budget_tokens=budget_cap, addition_tokens=est):
        context.add_violation(
            PolicyViolation(
                kind="context_budget_overflow",
                agent_id="retrieval",
                detail="estimated prompt tokens exceed remaining budget before call",
            )
        )
    plan, pt, ct = await llm.complete_json(messages=messages, agent_id="retrieval")
    budget.record_usage(agent_id="retrieval", turn_id=turn_id, tokens=pt + ct)
    snap = budget.snapshot_report(agent_id="retrieval", turn_id=turn_id, budget_tokens=budget_cap)
    if snap.used_tokens > budget_cap:
        context.add_violation(
            PolicyViolation(
                kind="context_budget_overflow",
                agent_id="retrieval",
                detail="post-call token usage exceeds declared cap",
            )
        )

    evidence = await multi_hop_retrieve(
        session,
        hop1_query=str(plan.get("hop1_query", context.user_query)),
        hop2_query=str(plan.get("hop2_query", context.user_query + " context")),
    )
    plan["evidence"] = evidence
    if len(evidence["chunks"]) < 2:
        context.add_violation(
            PolicyViolation(
                kind="multi_hop_evidence_insufficient",
                agent_id="retrieval",
                detail="fewer than two chunks available after two-hop retrieval",
            )
        )

    top_ids = [c["chunk_id"] for c in evidence["chunks"][:2]]
    for claim in plan.get("claims", []):
        existing = [str(x) for x in claim.get("chunk_ids", [])]
        merged = list(dict.fromkeys(existing + top_ids))
        claim["chunk_ids"] = merged[: max(2, len(merged))] if len(top_ids) >= 2 else merged
    chunk_index = {c["chunk_id"]: c for c in evidence["chunks"]}
    for claim in plan.get("claims", []):
        claim_text = str(claim.get("text", ""))
        for cid in claim.get("chunk_ids", []):
            meta = chunk_index.get(str(cid), {})
            span = str(meta.get("text", cid))[:500]
            context.citations.append(
                CitationRecord(
                    chunk_id=str(cid),
                    document_id=str(meta.get("document_id", "unknown")),
                    text_span=span,
                    used_for_claim=claim_text,
                )
            )

    context.retrieval_scratchpad = plan
    context.append_message(
        AgentMessage(
            role="assistant",
            agent_id="retrieval",
            content=json.dumps(plan),
            metadata={"structured": True},
        )
    )
    await persist_structured_log(
        session,
        StructuredLogEntry(
            job_id=str(job_uuid),
            agent_id="retrieval",
            event_type="agent_end",
            input_hash=sha256_json(messages),
            output_hash=sha256_json(plan),
            token_count=pt + ct,
            payload={"chunks": len(evidence["chunks"])},
        ),
    )
    return plan
