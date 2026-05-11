"""End-to-end orchestration with explicit tool policies and streamed events."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from mega_ai.agents.compression import run_compression
from mega_ai.agents.critique import run_critique
from mega_ai.agents.decomposition import run_decomposition
from mega_ai.agents.retrieval import run_retrieval
from mega_ai.agents.synthesis import prepare_synthesis, stream_answer_text
from mega_ai.core.context import AgentMessage, PolicyViolation, SharedContext
from mega_ai.core.hashing import sha256_json
from mega_ai.core.settings import Settings
from mega_ai.core.structured_logging import StructuredLogEntry
from mega_ai.db import models
from mega_ai.db.log_service import persist_structured_log
from mega_ai.llm.client import LLMClient
from mega_ai.orchestrator.context_budget import ContextBudgetManager, estimate_tokens
from mega_ai.orchestrator.dag import task_execution_order
from mega_ai.orchestrator.router import decide_next_route
from mega_ai.orchestrator.tool_gateway import ToolGateway
from mega_ai.prompts.registry import PromptBundle
from mega_ai.rag.service import seed_demo_corpus
from mega_ai.tools.nl2sql import nl_to_sql_lookup
from mega_ai.tools.python_sandbox import run_python_sandbox
from mega_ai.tools.self_reflect import self_reflect
from mega_ai.tools.web_search import web_search_stub


def _wants_sql(text: str) -> bool:
    return bool(re.search(r"\b(sql|database|table|capitals|inventors)\b", text, re.I))


def _wants_code(text: str) -> bool:
    return bool(re.search(r"\b(python|calculate|compute|code)\b", text, re.I))


def _wants_search(text: str) -> bool:
    return bool(re.search(r"\b(latest|news|search|web)\b", text, re.I))


async def _emit_budget(
    context: SharedContext,
    budget: ContextBudgetManager,
    decision_budgets: dict[str, int],
) -> dict[str, Any]:
    snapshot: dict[str, int] = {}
    for agent, cap in decision_budgets.items():
        snapshot[agent] = budget.snapshot_report(agent_id=agent, turn_id=context.job_id, budget_tokens=cap).remaining
    return {
        "type": "context_budget",
        "remaining": snapshot,
        "violations": [v.detail for v in context.policy_violations],
    }


async def _maybe_compress_history(
    *,
    session: AsyncSession,
    job_uuid: uuid.UUID,
    context: SharedContext,
    llm: LLMClient,
    prompts: PromptBundle,
    budget: ContextBudgetManager,
    caps: dict[str, int],
    turn_id: str,
    budget_cap: int,
) -> None:
    convo_parts: list[str] = []
    structured: dict[str, Any] = {}
    for msg in context.messages:
        if msg.metadata.get("structured"):
            structured[f"{msg.agent_id}:{len(structured)}"] = msg.content
        else:
            convo_parts.append(f"{msg.agent_id}: {msg.content}")
    conversational = "\n".join(convo_parts)
    est = estimate_tokens(conversational)
    if est <= budget_cap // 2:
        return
    compressed = await run_compression(
        session=session,
        job_uuid=job_uuid,
        context=context,
        llm=llm,
        system_prompt=prompts.compression_system,
        budget=budget,
        turn_id=turn_id,
        budget_cap=int(caps.get("compression", 1024)),
        conversational_text=conversational,
        structured_facts=structured,
    )
    context.messages = [m for m in context.messages if m.metadata.get("structured")] + [
        AgentMessage(
            role="assistant",
            agent_id="compression",
            content=compressed,
            metadata={"structured": False},
        )
    ]


async def orchestrator_tool_pass(
    *,
    gateway: ToolGateway,
    context: SharedContext,
    llm: LLMClient,
    prompts: PromptBundle,
) -> None:
    query = context.user_query

    if _wants_search(query):

        async def run_search(payload: dict[str, Any]) -> dict[str, Any]:
            return await web_search_stub(str(payload["query"]))

        await gateway.invoke(
            context=context,
            tool_name="web_search_stub",
            agent_id="orchestrator",
            payload_factory=lambda attempt: {"query": f"{query} #{attempt}"},
            runner=run_search,
            accept=lambda out: out.get("ok") is True,
        )

    if _wants_sql(query):

        async def run_sql(payload: dict[str, Any]) -> dict[str, Any]:
            return await nl_to_sql_lookup(str(payload["question"]), llm=llm, nl2sql_system_prompt=prompts.nl2sql_system)

        await gateway.invoke(
            context=context,
            tool_name="nl2sql",
            agent_id="orchestrator",
            payload_factory=lambda attempt: {"question": query if attempt == 0 else f"{query} (retry {attempt})"},
            runner=run_sql,
            accept=lambda out: out.get("ok") is True,
        )

    if _wants_code(query):

        async def run_code(payload: dict[str, Any]) -> dict[str, Any]:
            return await run_python_sandbox(str(payload["code"]))

        snippet = "print(40+2)\n"
        await gateway.invoke(
            context=context,
            tool_name="python_sandbox",
            agent_id="orchestrator",
            payload_factory=lambda _attempt: {"code": snippet},
            runner=run_code,
            accept=lambda out: out.get("ok") is True and out.get("data", {}).get("exit_code") == 0,
        )


async def _stream_json_agent_output(agent: str, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    text = json.dumps(payload)
    async for chunk in _stream_text_chunks(agent, text):
        yield chunk


async def _stream_text_chunks(agent: str, text: str) -> AsyncIterator[dict[str, Any]]:
    for word in text.split(" "):
        yield {"type": "token", "agent": agent, "text": word + " "}


def _build_trace(context: SharedContext) -> dict[str, Any]:
    return {
        "routing": [d.model_dump(mode="json") for d in context.routing_decisions],
        "tool_results": [t.model_dump(mode="json") for t in context.tool_results],
        "critiques": [c.model_dump(mode="json") for c in context.critique_results],
        "violations": [v.model_dump(mode="json") for v in context.policy_violations],
        "citations": [c.model_dump(mode="json") for c in context.citations],
        "provenance": [p.model_dump(mode="json") for p in context.provenance],
    }


async def run_streaming_job(
    *,
    session: AsyncSession,
    job_row: models.JobRow,
    settings: Settings,
    prompts: PromptBundle,
) -> AsyncIterator[dict[str, Any]]:
    await seed_demo_corpus(session, settings)
    context = SharedContext(job_id=str(job_row.id), user_query=job_row.user_query)
    llm = LLMClient(settings)
    budget = ContextBudgetManager()
    gateway = ToolGateway(session, job_row.id)
    turn_id = str(job_row.id)

    yield {"type": "job", "job_id": str(job_row.id), "status": "started"}

    prior: dict[str, str] = {}
    decision = await decide_next_route(
        session=session,
        job_uuid=job_row.id,
        context=context,
        llm=llm,
        orchestrator_prompt=prompts.orchestrator_system,
        prior_outputs=prior,
    )
    caps = decision.context_budget_tokens
    yield await _emit_budget(context, budget, caps)

    synthesis_done = False

    for agent in decision.order:
        yield {"type": "agent_start", "agent": agent}
        if agent == "decomposition":
            cap = int(caps.get("decomposition", 2048))
            await _maybe_compress_history(
                session=session,
                job_uuid=job_row.id,
                context=context,
                llm=llm,
                prompts=prompts,
                budget=budget,
                caps=caps,
                turn_id=turn_id,
                budget_cap=cap,
            )
            data = await run_decomposition(
                session=session,
                job_uuid=job_row.id,
                context=context,
                llm=llm,
                system_prompt=prompts.decomposition_system,
                budget=budget,
                turn_id=turn_id,
                budget_cap=cap,
            )
            tasks = list(data.get("tasks", []))
            _order, violations = task_execution_order(tasks)
            _ = _order
            for msg in violations:
                context.add_violation(PolicyViolation(kind="decomposition_dag", agent_id="decomposition", detail=msg))
            prior["decomposition"] = json.dumps(data)
            async for piece in _stream_json_agent_output(agent, data):
                yield piece
        elif agent == "retrieval":
            await orchestrator_tool_pass(gateway=gateway, context=context, llm=llm, prompts=prompts)
            cap = int(caps.get("retrieval", 4096))
            await _maybe_compress_history(
                session=session,
                job_uuid=job_row.id,
                context=context,
                llm=llm,
                prompts=prompts,
                budget=budget,
                caps=caps,
                turn_id=turn_id,
                budget_cap=cap,
            )
            plan = await run_retrieval(
                session=session,
                job_uuid=job_row.id,
                context=context,
                llm=llm,
                system_prompt=prompts.retrieval_system,
                budget=budget,
                turn_id=turn_id,
                budget_cap=cap,
            )
            prior["retrieval"] = json.dumps(plan)
            async for piece in _stream_json_agent_output(agent, plan):
                yield piece
        elif agent == "critique":
            cap = int(caps.get("critique", 3072))
            await run_critique(
                session=session,
                job_uuid=job_row.id,
                context=context,
                llm=llm,
                system_prompt=prompts.critique_system,
                budget=budget,
                turn_id=turn_id,
                budget_cap=cap,
                target_agents=["decomposition", "retrieval"],
            )
            crit_dump = json.dumps([c.model_dump() for c in context.critique_results])
            prior["critique"] = crit_dump
            async for piece in _stream_text_chunks(agent, crit_dump):
                yield piece
            if context.critique_results:
                had_disagree = any(span.disagree for span in context.critique_results[-1].spans)
            else:
                had_disagree = False
            if had_disagree or "contradiction" in context.user_query.lower():

                async def run_reflect(payload: dict[str, Any]) -> dict[str, Any]:
                    return await asyncio.to_thread(
                        lambda: self_reflect(
                            context=context,
                            focus=str(payload.get("focus", "")) or None,
                            agent_scope=payload.get("agent_scope"),
                        )
                    )

                await gateway.invoke(
                    context=context,
                    tool_name="self_reflect",
                    agent_id="orchestrator",
                    payload_factory=lambda attempt: {"focus": context.user_query, "attempt": attempt},
                    runner=run_reflect,
                    accept=lambda out: out.get("ok") is True,
                )
        elif agent == "synthesis":
            cap = int(caps.get("synthesis", 4096))
            syn = await prepare_synthesis(
                session=session,
                job_uuid=job_row.id,
                context=context,
                llm=llm,
                system_prompt=prompts.synthesis_system,
                budget=budget,
                turn_id=turn_id,
                budget_cap=cap,
            )
            answer = str(syn.get("final_answer", ""))
            async for chunk in stream_answer_text(answer):
                yield {"type": "token", "agent": "synthesis", "text": chunk}
            synthesis_done = True
        yield await _emit_budget(context, budget, caps)
        yield {"type": "agent_end", "agent": agent}

    second = await decide_next_route(
        session=session,
        job_uuid=job_row.id,
        context=context,
        llm=llm,
        orchestrator_prompt=prompts.orchestrator_system,
        prior_outputs=prior,
    )
    if not synthesis_done and "synthesis" in second.order:
        yield {"type": "routing_update", "decision": second.model_dump(mode="json")}
        cap = int(second.context_budget_tokens.get("synthesis", 4096))
        syn = await prepare_synthesis(
            session=session,
            job_uuid=job_row.id,
            context=context,
            llm=llm,
            system_prompt=prompts.synthesis_system,
            budget=budget,
            turn_id=turn_id,
            budget_cap=cap,
        )
        answer = str(syn.get("final_answer", ""))
        async for chunk in stream_answer_text(answer):
            yield {"type": "token", "agent": "synthesis", "text": chunk}

    job_row.shared_context_snapshot = json.loads(context.model_dump_json())
    job_row.trace = _build_trace(context)
    job_row.status = "completed"
    await session.flush()


async def drain_streaming_job(
    *,
    session: AsyncSession,
    job_row: models.JobRow,
    settings: Settings,
    prompts: PromptBundle,
) -> None:
    async for _evt in run_streaming_job(session=session, job_row=job_row, settings=settings, prompts=prompts):
        pass


async def persist_log_token(
    session: AsyncSession,
    job_id: uuid.UUID,
    agent: str,
    text: str,
) -> None:
    await persist_structured_log(
        session,
        StructuredLogEntry(
            job_id=str(job_id),
            agent_id=agent,
            event_type="sse_chunk",
            input_hash=sha256_json({"agent": agent}),
            output_hash=sha256_json({"text": text}),
            token_count=estimate_tokens(text),
            payload={"stream": True},
        ),
    )
