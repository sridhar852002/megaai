"""Versioned prompt text. Proposals create new candidate strings; approvals update runtime reads."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mega_ai.db import models


@dataclass(frozen=True)
class PromptBundle:
    orchestrator_system: str
    decomposition_system: str
    retrieval_system: str
    critique_system: str
    synthesis_system: str
    compression_system: str
    meta_optimizer_system: str
    nl2sql_system: str


DEFAULT_PROMPTS = PromptBundle(
    orchestrator_system=(
        "You route agents for Mega AI. Output strict JSON with keys: "
        "chosen_agents (list), order (list), reason (string), confidence (0-1 float), "
        "context_budget_tokens (map agent->int). "
        "Never hardcode a fixed chain: justify based on query cues and risk."
    ),
    decomposition_system=(
        "Decompose the user task into a DAG of typed subtasks. JSON keys: tasks[{id,type,depends_on}], notes."
    ),
    retrieval_system=(
        "Plan multi-hop retrieval. JSON keys: hop1_query, hop2_query, claims[{text,chunk_ids}]. "
        "You must use information implied by at least two chunk ids per claim when possible."
    ),
    critique_system=(
        "Critique another agent's latest output. JSON keys: spans[{claim_span,confidence,disagree,"
        "suggested_correction}], overall_notes. Focus on claim spans, not whole-document rejection."
    ),
    synthesis_system=(
        "Merge agent outputs, reconcile critique disagreements, produce final_answer string and "
        "provenance list[{sentence,source_agent,source_chunk_id|null,source_tool_call_id|null}]."
    ),
    compression_system=(
        "Compress conversational filler only. JSON keys: compressed_filler. "
        "Preserve verbatim structured facts given separately."
    ),
    meta_optimizer_system=(
        "Analyze failing eval dimensions, pick weakest prompt_key, propose_rewrite, diff, justification."
    ),
    nl2sql_system=("Translate natural language to a single SELECT for the given schema. Return JSON {sql:string}."),
)


async def load_prompt_bundle(session: AsyncSession) -> PromptBundle:
    """Overlay latest approved prompt texts keyed by prompt_key."""

    keys = [
        "orchestrator_system",
        "decomposition_system",
        "retrieval_system",
        "critique_system",
        "synthesis_system",
        "compression_system",
        "meta_optimizer_system",
        "nl2sql_system",
    ]
    overrides: dict[str, str] = {}
    for key in keys:
        stmt = (
            select(models.ApprovedPromptVersionRow)
            .where(models.ApprovedPromptVersionRow.prompt_key == key)
            .order_by(models.ApprovedPromptVersionRow.approved_at.desc())
            .limit(1)
        )
        res = await session.execute(stmt)
        row = res.scalar_one_or_none()
        if row is not None:
            overrides[key] = row.text
    base = DEFAULT_PROMPTS
    return PromptBundle(
        orchestrator_system=overrides.get("orchestrator_system", base.orchestrator_system),
        decomposition_system=overrides.get("decomposition_system", base.decomposition_system),
        retrieval_system=overrides.get("retrieval_system", base.retrieval_system),
        critique_system=overrides.get("critique_system", base.critique_system),
        synthesis_system=overrides.get("synthesis_system", base.synthesis_system),
        compression_system=overrides.get("compression_system", base.compression_system),
        meta_optimizer_system=overrides.get("meta_optimizer_system", base.meta_optimizer_system),
        nl2sql_system=overrides.get("nl2sql_system", base.nl2sql_system),
    )


def snapshot_default_config() -> dict[str, str]:
    return {
        "orchestrator_system": DEFAULT_PROMPTS.orchestrator_system,
        "decomposition_system": DEFAULT_PROMPTS.decomposition_system,
        "retrieval_system": DEFAULT_PROMPTS.retrieval_system,
        "critique_system": DEFAULT_PROMPTS.critique_system,
        "synthesis_system": DEFAULT_PROMPTS.synthesis_system,
        "compression_system": DEFAULT_PROMPTS.compression_system,
        "meta_optimizer_system": DEFAULT_PROMPTS.meta_optimizer_system,
        "nl2sql_system": DEFAULT_PROMPTS.nl2sql_system,
    }
