"""Meta-agent proposes prompt rewrites for human review — never auto-applies."""

from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from mega_ai.core.hashing import sha256_json
from mega_ai.core.settings import Settings
from mega_ai.db import models
from mega_ai.llm.client import LLMClient
from mega_ai.prompts.registry import DEFAULT_PROMPTS


async def maybe_propose_prompt_update(
    *,
    session: AsyncSession,
    eval_run_id: uuid.UUID,
    settings: Settings,
) -> models.PromptRewriteProposalRow | None:
    stmt = select(models.EvalCaseResultRow).where(models.EvalCaseResultRow.eval_run_id == eval_run_id)
    rows = list((await session.execute(stmt)).scalars().all())
    if not rows:
        return None

    failure_context: dict[str, Any] = {
        "cases": [
            {
                "case_id": r.case_id,
                "category": r.category,
                "scores": r.scores,
            }
            for r in rows
        ]
    }
    llm = LLMClient(settings)
    messages = [
        {
            "role": "system",
            "content": DEFAULT_PROMPTS.meta_optimizer_system,
        },
        {"role": "user", "content": json.dumps(failure_context)},
    ]
    data, _pt, _ct = await llm.complete_json(messages=messages, agent_id="meta_prompt_optimizer")
    prompt_key = str(data.get("weakest_prompt_key", "orchestrator_system"))
    proposal = models.PromptRewriteProposalRow(
        eval_run_id=eval_run_id,
        prompt_key=prompt_key,
        proposed_text=str(data.get("proposed_rewrite", "")),
        diff=str(data.get("diff", "")),
        justification=str(data.get("justification", "")),
        status="pending",
    )
    session.add(proposal)
    await session.flush()
    return proposal


async def score_delta_on_cases(
    session: AsyncSession,
    *,
    case_ids: list[str],
    settings: Settings,
) -> dict[str, Any]:
    """Re-score subset after approvals; compares average score to previous eval snapshot."""
    from mega_ai.evals.harness import run_eval_suite

    previous = await session.execute(select(models.EvalRunRow).order_by(models.EvalRunRow.created_at.desc()))
    last = previous.scalars().first()
    before = (last.summary or {}).get("per_dimension_avg", {}) if last else {}
    new_run = await run_eval_suite(
        session=session,
        settings=settings,
        label="targeted_reeval",
        case_filter=set(case_ids),
        trigger_meta=False,
    )
    after = new_run.summary.get("per_dimension_avg", {})
    delta = {
        "before": before,
        "after": after,
        "diff_sha": sha256_json({"before": before, "after": after}),
    }
    merged = dict(new_run.summary)
    merged["performance_delta"] = delta
    new_run.summary = merged
    flag_modified(new_run, "summary")
    await session.flush()
    return delta
