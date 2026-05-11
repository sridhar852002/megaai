"""Runs the offline evaluation suite and persists reproducible artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mega_ai.core.context import SharedContext
from mega_ai.core.settings import Settings
from mega_ai.db import models
from mega_ai.evals.cases import EVAL_CASES
from mega_ai.evals.meta import maybe_propose_prompt_update
from mega_ai.evals.scoring import score_case
from mega_ai.orchestrator.pipeline import drain_streaming_job
from mega_ai.prompts.registry import load_prompt_bundle, snapshot_default_config


def _final_answer_from_snapshot(snapshot: dict[str, Any]) -> str:
    for msg in reversed(snapshot.get("messages", [])):
        if msg.get("agent_id") == "synthesis":
            try:
                payload = json.loads(msg.get("content", "{}"))
                return str(payload.get("final_answer", ""))
            except json.JSONDecodeError:
                return ""
    return ""


async def run_eval_suite(
    *,
    session: AsyncSession,
    settings: Settings,
    label: str = "full",
    case_filter: set[str] | None = None,
    trigger_meta: bool = True,
) -> models.EvalRunRow:
    prompts = await load_prompt_bundle(session)
    run = models.EvalRunRow(
        label=label,
        config_snapshot={
            "prompts": snapshot_default_config(),
            "llm_mode": settings.llm_mode,
            "litellm_model": settings.litellm_model,
        },
        summary={},
        human_readable_report="",
    )
    session.add(run)
    await session.flush()

    per_category: dict[str, list[float]] = defaultdict(list)
    per_dimension: dict[str, list[float]] = defaultdict(list)
    failed_case_ids: list[str] = []

    for case in EVAL_CASES:
        if case_filter is not None and case.case_id not in case_filter:
            continue
        job = models.JobRow(user_query=case.query, status="queued", trace={}, shared_context_snapshot={})
        session.add(job)
        await session.flush()
        await drain_streaming_job(
            session=session,
            job_row=job,
            settings=settings,
            prompts=prompts,
        )
        snapshot = job.shared_context_snapshot
        answer = _final_answer_from_snapshot(snapshot)
        ctx = SharedContext.model_validate(snapshot)
        report = score_case(
            final_answer=answer,
            context=ctx,
            expected=case.expected_answer,
            category=case.category,
        )
        if report.failed:
            failed_case_ids.append(case.case_id)

        row = models.EvalCaseResultRow(
            eval_run_id=run.id,
            case_id=case.case_id,
            category=case.category,
            trace_payload={
                "job_id": str(job.id),
                "prompts": snapshot.get("messages", []),
                "tool_results": snapshot.get("tool_results", []),
                "routing": snapshot.get("routing_decisions", []),
                "final_answer": answer,
            },
            scores=[d.model_dump(mode="json") for d in report.dimensions],
        )
        session.add(row)

        for dim in report.dimensions:
            per_category[case.category].append(dim.score)
            per_dimension[dim.dimension].append(dim.score)

    summary = {
        "per_category_avg": {k: sum(v) / len(v) for k, v in per_category.items()},
        "per_dimension_avg": {k: sum(v) / len(v) for k, v in per_dimension.items()},
        "failed_case_ids": failed_case_ids,
    }
    run.summary = summary
    run.human_readable_report = json.dumps(summary, indent=2)
    await session.flush()

    if trigger_meta and failed_case_ids:
        await maybe_propose_prompt_update(session=session, eval_run_id=run.id, settings=settings)

    return run


async def latest_eval_run(session: AsyncSession) -> models.EvalRunRow | None:
    stmt = select(models.EvalRunRow).order_by(models.EvalRunRow.created_at.desc()).limit(1)
    res = await session.execute(stmt)
    return res.scalar_one_or_none()
