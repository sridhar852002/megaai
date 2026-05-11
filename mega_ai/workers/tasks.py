"""Dramatiq actors for long-running evaluation work."""

from __future__ import annotations

import asyncio

import dramatiq

import mega_ai.workers.broker  # noqa: F401


@dramatiq.actor
def run_targeted_reeval_task() -> None:
    from mega_ai.core.settings import get_settings
    from mega_ai.db.session import SessionLocal
    from mega_ai.evals.harness import latest_eval_run
    from mega_ai.evals.meta import score_delta_on_cases

    async def _run() -> None:
        settings = get_settings()
        async with SessionLocal() as session:
            async with session.begin():
                latest = await latest_eval_run(session)
                if latest is None:
                    return
                failed = list(latest.summary.get("failed_case_ids", []))
                if not failed:
                    return
                await score_delta_on_cases(session, case_ids=failed, settings=settings)

    asyncio.run(_run())


@dramatiq.actor
def run_full_eval_task(label: str = "scheduled") -> None:
    from mega_ai.core.settings import get_settings
    from mega_ai.db.session import SessionLocal
    from mega_ai.evals.harness import run_eval_suite

    async def _run() -> None:
        settings = get_settings()
        async with SessionLocal() as session:
            async with session.begin():
                await run_eval_suite(session=session, settings=settings, label=label)

    asyncio.run(_run())
