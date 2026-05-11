"""Run full evaluation suite (invokes LLM_MODE from environment)."""

from __future__ import annotations

import asyncio

from mega_ai.core.settings import get_settings
from mega_ai.db.session import SessionLocal
from mega_ai.evals.harness import run_eval_suite


async def main() -> None:
    settings = get_settings()
    async with SessionLocal() as session:
        async with session.begin():
            await run_eval_suite(session=session, settings=settings)


if __name__ == "__main__":
    asyncio.run(main())
