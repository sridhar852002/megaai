"""Web search stub returning structured hits."""

from __future__ import annotations

import time
from typing import Any

from mega_ai.tools.contracts import ToolFailure, ToolFailureCode, ToolSuccess


async def web_search_stub(query: str, *, timeout_s: float = 1.5) -> dict[str, Any]:
    started = time.perf_counter()
    if not query.strip():
        return ToolFailure(
            code=ToolFailureCode.MALFORMED_INPUT,
            detail="query must be non-empty",
        ).model_dump()

    if "timeoutplease" in query.lower():
        await __import__("asyncio").sleep(timeout_s + 0.1)
        latency = (time.perf_counter() - started) * 1000
        return ToolFailure(
            code=ToolFailureCode.TIMEOUT,
            detail=f"exceeded {timeout_s}s budget",
            partial={"latency_ms": latency},
        ).model_dump()

    if "emptyplease" in query.lower():
        return ToolFailure(
            code=ToolFailureCode.EMPTY_RESULT,
            detail="stub chose empty catalog slice",
        ).model_dump()

    latency = (time.perf_counter() - started) * 1000
    data = {
        "results": [
            {
                "title": f"Stub hit for: {query[:40]}",
                "url": "https://example.com/stub",
                "relevance_score": 0.91,
                "snippet": "Synthetic snippet for orchestration testing.",
            }
        ],
        "latency_ms": latency,
    }
    return ToolSuccess(data=data).model_dump()
