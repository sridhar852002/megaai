"""Invokes tools with explicit retry budgets and structured logging."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from mega_ai.core.context import SharedContext, ToolResultRecord
from mega_ai.core.hashing import sha256_json
from mega_ai.core.structured_logging import StructuredLogEntry
from mega_ai.db.log_service import persist_structured_log
from mega_ai.tools.contracts import ToolFailureCode

ToolCallable = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
PayloadFactory = Callable[[int], dict[str, Any]]


class ToolGateway:
    """Orchestrator-owned tool execution with acceptance callbacks and bounded retries."""

    def __init__(self, session: AsyncSession, job_uuid: uuid.UUID) -> None:
        self._session = session
        self._job_uuid = job_uuid

    async def invoke(
        self,
        *,
        context: SharedContext,
        tool_name: str,
        agent_id: str,
        payload_factory: PayloadFactory,
        runner: ToolCallable,
        accept: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any]:
        accept = accept or (lambda outcome: bool(outcome.get("ok") is True))
        last: dict[str, Any] | None = None
        for retry_index in range(3):
            payload = payload_factory(retry_index)
            call_id = str(uuid.uuid4())
            start = time.perf_counter()
            outcome = await runner(payload)
            latency_ms = (time.perf_counter() - start) * 1000
            accepted = accept(outcome)
            record = ToolResultRecord(
                tool_name=tool_name,
                call_id=call_id,
                success=bool(outcome.get("ok") is True),
                structured=outcome,
                latency_ms=latency_ms,
                accepted=accepted,
                retry_index=retry_index,
            )
            context.tool_results.append(record)
            log = StructuredLogEntry(
                job_id=str(self._job_uuid),
                agent_id=agent_id,
                event_type="tool_retry" if retry_index else "tool_call",
                input_hash=sha256_json({"tool": tool_name, "payload": payload, "retry": retry_index}),
                output_hash=sha256_json(outcome),
                latency_ms=latency_ms,
                token_count=None,
                policy_violations=[],
                payload={
                    "tool": tool_name,
                    "retry_index": retry_index,
                    "accepted": accepted,
                },
            )
            await persist_structured_log(self._session, log)

            if accepted:
                return outcome
            if outcome.get("ok") is False and outcome.get("code") == ToolFailureCode.MALFORMED_INPUT.value:
                return outcome
            if retry_index == 2:
                return last or outcome
            last = outcome
        return last or {}
