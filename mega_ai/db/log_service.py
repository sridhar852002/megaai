"""Persist structured logs and traces."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from mega_ai.core.structured_logging import StructuredLogEntry
from mega_ai.db import models


async def persist_structured_log(session: AsyncSession, entry: StructuredLogEntry) -> None:
    row = models.StructuredLogRow(
        job_id=uuid.UUID(entry.job_id),
        agent_id=entry.agent_id,
        event_type=entry.event_type,
        input_hash=entry.input_hash,
        output_hash=entry.output_hash,
        latency_ms=entry.latency_ms,
        token_count=entry.token_count,
        policy_violations=entry.policy_violations,
        payload=entry.payload,
        timestamp=entry.timestamp,
    )
    session.add(row)
    await session.flush()
