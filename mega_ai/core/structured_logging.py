"""Structured logging emitted by orchestrator and agents."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

LogEventType = Literal[
    "routing",
    "agent_start",
    "agent_end",
    "tool_call",
    "tool_retry",
    "token_usage",
    "policy_violation",
    "sse_chunk",
    "eval_case",
    "meta_prompt_proposal",
    "prompt_review",
    "compression",
]


class StructuredLogEntry(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    job_id: str
    agent_id: str
    event_type: LogEventType
    input_hash: str
    output_hash: str
    latency_ms: float | None = None
    token_count: int | None = None
    policy_violations: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
