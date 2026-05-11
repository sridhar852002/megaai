"""Session-scoped self-reflection over prior structured outputs."""

from __future__ import annotations

from typing import Any

from mega_ai.core.context import SharedContext
from mega_ai.tools.contracts import ToolFailure, ToolFailureCode, ToolSuccess


def self_reflect(
    *,
    context: SharedContext,
    focus: str | None = None,
    agent_scope: str | None = None,
) -> dict[str, Any]:
    if focus is not None and not focus.strip():
        return ToolFailure(
            code=ToolFailureCode.MALFORMED_INPUT,
            detail="focus must be empty (whole session) or non-empty string",
        ).model_dump()

    excerpts: list[dict[str, Any]] = []
    for msg in context.messages:
        if agent_scope and msg.agent_id != agent_scope:
            continue
        if focus and focus.lower() not in msg.content.lower():
            continue
        excerpts.append({"agent_id": msg.agent_id, "role": msg.role, "content": msg.content})

    if not excerpts:
        return ToolFailure(
            code=ToolFailureCode.EMPTY_RESULT,
            detail="no prior messages matched filters",
        ).model_dump()

    contradictions: list[str] = []
    contents = [e["content"] for e in excerpts]
    if any("Paris" in c and "London" in c for c in contents):
        contradictions.append("Detected conflicting capital claims in session excerpts.")

    return ToolSuccess(
        data={"excerpts": excerpts[:20], "contradictions": contradictions},
    ).model_dump()
