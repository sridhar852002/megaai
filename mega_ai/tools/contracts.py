"""Tool failure taxonomy and payloads — never silent recovery."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolFailureCode(StrEnum):
    TIMEOUT = "TIMEOUT"
    EMPTY_RESULT = "EMPTY_RESULT"
    MALFORMED_INPUT = "MALFORMED_INPUT"


class ToolSuccess(BaseModel):
    ok: Literal[True] = True
    data: dict[str, Any]


class ToolFailure(BaseModel):
    ok: Literal[False] = False
    code: ToolFailureCode
    detail: str
    partial: dict[str, Any] = Field(default_factory=dict)


ToolOutcome = ToolSuccess | ToolFailure
