"""HTTP exception envelopes with machine-readable codes."""

from __future__ import annotations

from typing import Any
from uuid import UUID


class MegaAIError(Exception):
    def __init__(self, *, code: str, message: str, status_code: int, job_id: UUID | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.job_id = job_id

    def as_body(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "job_id": str(self.job_id) if self.job_id is not None else None,
            }
        }
