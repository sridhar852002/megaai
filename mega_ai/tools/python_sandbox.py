"""Isolated Python execution in a subprocess — user code runs in a temp file, not in-process."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from mega_ai.core.settings import get_settings
from mega_ai.tools.contracts import ToolFailure, ToolFailureCode, ToolSuccess


async def run_python_sandbox(code: str) -> dict[str, Any]:
    settings = get_settings()
    timeout = settings.sandbox_timeout_seconds
    started = time.perf_counter()
    if not isinstance(code, str) or not code.strip():
        return ToolFailure(
            code=ToolFailureCode.MALFORMED_INPUT,
            detail="code must be non-empty string",
        ).model_dump()

    tmp = Path(tempfile.mkstemp(prefix="mega_ai_sandbox_", suffix=".py")[1])
    try:
        tmp.write_text(code, encoding="utf-8")
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(tmp),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return ToolFailure(
                code=ToolFailureCode.TIMEOUT,
                detail=f"sandbox exceeded {timeout}s",
                partial={"latency_ms": (time.perf_counter() - started) * 1000},
            ).model_dump()
    finally:
        tmp.unlink(missing_ok=True)

    latency = (time.perf_counter() - started) * 1000
    data = {
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
        "exit_code": int(proc.returncode or 0),
        "latency_ms": latency,
    }
    if data["stdout"] == "" and data["stderr"] == "" and proc.returncode == 0:
        return ToolFailure(
            code=ToolFailureCode.EMPTY_RESULT,
            detail="process produced no output",
            partial=data,
        ).model_dump()
    return ToolSuccess(data=data).model_dump()
