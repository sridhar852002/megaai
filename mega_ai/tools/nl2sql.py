"""Schema-aware NL→SQL with SELECT-only guardrails."""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from mega_ai.llm.client import LLMClient
from mega_ai.tools.contracts import ToolFailure, ToolFailureCode, ToolSuccess

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS capitals (
  country TEXT PRIMARY KEY,
  capital TEXT NOT NULL,
  population TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inventors (
  invention TEXT PRIMARY KEY,
  inventor TEXT NOT NULL,
  year INTEGER NOT NULL
);
"""


def _ensure_sqlite_path() -> Path:
    base = Path(__file__).resolve().parent / "_mega_ai_demo.sqlite3"
    return base


def _init_sqlite(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.execute("DELETE FROM capitals")
        conn.execute("DELETE FROM inventors")
        conn.executemany(
            "INSERT INTO capitals(country, capital, population) VALUES (?,?,?)",
            [
                ("France", "Paris", "2.1M"),
                ("Germany", "Berlin", "3.6M"),
                ("Japan", "Tokyo", "14M"),
            ],
        )
        conn.executemany(
            "INSERT INTO inventors(invention, inventor, year) VALUES (?,?,?)",
            [
                ("telephone", "Alexander Graham Bell", 1876),
                ("light bulb", "Thomas Edison", 1879),
            ],
        )
        conn.commit()
    finally:
        conn.close()


async def nl_to_sql_lookup(
    question: str,
    *,
    llm: LLMClient,
    nl2sql_system_prompt: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not question.strip():
        return ToolFailure(
            code=ToolFailureCode.MALFORMED_INPUT,
            detail="question empty",
        ).model_dump()

    db_path = _ensure_sqlite_path()
    if not db_path.exists():
        _init_sqlite(db_path)

    schema_hint = "Tables: capitals(country,capital,population); inventors(invention,inventor,year)"
    messages = [
        {"role": "system", "content": nl2sql_system_prompt + "\n" + schema_hint},
        {"role": "user", "content": question},
    ]
    parsed, _pt, _ct = await llm.complete_json(messages=messages, agent_id="nl2sql")
    sql = str(parsed.get("sql", "")).strip()
    if not re.fullmatch(r"(?is)\s*select\b.*", sql or ""):
        return ToolFailure(
            code=ToolFailureCode.MALFORMED_INPUT,
            detail="only single SELECT statements are permitted",
            partial={"candidate": sql},
        ).model_dump()
    forbidden = re.compile(r"\b(update|delete|insert|drop|alter|pragma|attach)\b", re.I)
    if forbidden.search(sql):
        return ToolFailure(
            code=ToolFailureCode.MALFORMED_INPUT,
            detail="DML/DDL keywords not allowed",
            partial={"candidate": sql},
        ).model_dump()

    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql)
        rows = [dict(row) for row in cur.fetchmany(50)]
    except sqlite3.Error as exc:
        return ToolFailure(
            code=ToolFailureCode.MALFORMED_INPUT,
            detail=f"sql rejected by sqlite: {exc}",
            partial={"candidate": sql},
        ).model_dump()
    finally:
        conn.close()

    latency = (time.perf_counter() - started) * 1000
    if not rows:
        return ToolFailure(
            code=ToolFailureCode.EMPTY_RESULT,
            detail="query returned zero rows",
            partial={"sql": sql, "latency_ms": latency},
        ).model_dump()

    return ToolSuccess(
        data={"sql": sql, "rows": rows, "latency_ms": latency},
    ).model_dump()
