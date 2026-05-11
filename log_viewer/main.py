"""Read-only browsing UI for structured_logs (separate from the five core API routes)."""

from __future__ import annotations

import html
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, text

app = FastAPI(title="Mega AI Log Viewer")
engine = create_engine(os.environ["DATABASE_SYNC_URL"], pool_pre_ping=True)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, job_id, agent_id, event_type, timestamp "
                "FROM structured_logs ORDER BY id DESC LIMIT 200"
            )
        ).mappings().all()
    parts = [
        "<html><head><title>Mega AI Logs</title>",
        "<style>body{font-family:system-ui;} table{border-collapse:collapse;} td,th{border:1px solid #ccc;padding:4px;}</style>",
        "</head><body><h1>Structured logs (latest 200)</h1><table><tr><th>id</th><th>job_id</th><th>agent</th><th>event</th><th>ts</th></tr>",
    ]
    for r in rows:
        row_html = "".join(
            [
                "<tr>",
                f"<td>{r['id']}</td>",
                f"<td>{html.escape(str(r['job_id']))}</td>",
                f"<td>{html.escape(str(r['agent_id']))}</td>",
                f"<td>{html.escape(str(r['event_type']))}</td>",
                f"<td>{html.escape(str(r['timestamp']))}</td>",
                "</tr>",
            ]
        )
        parts.append(row_html)
    parts.append("</table></body></html>")
    return "".join(parts)
