"""FastAPI surface exposing exactly the five contract endpoints."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from mega_ai.api.errors import MegaAIError
from mega_ai.core.settings import get_settings
from mega_ai.db import models
from mega_ai.db.session import SessionLocal
from mega_ai.evals.harness import latest_eval_run
from mega_ai.orchestrator.pipeline import run_streaming_job
from mega_ai.prompts.registry import load_prompt_bundle
from mega_ai.workers.tasks import run_targeted_reeval_task

app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)


@app.exception_handler(MegaAIError)
async def mega_error_handler(_request: Request, exc: MegaAIError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.as_body())


@app.exception_handler(RequestValidationError)
async def validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": json.dumps(exc.errors()),
                "job_id": None,
            }
        },
    )


class QueryBody(BaseModel):
    query: str = Field(min_length=1)


class PromptReviewBody(BaseModel):
    proposal_id: uuid.UUID
    decision: str = Field(pattern="^(approve|reject)$")
    reviewer_note: str = ""


@app.post("/query")
async def post_query(body: QueryBody) -> EventSourceResponse:
    settings = get_settings()

    async def events() -> AsyncIterator[dict[str, Any]]:
        async with SessionLocal() as session:
            async with session.begin():
                job = models.JobRow(user_query=body.query, status="running", trace={}, shared_context_snapshot={})
                session.add(job)
                await session.flush()
                prompts = await load_prompt_bundle(session)
                async for payload in run_streaming_job(
                    session=session,
                    job_row=job,
                    settings=settings,
                    prompts=prompts,
                ):
                    yield {"event": payload.get("type", "message"), "data": json.dumps(payload)}

    return EventSourceResponse(events())


@app.get("/trace/{job_id}")
async def get_trace(job_id: uuid.UUID) -> dict[str, Any]:
    async with SessionLocal() as session:
        job = await session.get(models.JobRow, job_id)
        if job is None:
            raise MegaAIError(
                code="JOB_NOT_FOUND",
                message="Unknown job_id",
                status_code=404,
                job_id=job_id,
            )
        return {
            "job_id": str(job.id),
            "status": job.status,
            "trace": job.trace,
            "shared_context": job.shared_context_snapshot,
        }


@app.get("/evals/latest")
async def get_latest_eval() -> dict[str, Any]:
    async with SessionLocal() as session:
        run = await latest_eval_run(session)
        if run is None:
            raise MegaAIError(
                code="EVAL_NOT_FOUND",
                message="No evaluation run recorded yet",
                status_code=404,
            )
        stmt = select(models.EvalCaseResultRow).where(models.EvalCaseResultRow.eval_run_id == run.id)
        case_rows = list((await session.execute(stmt)).scalars().all())
        return {
            "eval_run_id": str(run.id),
            "created_at": run.created_at.isoformat(),
            "label": run.label,
            "summary": run.summary,
            "cases": [
                {
                    "case_id": row.case_id,
                    "category": row.category,
                    "scores": row.scores,
                }
                for row in case_rows
            ],
        }


@app.post("/prompt-review")
async def prompt_review(body: PromptReviewBody) -> dict[str, Any]:
    async with SessionLocal() as session:
        async with session.begin():
            proposal = await session.get(models.PromptRewriteProposalRow, body.proposal_id)
            if proposal is None:
                raise MegaAIError(
                    code="PROPOSAL_NOT_FOUND",
                    message="Unknown proposal_id",
                    status_code=404,
                )
            if proposal.status != "pending":
                raise MegaAIError(
                    code="PROPOSAL_NOT_PENDING",
                    message="Proposal already reviewed",
                    status_code=409,
                )
            proposal.status = "approved" if body.decision == "approve" else "rejected"
            session.add(
                models.PromptReviewActionRow(
                    proposal_id=proposal.id,
                    decision=body.decision,
                    reviewer_note=body.reviewer_note,
                )
            )
            if body.decision == "approve":
                session.add(
                    models.ApprovedPromptVersionRow(
                        prompt_key=proposal.prompt_key,
                        text=proposal.proposed_text,
                        proposal_id=proposal.id,
                    )
                )
            return {
                "proposal_id": str(proposal.id),
                "status": proposal.status,
            }


@app.post("/reeval")
async def reeval() -> dict[str, Any]:
    run_targeted_reeval_task.send()
    return {"queued": True}
