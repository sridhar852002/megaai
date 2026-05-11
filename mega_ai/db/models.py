"""SQLAlchemy models."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    user_query: Mapped[str] = mapped_column(Text())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    trace: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    shared_context_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class StructuredLogRow(Base):
    __tablename__ = "structured_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("jobs.id"), index=True)
    agent_id: Mapped[str] = mapped_column(String(128))
    event_type: Mapped[str] = mapped_column(String(64))
    input_hash: Mapped[str] = mapped_column(String(64))
    output_hash: Mapped[str] = mapped_column(String(64))
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    policy_violations: Mapped[list[str]] = mapped_column(JSONB, default=list)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(512))
    source_uri: Mapped[str] = mapped_column(String(1024))


class ChunkRow(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id"))
    text: Mapped[str] = mapped_column(Text())
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class EvalRunRow(Base):
    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    label: Mapped[str] = mapped_column(String(128), default="full")
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    summary: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    human_readable_report: Mapped[str] = mapped_column(Text(), default="")


class EvalCaseResultRow(Base):
    __tablename__ = "eval_case_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    eval_run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("eval_runs.id"))
    case_id: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(32))
    trace_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    scores: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)


class PromptRewriteProposalRow(Base):
    __tablename__ = "prompt_rewrite_proposals"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    eval_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("eval_runs.id"), nullable=True)
    prompt_key: Mapped[str] = mapped_column(String(128))
    proposed_text: Mapped[str] = mapped_column(Text())
    diff: Mapped[str] = mapped_column(Text())
    justification: Mapped[str] = mapped_column(Text())
    status: Mapped[str] = mapped_column(String(32), default="pending")


class PromptReviewActionRow(Base):
    __tablename__ = "prompt_review_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    proposal_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("prompt_rewrite_proposals.id"))
    decision: Mapped[str] = mapped_column(String(16))
    reviewer_note: Mapped[str] = mapped_column(Text(), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ApprovedPromptVersionRow(Base):
    __tablename__ = "approved_prompt_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prompt_key: Mapped[str] = mapped_column(String(128), index=True)
    text: Mapped[str] = mapped_column(Text())
    approved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    proposal_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("prompt_rewrite_proposals.id"), nullable=True
    )
