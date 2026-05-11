"""Shared orchestration context — the only inter-agent communication surface."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    agent_id: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResultRecord(BaseModel):
    tool_name: str
    call_id: str
    success: bool
    structured: dict[str, Any]
    latency_ms: float
    accepted: bool | None = None
    retry_index: int = 0


class CitationRecord(BaseModel):
    chunk_id: str
    document_id: str
    text_span: str
    used_for_claim: str


class ProvenanceEntry(BaseModel):
    sentence: str
    source_agent: str
    source_chunk_id: str | None = None
    source_tool_call_id: str | None = None


class TokenUsage(BaseModel):
    agent_id: str
    turn_id: str
    prompt_tokens: int
    completion_tokens: int


class PolicyViolation(BaseModel):
    kind: str
    agent_id: str
    detail: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RoutingDecision(BaseModel):
    decision_id: str
    chosen_agents: list[str]
    order: list[str]
    reason: str
    confidence: float
    context_budget_tokens: dict[str, int]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CritiqueSpan(BaseModel):
    target_agent: str
    claim_span: str
    confidence: float
    disagree: bool
    suggested_correction: str | None = None


class CritiqueResult(BaseModel):
    critique_id: str
    source_output_agent: str
    spans: list[CritiqueSpan]
    overall_notes: str


class EvalScoreDimension(BaseModel):
    dimension: str
    score: float
    justification: str


class SharedContext(BaseModel):
    job_id: str
    session_id: str | None = None
    user_query: str

    messages: list[AgentMessage] = Field(default_factory=list)
    tool_results: list[ToolResultRecord] = Field(default_factory=list)
    citations: list[CitationRecord] = Field(default_factory=list)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
    token_usage: list[TokenUsage] = Field(default_factory=list)
    policy_violations: list[PolicyViolation] = Field(default_factory=list)
    routing_decisions: list[RoutingDecision] = Field(default_factory=list)
    critique_results: list[CritiqueResult] = Field(default_factory=list)
    eval_scores: list[EvalScoreDimension] = Field(default_factory=list)

    decomposition_graph: dict[str, Any] | None = None
    retrieval_scratchpad: dict[str, Any] | None = None

    model_config = {"arbitrary_types_allowed": False}

    def append_message(self, msg: AgentMessage) -> None:
        self.messages.append(msg)

    def record_routing(self, decision: RoutingDecision) -> None:
        self.routing_decisions.append(decision)

    def add_violation(self, violation: PolicyViolation) -> None:
        self.policy_violations.append(violation)
