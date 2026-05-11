"""Hand-rolled scoring dimensions with explicit textual justifications."""

from __future__ import annotations

from dataclasses import dataclass

from mega_ai.core.context import EvalScoreDimension, SharedContext


@dataclass(frozen=True)
class ScoreReport:
    dimensions: list[EvalScoreDimension]
    failed: bool


def _has_substring(haystack: str, needle: str | None) -> bool:
    if needle is None:
        return True
    return needle.lower() in haystack.lower()


def score_case(
    *,
    final_answer: str,
    context: SharedContext,
    expected: str | None,
    category: str,
) -> ScoreReport:
    violations = context.policy_violations
    tool_calls = context.tool_results

    correctness = 0.0
    if expected:
        correctness = 1.0 if _has_substring(final_answer, expected) else 0.2
    else:
        correctness = 0.6 if final_answer.strip() else 0.0
    correctness_just = (
        f"Compared model answer against expected hint '{expected}' for category {category}."
        if expected
        else "No deterministic gold answer; scored completeness instead."
    )

    cited_chunks = {c.chunk_id for c in context.citations}
    citation_score = 1.0 if len(cited_chunks) >= 2 else 0.4
    citation_just = f"Unique cited chunks observed: {len(cited_chunks)}."

    contradictions = [s for c in context.critique_results for s in c.spans if s.disagree]
    resolution_score = 1.0 - min(1.0, len(contradictions) * 0.25)
    if "London" in final_answer and "France" in final_answer:
        resolution_score = min(resolution_score, 0.4)
    resolution_just = f"Critique disagreements recorded: {len(contradictions)}; synthesis reconciliation inferred."

    unnecessary_penalty = max(0.0, len(tool_calls) - 12)
    tool_score = max(0.0, 1.0 - unnecessary_penalty * 0.05)
    tool_just = f"Tool calls observed: {len(tool_calls)} (penalize runaway usage)."

    budget_violations = [v for v in violations if "context_budget" in v.kind]
    budget_score = 1.0 if not budget_violations else 0.3
    budget_just = f"Budget policy violations: {len(budget_violations)}."

    critique_agree = 0.0
    if context.critique_results:
        spans = [s for c in context.critique_results for s in c.spans]
        if spans:
            agree = sum(1 for s in spans if not s.disagree)
            critique_agree = agree / len(spans)
    critique_just = "Share of critique spans not marked disagree."

    dims = [
        EvalScoreDimension(dimension="answer_correctness", score=correctness, justification=correctness_just),
        EvalScoreDimension(dimension="citation_accuracy", score=citation_score, justification=citation_just),
        EvalScoreDimension(dimension="contradiction_resolution", score=resolution_score, justification=resolution_just),
        EvalScoreDimension(dimension="tool_efficiency", score=tool_score, justification=tool_just),
        EvalScoreDimension(dimension="context_budget_compliance", score=budget_score, justification=budget_just),
        EvalScoreDimension(dimension="critique_agreement_rate", score=critique_agree, justification=critique_just),
    ]
    failed = any(d.score < 0.55 for d in dims)
    return ScoreReport(dimensions=dims, failed=failed)
