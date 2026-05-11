"""Token budget accounting per agent per turn."""

from __future__ import annotations

from dataclasses import dataclass


def estimate_tokens(text: str) -> int:
    """Heuristic token estimate when provider usage is unavailable."""
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class BudgetReport:
    agent_id: str
    turn_id: str
    budget_tokens: int
    used_tokens: int

    @property
    def remaining(self) -> int:
        return max(0, self.budget_tokens - self.used_tokens)


class ContextBudgetManager:
    """Tracks declared budgets; triggers compression requests; never silently truncates."""

    def __init__(self) -> None:
        self._used: dict[tuple[str, str], int] = {}

    def remaining(self, agent_id: str, turn_id: str, budget_tokens: int) -> int:
        used = self._used.get((agent_id, turn_id), 0)
        return max(0, budget_tokens - used)

    def check_can_add(
        self,
        *,
        agent_id: str,
        turn_id: str,
        budget_tokens: int,
        addition_tokens: int,
    ) -> bool:
        return self.remaining(agent_id, turn_id, budget_tokens) >= addition_tokens

    def record_usage(self, *, agent_id: str, turn_id: str, tokens: int) -> BudgetReport:
        key = (agent_id, turn_id)
        self._used[key] = self._used.get(key, 0) + tokens
        # Budget is looked up externally; report uses 0 budget if unknown caller
        return BudgetReport(agent_id=agent_id, turn_id=turn_id, budget_tokens=0, used_tokens=self._used[key])

    def snapshot_report(self, *, agent_id: str, turn_id: str, budget_tokens: int) -> BudgetReport:
        used = self._used.get((agent_id, turn_id), 0)
        return BudgetReport(
            agent_id=agent_id,
            turn_id=turn_id,
            budget_tokens=budget_tokens,
            used_tokens=used,
        )
