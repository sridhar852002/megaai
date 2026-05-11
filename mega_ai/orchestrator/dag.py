"""Topological task ordering for decomposition DAGs."""

from __future__ import annotations

from typing import Any


def task_execution_order(tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Returns ordered runnable tasks and any cycle violations."""

    by_id = {t["id"]: t for t in tasks}
    completed: set[str] = set()
    order: list[dict[str, Any]] = []
    violations: list[str] = []

    while len(completed) < len(by_id):
        progressed = False
        for tid, task in by_id.items():
            if tid in completed:
                continue
            deps = list(task.get("depends_on", []))
            if all(d in completed for d in deps):
                order.append(task)
                completed.add(tid)
                progressed = True
        if not progressed:
            remaining = [tid for tid in by_id if tid not in completed]
            violations.append(f"cycle_or_missing_dependency:{remaining}")
            break
    return order, violations
