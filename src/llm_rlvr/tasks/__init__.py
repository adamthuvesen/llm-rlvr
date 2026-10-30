"""Concrete RLVR tasks, keyed by name.

New tasks are registered here for use with the shared GRPO driver and harness reward. Text-to-SQL
is the reference task. Each value is a zero-argument factory that returns a `harness.Task`; see
`docs/harness.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm_rlvr.harness import Task
from llm_rlvr.tasks.sql import sql_task

TASKS: dict[str, Callable[[], Task[Any, Any, Any]]] = {
    "sql": sql_task,
}


def get_task(name: str) -> Task[Any, Any, Any]:
    """Return the registered task `name`, or raise with the available names."""
    try:
        factory = TASKS[name]
    except KeyError:
        raise ValueError(f"unknown task {name!r}; registered: {sorted(TASKS)}") from None
    return factory()


__all__ = ["TASKS", "get_task"]
