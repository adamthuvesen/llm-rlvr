"""Concrete RLVR tasks, keyed by name.

Text-to-SQL is the reference task; register your own here to reuse the shared SFT/GRPO/eval
machinery (see `docs/harness.md`). Each value is a zero-arg factory returning a `harness.Task`.
"""

from __future__ import annotations

from collections.abc import Callable

from llm_rlvr_sql.harness import Task
from llm_rlvr_sql.tasks.sql import SqlTask, sql_task

TASKS: dict[str, Callable[[], Task]] = {
    "sql": sql_task,
}


def get_task(name: str) -> Task:
    """Return the registered task `name`, or raise with the available names."""
    try:
        factory = TASKS[name]
    except KeyError:
        raise ValueError(f"unknown task {name!r}; registered: {sorted(TASKS)}") from None
    return factory()


__all__ = ["TASKS", "SqlTask", "get_task", "sql_task"]
