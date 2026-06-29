"""Task-agnostic RLVR harness.

The reusable core of this project, factored out of the text-to-SQL specifics: a *verifiable
reward* is a deterministic program that decides whether a model's answer is correct, and the same
program is used both as the GRPO training reward and as the offline evaluation scorer. Everything
in this package is independent of SQL — a new task plugs in by implementing two small protocols
(`AnswerParser`, `Verifier`) and bundling them into a `Task`.

  - `AnswerParser`  — pull the answer out of a raw completion, and judge its format.
  - `Verifier`      — execute/check the answer against a reference; return a `Verdict`.
  - `RewardConfig`  — generic shaping (format / valid / matched / length); arm selection.
  - `score_answer`  — compose a parser + verifier + shaping into one per-completion reward.
  - `Task`          — the full extension surface a concrete task implements (see `docs/harness.md`).

Text-to-SQL is the reference task (`llm_rlvr_sql.tasks.sql`); see `docs/harness.md` for how to add
your own.
"""

from __future__ import annotations

from llm_rlvr_sql.harness.answer import AnswerParser
from llm_rlvr_sql.harness.reward import (
    RewardArm,
    RewardBreakdown,
    RewardConfig,
    completion_text,
    score_answer,
)
from llm_rlvr_sql.harness.task import Task
from llm_rlvr_sql.harness.verifier import ReferenceUnverifiable, Verdict, Verifier

__all__ = [
    "AnswerParser",
    "ReferenceUnverifiable",
    "RewardArm",
    "RewardBreakdown",
    "RewardConfig",
    "Task",
    "Verdict",
    "Verifier",
    "completion_text",
    "score_answer",
]
