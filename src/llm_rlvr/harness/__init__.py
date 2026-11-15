"""Task-agnostic RLVR reward shaping.

A *verifiable reward* is a deterministic program that decides whether a model's answer is correct,
and the same program is used both as the GRPO training reward and as the offline evaluation scorer.
Nothing in this package depends on SQL:

  - `Verdict`       — a verifier's outcome (`matched`, `valid`, `reason`).
  - `RewardConfig`  — generic shaping (format / valid / matched / length); arm selection.
  - `score_answer`  — compose an answer extractor + format check + verifier into one reward.

Text-to-SQL (`llm_rlvr.tasks.sql`) is the one task; see `docs/harness.md`.
"""

from __future__ import annotations

from llm_rlvr.harness.reward import (
    RewardArm,
    RewardBreakdown,
    RewardConfig,
    completion_text,
    score_answer,
)
from llm_rlvr.harness.verifier import ReferenceUnverifiable, Verdict

__all__ = [
    "ReferenceUnverifiable",
    "RewardArm",
    "RewardBreakdown",
    "RewardConfig",
    "Verdict",
    "completion_text",
    "score_answer",
]
