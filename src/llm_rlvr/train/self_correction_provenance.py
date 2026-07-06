"""Shared provenance contract for self-correction GRPO analysis runs."""

from __future__ import annotations

from typing import Literal

from llm_rlvr._base import Frozen
from llm_rlvr.env.text2sql_env import FeedbackConfig
from llm_rlvr.rewards.multiturn import MultiTurnRewardConfig
from llm_rlvr.train.grpo import GrpoProvenance

SelfCorrectionRollout = Literal["multiturn", "forced_revision"]


class SelfCorrectionGrpoProvenance(Frozen):
    """Common provenance for GRPO runs that add an environment and trajectory reward."""

    grpo: GrpoProvenance
    environment: FeedbackConfig
    multiturn_reward: MultiTurnRewardConfig
    rollout: SelfCorrectionRollout
    prompt_style: str = "cot"
