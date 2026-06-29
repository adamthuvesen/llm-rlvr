"""Multi-turn GRPO provenance.

Extends the Stage-2 provenance contract to the multi-turn run: everything Stage 2 records
(base + warm-start + GRPO hyperparams + versions + git) plus the pieces unique to this stage —
the environment config (turn budget, feedback formatting) and the multi-turn reward config
(credit scheme, blend weights, SCoRe-guard switches). Per the multiturn-training spec, this must
suffice to reconstruct the exact environment, reward, warm-start, and versions of a checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from llm_rlvr_sql.env.text2sql_env import FeedbackConfig
from llm_rlvr_sql.rewards.multiturn import MultiTurnRewardConfig
from llm_rlvr_sql.train.grpo import GrpoHyperparams, GrpoProvenance, build_grpo_provenance
from llm_rlvr_sql.train.stage3_provenance import Stage3GrpoProvenance


class MultiTurnProvenance(Stage3GrpoProvenance):
    """Everything needed to reproduce a multi-turn GRPO checkpoint."""

    grpo: GrpoProvenance
    rollout: Literal["multiturn"] = "multiturn"
    # Multi-turn prompts depart from the frozen single-turn `cot` prompt: the instruction
    # must offer the run_sql protocol or the policy never calls it.
    prompt_style: str = "cot"


def build_multiturn_provenance(
    *,
    base_model: str,
    warm_start_adapter: str | Path,
    hyperparams: GrpoHyperparams,
    environment: FeedbackConfig,
    multiturn_reward: MultiTurnRewardConfig,
    dataset: str,
    dataset_revision: str,
    n_train: int,
    n_dev: int,
    prompt_style: str = "cot",
    base_model_revision: str | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> MultiTurnProvenance:
    """Assemble multi-turn checkpoint provenance on top of the Stage-2 builder."""
    return MultiTurnProvenance(
        prompt_style=prompt_style,
        grpo=build_grpo_provenance(
            base_model=base_model,
            warm_start_adapter=warm_start_adapter,
            hyperparams=hyperparams,
            reward=multiturn_reward.base,
            dataset=dataset,
            dataset_revision=dataset_revision,
            n_train=n_train,
            n_dev=n_dev,
            base_model_revision=base_model_revision,
            commit=commit,
            dirty=dirty,
        ),
        environment=environment,
        multiturn_reward=multiturn_reward,
    )
