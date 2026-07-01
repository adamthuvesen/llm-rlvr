"""Forced-revision GRPO smoke: mask + IS-ratio assertions before paid runs."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from llm_rlvr._base import Frozen
from llm_rlvr.data.models import DatabaseSchema, Example
from llm_rlvr.env.text2sql_env import FeedbackConfig
from llm_rlvr.eval.completions import write_json
from llm_rlvr.revision.train import (
    ForcedRevisionProvenance,
    _EpisodeLog,
    build_forced_revision_provenance,
    build_trainer_pieces,
)
from llm_rlvr.rewards.multiturn import MultiTurnRewardConfig
from llm_rlvr.train.diagnostics import CollapseDiagnostics, compute_diagnostics
from llm_rlvr.train.grpo import (
    GrpoHyperparams,
    build_grpo_config_from_kwargs,
    grpo_config_kwargs,
    load_trainable_policy,
)
from llm_rlvr.train.grpo_smoke import (
    IsRatioCheck,
    MaskCheck,
    capturing_grpo_trainer,
    check_is_ratio,
    check_tool_masking,
    logged_mean,
)

SMOKE_REPORT_FILENAME = "smoke-report.json"


def count_nan_log_values(log_history: Sequence[Mapping[str, object]]) -> int:
    count = 0
    for entry in log_history:
        for key in ("loss", "grad_norm"):
            value = entry.get(key)
            if isinstance(value, float) and math.isnan(value):
                count += 1
    return count


def fr_smoke_failures(
    *,
    n_episodes: int,
    mask: MaskCheck,
    is_ratio: IsRatioCheck,
    nan_log_values: int,
    none_logprob_tokens: int,
) -> tuple[str, ...]:
    failures: list[str] = []
    if n_episodes == 0:
        failures.append("no episodes were scored — the rollout never reached the reward")
    if mask.rows_with_masked_tokens < mask.n_rows:
        failures.append(
            f"only {mask.rows_with_masked_tokens}/{mask.n_rows} rollout rows carry masked "
            "tokens — forced revision must splice feedback into every episode"
        )
    if mask.masked_tokens_total == 0:
        failures.append("no completion token was feedback-masked")
    if not mask.feedback_in_masked_text:
        failures.append("env feedback never appeared in the masked (loss-excluded) spans")
    if mask.feedback_in_unmasked_text:
        failures.append("env feedback leaked into gradient-carrying tokens")
    if not is_ratio.in_band:
        failures.append(
            f"importance-sampling ratio outside the expected band {is_ratio.band}: "
            f"means avg {is_ratio.mean_of_means}, n_logged {is_ratio.n_logged}"
        )
    if nan_log_values > 0:
        failures.append(f"{nan_log_values} NaN loss/grad_norm values in the trainer log")
    if none_logprob_tokens > 0:
        failures.append(
            f"{none_logprob_tokens} sampled tokens came back with NaN logprobs from vLLM"
        )
    return tuple(failures)


class FrSmokeReport(Frozen):
    passed: bool
    failures: tuple[str, ...]
    n_train: int
    n_steps: int
    wall_time_s: float
    mean_step_time_s: float | None
    n_episodes: int
    n_unparseable_attempts: int
    n_final_sql_missing: int
    mask: MaskCheck
    is_ratio: IsRatioCheck
    nan_log_values: int
    none_logprob_tokens: int
    diagnostics: CollapseDiagnostics | None
    mean_reward: float | None
    peak_torch_memory_gib: float | None
    device_memory_used_gib: float | None
    completions_mean_length: float | None
    provenance: ForcedRevisionProvenance


def run_forced_revision_smoke(
    examples: Sequence[Example],
    schemas: Mapping[str, DatabaseSchema],
    db_root: str | Path,
    output_dir: str | Path,
    *,
    base_model: str,
    warm_start_adapter: str | Path,
    dataset: str,
    dataset_revision: str,
    base_model_revision: str | None = None,
    hyperparams: GrpoHyperparams | None = None,
    environment: FeedbackConfig | None = None,
    multiturn_reward: MultiTurnRewardConfig | None = None,
    max_steps: int | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> Path:
    import torch
    from trl import GRPOTrainer

    hp = hyperparams or GrpoHyperparams()
    env_cfg = environment or FeedbackConfig()
    reward_cfg = multiturn_reward or MultiTurnRewardConfig()

    log = _EpisodeLog()
    train_examples, dev_examples, train_dataset, rollout, reward_fn = build_trainer_pieces(
        examples,
        schemas,
        db_root,
        hyperparams=hp,
        environment=env_cfg,
        reward_config=reward_cfg,
        on_episode=log,
    )

    captured: list[dict[str, list[list[int]]]] = []
    none_logprob_total = 0

    def counting_rollout(prompts, trainer):
        nonlocal none_logprob_total
        output = rollout(prompts, trainer)
        none_logprob_total += sum(output["fr_none_logprob_tokens"])
        return output

    capturing_trainer_cls = capturing_grpo_trainer(GRPOTrainer)
    desired = grpo_config_kwargs(hp, output_dir)
    if max_steps is not None:
        desired["max_steps"] = max_steps

    policy = load_trainable_policy(base_model, base_model_revision, warm_start_adapter)
    torch.cuda.reset_peak_memory_stats()
    trainer = capturing_trainer_cls(
        model=policy,
        reward_funcs=[reward_fn],
        args=build_grpo_config_from_kwargs(desired),
        train_dataset=train_dataset,
        rollout_func=counting_rollout,
        captured=captured,
    )
    started = time.perf_counter()
    trainer.train()
    wall_time_s = time.perf_counter() - started

    tokenizer = trainer.processing_class

    def decode(ids: Sequence[int]) -> str:
        return tokenizer.decode(list(ids), skip_special_tokens=True)

    mask = check_tool_masking(
        [row for batch in captured for row in batch["completion_ids"]],
        [row for batch in captured for row in batch["completion_mask"]],
        [row for batch in captured for row in batch["tool_mask"]],
        decode,
    )
    log_history = trainer.state.log_history
    is_ratio = check_is_ratio(log_history)
    nan_log_values = count_nan_log_values(log_history)

    failures = fr_smoke_failures(
        n_episodes=len(log.episodes),
        mask=mask,
        is_ratio=is_ratio,
        nan_log_values=nan_log_values,
        none_logprob_tokens=none_logprob_total,
    )

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    report = FrSmokeReport(
        passed=not failures,
        failures=failures,
        n_train=len(train_examples),
        n_steps=int(trainer.state.global_step),
        wall_time_s=wall_time_s,
        mean_step_time_s=logged_mean(log_history, "step_time"),
        n_episodes=len(log.episodes),
        n_unparseable_attempts=sum(1 for sqls, _ in log.episodes if sqls[0] is None),
        n_final_sql_missing=sum(1 for sqls, _ in log.episodes if sqls[-1] is None),
        mask=mask,
        is_ratio=is_ratio,
        nan_log_values=nan_log_values,
        none_logprob_tokens=none_logprob_total,
        diagnostics=compute_diagnostics(log.episodes) if log.episodes else None,
        mean_reward=logged_mean(log_history, "reward"),
        peak_torch_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
        device_memory_used_gib=(total_bytes - free_bytes) / 2**30,
        completions_mean_length=logged_mean(log_history, "completions/mean_length"),
        provenance=build_forced_revision_provenance(
            base_model=base_model,
            warm_start_adapter=warm_start_adapter,
            hyperparams=hp,
            environment=env_cfg,
            multiturn_reward=reward_cfg,
            dataset=dataset,
            dataset_revision=dataset_revision,
            n_train=len(train_examples),
            n_dev=len(dev_examples),
            base_model_revision=base_model_revision,
            commit=commit,
            dirty=dirty,
        ),
    )
    out = write_json(report, Path(output_dir) / SMOKE_REPORT_FILENAME)
    if failures:
        raise RuntimeError(
            "forced-revision smoke FAILED — paid runs are blocked: "
            + "; ".join(failures)
            + f" (report: {out})"
        )
    return out
