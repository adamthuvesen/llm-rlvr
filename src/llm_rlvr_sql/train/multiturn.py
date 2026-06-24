"""Multi-turn GRPO wiring + the de-risk smoke (Stage 3, design D1/D5, tasks 2.1/2.2).

TRL v1.5.1 runs the tool loop natively: `GRPOTrainer(environment_factory=...)` exposes
`SqliteText2SqlEnv.run_sql` as a model-callable tool, owns the regenerate-after-feedback loop
(bounded by `GRPOConfig.max_tool_calling_iterations`), masks tool-output tokens via `tool_mask`,
and hands the env instances to reward functions through the `environments=` kwarg.

Two things here, per the repo's builder/driver split:

  - Pure, CPU-testable pieces: the config kwargs, the TRL-facing trajectory reward fn, and the
    smoke's two checks — `check_tool_masking` (no execution-feedback token may carry gradient)
    and `check_is_ratio` (the TRL #4543 guard: the vLLM importance-sampling ratio must sit in
    the band the Stage-2 smoke measured, mean 1.02).
  - `run_multiturn_smoke`: the GPU driver Modal calls. It trains a few real steps, captures the
    trainer's masked batches, and writes a `MultiTurnSmokeReport` JSON — then fails loudly if
    any assertion failed, AFTER persisting the evidence (a failed smoke blocks paid runs and
    invokes the design-D5 fallback ladder; the report is the record either way).

KL-anchor API truth (design Open Question, resolved by reading the tagged v1.5.1 source): when
the policy is a PeftModel with a pretrained adapter and `beta != 0`, the trainer snapshots that
adapter as a frozen `"ref"` adapter and computes the KL against it — i.e. our stock `beta=0.04`
KL anchors the *whole completion* to the Stage-2 warm-start policy. A turn-1-span-only anchor
would need a custom loss; we start with the stock whole-completion anchor and only build the
span-restricted variant if pilot diagnostics show turn-1 sandbagging despite it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from llm_rlvr_sql._base import Frozen
from llm_rlvr_sql.data.models import DatabaseSchema, Example
from llm_rlvr_sql.data.spider import resolve_database
from llm_rlvr_sql.env.text2sql_env import FeedbackConfig, make_environment_factory
from llm_rlvr_sql.eval.completions import write_json
from llm_rlvr_sql.eval.parsing import extract_sql
from llm_rlvr_sql.rewards.multiturn import (
    MultiTurnRewardConfig,
    TrajectoryBreakdown,
    score_trajectory,
)
from llm_rlvr_sql.rewards.reward import _completion_text
from llm_rlvr_sql.train.diagnostics import CollapseDiagnostics, compute_diagnostics
from llm_rlvr_sql.train.grpo import (
    GrpoHyperparams,
    build_grpo_config_from_kwargs,
    build_grpo_rows,
    build_lora_config,
    grpo_config_kwargs,
    split_train_dev,
)
from llm_rlvr_sql.train.multiturn_provenance import (
    MultiTurnProvenance,
    build_multiturn_provenance,
)

SMOKE_REPORT_FILENAME = "smoke-report.json"

# Every env observation starts with this marker (`format_feedback` / the budget message), so the
# mask check can decide which side of the loss mask the feedback text landed on. A unit test
# pins the marker to the env's actual output — if the wording drifts, the test screams.
FEEDBACK_MARKER = "Execution feedback"

# The Stage-2 colocate smoke measured `sampling/importance_sampling_ratio` mean 1.02
# (min 0.43 / max 2.25) — the healthy band for the TRL #4543 check. A multi-turn mean outside
# this band means the trainer/vLLM logprobs disagree on the spliced transcripts: stop, fall back.
IS_RATIO_HEALTHY_BAND = (0.8, 1.25)


class MultiTurnHyperparams(GrpoHyperparams):
    """Stage-2 GRPO budget + the tool-loop bound (= the env's own turn budget, defense in depth)."""

    max_tool_calling_iterations: int = 3


def multiturn_grpo_config_kwargs(
    hyperparams: MultiTurnHyperparams, output_dir: str | Path
) -> dict[str, object]:
    """The Stage-2 GRPOConfig kwargs + the tool-loop bound."""
    return {
        **grpo_config_kwargs(hyperparams, output_dir),
        "max_tool_calling_iterations": hyperparams.max_tool_calling_iterations,
    }


def make_multiturn_reward_fn(
    db_root: str | Path,
    *,
    config: MultiTurnRewardConfig | None = None,
    resolver: Callable[[str, str | Path], Path] = resolve_database,
    on_episode: Callable[[list[str | None], TrajectoryBreakdown], None] | None = None,
) -> Callable[..., list[float]]:
    """Wrap `score_trajectory` in the TRL reward convention for environment rollouts.

    The trainer passes `environments=` (one env per rollout slot, aligned with `completions`);
    each env's recorded `turns` are the episode's tool queries, the completion is the final
    answer. `on_episode` (optional) sees every episode's SQLs + breakdown — the smoke and the
    diagnostics logging hang off it without touching reward values.
    """
    cfg = config or MultiTurnRewardConfig()

    def reward(completions, gold_sql, db_id, environments=None, **_kwargs) -> list[float]:
        if environments is None:
            raise ValueError(
                "multi-turn reward needs the trainer's environments= kwarg; "
                "was GRPOTrainer built with environment_factory?"
            )
        rewards: list[float] = []
        for completion, gold, dbid, env in zip(
            completions, gold_sql, db_id, environments, strict=True
        ):
            text = _completion_text(completion)
            queries = [turn.query for turn in env.turns]
            breakdown = score_trajectory(queries, text, gold, resolver(dbid, db_root), config=cfg)
            if on_episode is not None:
                on_episode([*queries, extract_sql(text)], breakdown)
            rewards.append(breakdown.total)
        return rewards

    reward.__name__ = f"multiturn_reward_{cfg.credit}"
    return reward


class MaskCheck(Frozen):
    """Did the loss mask exclude every execution-feedback token? (multiturn-training spec.)"""

    n_rows: int
    rows_with_masked_tokens: int
    masked_tokens_total: int
    feedback_in_masked_text: bool  # the env feedback must show up on the masked side…
    feedback_in_unmasked_text: bool  # …and never on the gradient-carrying side


def check_tool_masking(
    completion_ids: Sequence[Sequence[int]],
    completion_mask: Sequence[Sequence[int]],
    tool_mask: Sequence[Sequence[int]],
    decode: Callable[[Sequence[int]], str],
) -> MaskCheck:
    """Pure check over captured rollout batches: tool-output tokens (tool_mask=0) vs the rest.

    Padding is gated out via `completion_mask`; `decode` is the tokenizer's id→text (the GPU
    driver passes the real one, tests pass a fake).
    """
    rows_with_masked = 0
    masked_total = 0
    masked_texts: list[str] = []
    unmasked_texts: list[str] = []
    for ids, cmask, tmask in zip(completion_ids, completion_mask, tool_mask, strict=True):
        masked = [i for i, c, t in zip(ids, cmask, tmask, strict=True) if c == 1 and t == 0]
        unmasked = [i for i, c, t in zip(ids, cmask, tmask, strict=True) if c == 1 and t == 1]
        if masked:
            rows_with_masked += 1
            masked_total += len(masked)
            masked_texts.append(decode(masked))
        unmasked_texts.append(decode(unmasked))
    return MaskCheck(
        n_rows=len(completion_ids),
        rows_with_masked_tokens=rows_with_masked,
        masked_tokens_total=masked_total,
        feedback_in_masked_text=any(FEEDBACK_MARKER in text for text in masked_texts),
        feedback_in_unmasked_text=any(FEEDBACK_MARKER in text for text in unmasked_texts),
    )


class IsRatioCheck(Frozen):
    """The TRL #4543 guard: logged vLLM importance-sampling ratios vs the Stage-2 healthy band."""

    n_logged: int
    mean_of_means: float | None
    overall_min: float | None
    overall_max: float | None
    logp_diff_mean: float | None
    band: tuple[float, float]
    in_band: bool  # every logged step mean inside the band; False when nothing was logged


def check_is_ratio(
    log_history: Sequence[Mapping[str, object]],
    *,
    band: tuple[float, float] = IS_RATIO_HEALTHY_BAND,
) -> IsRatioCheck:
    """Collect `sampling/importance_sampling_ratio/*` from the trainer log history and judge it."""
    means = [
        float(entry["sampling/importance_sampling_ratio/mean"])
        for entry in log_history
        if "sampling/importance_sampling_ratio/mean" in entry
    ]
    mins = [
        float(entry["sampling/importance_sampling_ratio/min"])
        for entry in log_history
        if "sampling/importance_sampling_ratio/min" in entry
    ]
    maxes = [
        float(entry["sampling/importance_sampling_ratio/max"])
        for entry in log_history
        if "sampling/importance_sampling_ratio/max" in entry
    ]
    logp_diffs = [
        float(entry["sampling/sampling_logp_difference/mean"])
        for entry in log_history
        if "sampling/sampling_logp_difference/mean" in entry
    ]
    low, high = band
    return IsRatioCheck(
        n_logged=len(means),
        mean_of_means=(sum(means) / len(means)) if means else None,
        overall_min=min(mins) if mins else None,
        overall_max=max(maxes) if maxes else None,
        logp_diff_mean=(sum(logp_diffs) / len(logp_diffs)) if logp_diffs else None,
        band=band,
        in_band=bool(means) and all(low <= m <= high for m in means),
    )


class MultiTurnSmokeReport(Frozen):
    """The smoke's full evidence: assertions, tool-use census, envelope, and provenance."""

    passed: bool
    failures: tuple[str, ...]
    n_train: int
    n_steps: int
    wall_time_s: float
    mean_step_time_s: float | None
    n_episodes: int
    episodes_with_tool_calls: int
    episodes_with_2plus_tool_turns: int
    max_tool_turns: int
    tool_call_frequency: float | None  # trainer-logged tools/call_frequency mean
    tool_failure_frequency: float | None
    mask: MaskCheck
    is_ratio: IsRatioCheck
    diagnostics: CollapseDiagnostics | None  # None when no episodes were scored
    peak_torch_memory_gib: float | None
    device_memory_used_gib: float | None  # torch + vLLM colocate, from the device
    completions_mean_length: float | None
    provenance: MultiTurnProvenance


def _logged_mean(log_history: Sequence[Mapping[str, object]], key: str) -> float | None:
    values = [float(entry[key]) for entry in log_history if key in entry]
    return (sum(values) / len(values)) if values else None


def smoke_failures(
    *,
    episodes_with_tool_calls: int,
    episodes_with_2plus_tool_turns: int,
    mask: MaskCheck,
    is_ratio: IsRatioCheck,
) -> tuple[str, ...]:
    """The D5 verdict, pure: which smoke assertions failed (empty = paid runs may proceed)."""
    failures: list[str] = []
    if episodes_with_tool_calls == 0:
        failures.append("no episode called run_sql — the tool loop never fired")
    if episodes_with_2plus_tool_turns == 0:
        failures.append("no episode used >=2 tool turns — the spec needs a real 2-turn rollout")
    if mask.masked_tokens_total == 0:
        failures.append("no completion token was tool-masked")
    if not mask.feedback_in_masked_text:
        failures.append("env feedback never appeared in the masked (loss-excluded) spans")
    if mask.feedback_in_unmasked_text:
        failures.append("env feedback leaked into gradient-carrying tokens")
    if not is_ratio.in_band:
        failures.append(
            f"importance-sampling ratio outside the Stage-2 band {is_ratio.band}: "
            f"means avg {is_ratio.mean_of_means}, n_logged {is_ratio.n_logged}"
        )
    return tuple(failures)


def run_multiturn_smoke(
    examples: Sequence[Example],
    schemas: Mapping[str, DatabaseSchema],
    db_root: str | Path,
    output_dir: str | Path,
    *,
    base_model: str,
    warm_start_adapter: str | Path | None,
    dataset: str,
    dataset_revision: str,
    base_model_revision: str | None = None,
    hyperparams: MultiTurnHyperparams | None = None,
    environment: FeedbackConfig | None = None,
    multiturn_reward: MultiTurnRewardConfig | None = None,
    prompt_style: str = "mt_demo",
    max_steps: int | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> Path:
    """Run the de-risk smoke (tasks 2.1/2.2) on the GPU; persist `MultiTurnSmokeReport` JSON.

    Trains a few real multi-turn GRPO steps from the Stage-2 adapter, capturing every rollout
    batch's `tool_mask`, then asserts the D5 conditions. Raises `RuntimeError` on a failed
    assertion — AFTER writing the report, so the evidence survives the non-zero exit.
    """
    import torch
    from datasets import Dataset  # lazy: heavy training stack, train extra only
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from trl import GRPOTrainer

    hp = hyperparams or MultiTurnHyperparams()
    env_cfg = environment or FeedbackConfig()
    reward_cfg = multiturn_reward or MultiTurnRewardConfig()
    # Defense in depth (design D2) only works if both budgets agree; provenance records the
    # env's max_turns, so a silent divergence here would be invisible later. Refuse it.
    if hp.max_tool_calling_iterations != env_cfg.max_turns:
        raise ValueError(
            f"max_tool_calling_iterations ({hp.max_tool_calling_iterations}) must equal the "
            f"environment turn budget ({env_cfg.max_turns})"
        )

    train_examples, dev_examples = split_train_dev(
        examples, dev_size=hp.eval_slice_size, seed=hp.seed
    )
    train_dataset = Dataset.from_list(build_grpo_rows(train_examples, schemas, style=prompt_style))

    episodes: list[tuple[list[str | None], TrajectoryBreakdown]] = []
    reward_fn = make_multiturn_reward_fn(
        db_root,
        config=reward_cfg,
        on_episode=lambda sqls, breakdown: episodes.append((sqls, breakdown)),
    )

    captured: list[dict[str, list[list[int]]]] = []

    class _CapturingTrainer(GRPOTrainer):
        """Stock trainer + a tap on the scored rollout batches (the mask evidence)."""

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            if "tool_mask" in output:
                captured.append(
                    {
                        key: output[key].detach().cpu().tolist()
                        for key in ("completion_ids", "completion_mask", "tool_mask")
                    }
                )
            return output

    desired = multiturn_grpo_config_kwargs(hp, output_dir)
    if max_steps is not None:
        desired["max_steps"] = max_steps

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype="bfloat16",
        **({"revision": base_model_revision} if base_model_revision else {}),
    )
    if warm_start_adapter is not None:
        # Warm-start from the Stage-2 adapter; with beta != 0 TRL snapshots it as the frozen
        # "ref" adapter, so the KL anchors to the Stage-2 policy (the D4 anchor).
        policy = PeftModel.from_pretrained(base, str(warm_start_adapter), is_trainable=True)
    else:
        # Fresh LoRA: zero-init B matrices make this an identity at step 0, so the first
        # rollouts sample the BASE policy exactly — the entrenchment-probe arm.
        from peft import get_peft_model

        policy = get_peft_model(base, build_lora_config(hp))

    torch.cuda.reset_peak_memory_stats()
    trainer = _CapturingTrainer(
        model=policy,
        reward_funcs=[reward_fn],
        args=build_grpo_config_from_kwargs(desired),
        train_dataset=train_dataset,
        environment_factory=make_environment_factory(db_root, config=env_cfg),
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

    tool_turn_counts = [len(sqls) - 1 for sqls, _ in episodes]  # last entry is the final answer
    with_tools = sum(1 for n in tool_turn_counts if n >= 1)
    with_two_plus = sum(1 for n in tool_turn_counts if n >= 2)
    failures = smoke_failures(
        episodes_with_tool_calls=with_tools,
        episodes_with_2plus_tool_turns=with_two_plus,
        mask=mask,
        is_ratio=is_ratio,
    )

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    report = MultiTurnSmokeReport(
        passed=not failures,
        failures=failures,
        n_train=len(train_examples),
        n_steps=int(trainer.state.global_step),
        wall_time_s=wall_time_s,
        mean_step_time_s=_logged_mean(log_history, "step_time"),
        n_episodes=len(episodes),
        episodes_with_tool_calls=with_tools,
        episodes_with_2plus_tool_turns=with_two_plus,
        max_tool_turns=max(tool_turn_counts, default=0),
        tool_call_frequency=_logged_mean(log_history, "tools/call_frequency"),
        tool_failure_frequency=_logged_mean(log_history, "tools/failure_frequency"),
        mask=mask,
        is_ratio=is_ratio,
        diagnostics=compute_diagnostics(episodes) if episodes else None,
        peak_torch_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
        device_memory_used_gib=(total_bytes - free_bytes) / 2**30,
        completions_mean_length=_logged_mean(log_history, "completions/mean_length"),
        provenance=build_multiturn_provenance(
            base_model=base_model,
            warm_start_adapter=(
                warm_start_adapter
                if warm_start_adapter is not None
                else "none (fresh identity LoRA from base — entrenchment probe)"
            ),
            # Down-cast to the declared provenance type; the tool-loop bound it would lose is
            # pinned to environment.max_turns by the equality check above.
            hyperparams=GrpoHyperparams.model_validate(
                hp.model_dump(exclude={"max_tool_calling_iterations"})
            ),
            environment=env_cfg,
            multiturn_reward=reward_cfg,
            dataset=dataset,
            dataset_revision=dataset_revision,
            n_train=len(train_examples),
            n_dev=len(dev_examples),
            prompt_style=prompt_style,
            base_model_revision=base_model_revision,
            commit=commit,
            dirty=dirty,
        ),
    )
    out = write_json(report, Path(output_dir) / SMOKE_REPORT_FILENAME)
    if failures:
        raise RuntimeError(
            "multi-turn smoke FAILED — paid runs are blocked (design-D5 fallback ladder): "
            + "; ".join(failures)
            + f" (report: {out})"
        )
    return out
