"""Single-turn GRPO LoRA training.

TRL `GRPOTrainer` + PEFT LoRA, bf16, vLLM colocate rollouts scored by the execution-grounded reward
(`llm_rlvr.tasks.sql`). The policy is warm-started from the SFT adapter, loaded as a
trainable PEFT model so the saved adapter is the SFT→GRPO continuation, evaluated as a single
adapter over the base (the adapter-eval path needs no change). The base model is config-parametric
(Qwen default; Llama-3.2-3B for the cross-family control).

As in `sft.py`, the kwargs builders are pure (no trl/peft import) so hyperparameters are
unit-testable on CPU; `run_grpo` is the GPU-only driver Modal calls. The `GRPOConfig` /
`LoraConfig` kwargs are validated against the pinned TRL/vLLM pair on Modal, not at unit-test time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import Field

from llm_rlvr._base import Frozen
from llm_rlvr.artifacts.provenance import git_commit, git_is_dirty, library_versions
from llm_rlvr.data.models import EXAMPLE_IDENTITY_POLICY, example_identity
from llm_rlvr.harness import RewardConfig as ShapingConfig
from llm_rlvr.harness import Task
from llm_rlvr.tasks.sql import RewardConfig, sql_task
from llm_rlvr.train.configuration import (
    AdapterIdentity,
    RuntimeConfigRecord,
    build_training_config,
)
from llm_rlvr.train.prepared import execute_prepared_grpo, publish_training_run

_TRAIN_LIBRARIES = ("llm-rlvr", "trl", "peft", "transformers", "torch", "accelerate", "vllm")

PROVENANCE_FILENAME = "provenance.json"
T = TypeVar("T")


class GrpoHyperparams(Frozen):
    """Single-A100 colocate GRPO config."""

    # Legacy requested values. GRPO continues the warm-start adapter's actual LoRA configuration.
    lora_rank: int = Field(default=32, gt=0, strict=True)
    lora_alpha: int = Field(default=64, gt=0, strict=True)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1, allow_inf_nan=False)
    # GRPO LoRA LR: between SQL-R1's low full-finetune LR and common LoRA LRs.
    learning_rate: float = Field(default=1e-5, gt=0, allow_inf_nan=False)
    beta: float = Field(default=0.04, ge=0, allow_inf_nan=False)
    num_generations: int = Field(default=4, ge=2, strict=True)
    max_completion_length: int = Field(default=1024, gt=0, strict=True)
    temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    per_device_batch_size: int = Field(default=8, gt=0, strict=True)
    gradient_accumulation_steps: int = Field(default=4, gt=0, strict=True)
    num_train_epochs: int = Field(default=1, gt=0, strict=True)
    warmup_ratio: float = Field(default=0.03, ge=0, le=1, allow_inf_nan=False)
    vllm_gpu_memory_utilization: float = Field(default=0.3, gt=0, lt=1, allow_inf_nan=False)
    save_steps: int = Field(default=50, gt=0, strict=True)
    eval_slice_size: int = Field(default=500, ge=0, strict=True)
    seed: int = 0


def grpo_config_kwargs(hyperparams: GrpoHyperparams, output_dir: str | Path) -> dict[str, object]:
    """TRL `GRPOConfig` kwargs: bf16, G rollouts, vLLM colocate, periodic checkpoints."""
    return {
        "output_dir": str(output_dir),
        "bf16": True,
        "learning_rate": hyperparams.learning_rate,
        "beta": hyperparams.beta,
        "num_generations": hyperparams.num_generations,
        "max_completion_length": hyperparams.max_completion_length,
        "temperature": hyperparams.temperature,
        "top_p": hyperparams.top_p,
        "per_device_train_batch_size": hyperparams.per_device_batch_size,
        "gradient_accumulation_steps": hyperparams.gradient_accumulation_steps,
        "num_train_epochs": hyperparams.num_train_epochs,
        "warmup_ratio": hyperparams.warmup_ratio,
        "seed": hyperparams.seed,
        # vLLM colocate rollouts: the supported single-GPU path (colocate+PEFT hang is multi-GPU)
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": hyperparams.vllm_gpu_memory_utilization,
        "save_strategy": "steps",
        "save_steps": hyperparams.save_steps,
        "logging_steps": 10,
        "log_completions": True,
        "report_to": [],
    }


def build_grpo_config_from_kwargs(desired: Mapping[str, object]):
    from trl import GRPOConfig  # lazy import: train extra only

    return build_training_config(GRPOConfig, desired)


def build_grpo_config(hyperparams: GrpoHyperparams, output_dir: str | Path):
    return build_grpo_config_from_kwargs(grpo_config_kwargs(hyperparams, output_dir))


def split_train_dev(examples: Sequence[T], *, dev_size: int, seed: int) -> tuple[list[T], list[T]]:
    """Disjoint train/dev split for dev-EX checkpoint selection; the dev slice never trains.

    Deterministic in `seed`; `dev_size` is clamped so a small run still yields a train split.
    """
    import random

    if dev_size < 0:
        raise ValueError(f"dev_size must be >= 0, got {dev_size}")
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    k = min(dev_size, max(0, len(examples) - 1))
    dev_idx = set(order[:k])
    dev = [examples[i] for i in sorted(dev_idx)]
    train = [examples[i] for i in range(len(examples)) if i not in dev_idx]
    return train, dev


class GrpoProvenance(Frozen):
    """Everything needed to reproduce a GRPO checkpoint: base, warm-start, reward, GRPO config."""

    base_model: str
    base_model_revision: str | None = None
    warm_start_adapter: str
    hyperparams: GrpoHyperparams
    reward: ShapingConfig | RewardConfig
    dataset: str
    dataset_revision: str
    n_train: int
    n_dev: int
    runtime_config: RuntimeConfigRecord | None = None
    warm_start_identity: AdapterIdentity | None = None
    output_adapter_identity: AdapterIdentity | None = None
    requested_base_model_revision: str | None = None
    train_source_ids: tuple[str, ...] | None = None
    dev_source_ids: tuple[str, ...] | None = None
    # A GRPO dev split does not establish whether SFT previously exposed those examples.
    sft_dev_exposure: str = "unknown"
    sft_train_source_ids: tuple[str, ...] | None = None
    sft_exposed_dev_source_ids: tuple[str, ...] | None = None
    source_identity_policy: str | None = None
    library_versions: dict[str, str] = Field(default_factory=dict)
    git_commit: str = "unknown"
    git_dirty: bool = False


def build_grpo_provenance(
    *,
    base_model: str,
    warm_start_adapter: str | Path,
    hyperparams: GrpoHyperparams,
    reward: ShapingConfig,
    dataset: str,
    dataset_revision: str,
    n_train: int,
    n_dev: int,
    base_model_revision: str | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
    runtime_config: RuntimeConfigRecord | None = None,
    warm_start_identity: AdapterIdentity | None = None,
    output_adapter_identity: AdapterIdentity | None = None,
    requested_base_model_revision: str | None = None,
    train_source_ids: tuple[str, ...] | None = None,
    dev_source_ids: tuple[str, ...] | None = None,
    sft_train_source_ids: tuple[str, ...] | None = None,
) -> GrpoProvenance:
    """Assemble GRPO checkpoint provenance, capturing reward + GRPO config + library/git state."""
    return GrpoProvenance(
        base_model=base_model,
        base_model_revision=base_model_revision,
        warm_start_adapter=str(warm_start_adapter),
        hyperparams=hyperparams,
        reward=reward,
        dataset=dataset,
        dataset_revision=dataset_revision,
        n_train=n_train,
        n_dev=n_dev,
        runtime_config=runtime_config,
        warm_start_identity=warm_start_identity,
        output_adapter_identity=output_adapter_identity,
        requested_base_model_revision=requested_base_model_revision,
        train_source_ids=train_source_ids,
        dev_source_ids=dev_source_ids,
        sft_dev_exposure="known" if sft_train_source_ids is not None else "unknown",
        sft_train_source_ids=sft_train_source_ids,
        source_identity_policy=EXAMPLE_IDENTITY_POLICY if train_source_ids is not None else None,
        sft_exposed_dev_source_ids=(
            tuple(source for source in dev_source_ids if source in set(sft_train_source_ids))
            if sft_train_source_ids is not None and dev_source_ids is not None
            else None
        ),
        library_versions=library_versions(_TRAIN_LIBRARIES),
        git_commit=commit if commit is not None else git_commit(),
        git_dirty=dirty if dirty is not None else git_is_dirty(),
    )


def run_grpo(
    examples: Sequence[object],
    resources: Mapping[str, object],
    context_root: str | Path,
    output_dir: str | Path,
    *,
    task: Task | None = None,
    base_model: str,
    warm_start_adapter: str | Path,
    dataset: str,
    dataset_revision: str,
    base_model_revision: str | None = None,
    hyperparams: GrpoHyperparams | None = None,
    reward: ShapingConfig | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
    on_begin: Callable[[], None] | None = None,
) -> Path:
    """Train a GRPO LoRA adapter warm-started from the SFT adapter; persist it + provenance (GPU).

    Task-agnostic: the rows and the reward come from `task` (default: the text-to-SQL task) via
    `task.build_grpo_rows` + `task.make_reward_fn`, so a registered non-SQL `Task` trains through
    this same driver. `resources` is whatever the task's row builder needs (SQL: the schemas);
    `context_root` is where its verifier resolves context (SQL: the database root).

    Warm-starts by loading `warm_start_adapter` over `base_model` as a *trainable* PEFT model, so
    the saved adapter is the SFT→GRPO continuation (one adapter over the base at eval). A disjoint
    dev slice is held out for post-hoc dev-EX checkpoint selection; periodic checkpoints land under
    `output_dir`.
    """
    tk: Task = task or sql_task()
    hp = hyperparams or GrpoHyperparams()
    rw = reward or (RewardConfig() if tk.name == "sql" else ShapingConfig())
    if tk.name == "sql" and not isinstance(rw, RewardConfig):
        rw = RewardConfig.model_validate(rw.model_dump())
    if tk.name == "sql" and isinstance(rw, RewardConfig) and not rw.guard_trivial_empty:
        raise ValueError("SQL GRPO training requires guard_trivial_empty=True")

    train_examples, dev_examples = split_train_dev(
        examples, dev_size=hp.eval_slice_size, seed=hp.seed
    )
    rows = tk.build_grpo_rows(train_examples, resources)
    from llm_rlvr.data.models import Example

    def source_ids(items: Sequence[object]) -> tuple[str, ...] | None:
        if not all(isinstance(item, Example) for item in items):
            return None
        return tuple(example_identity(item) for item in items if isinstance(item, Example))

    train_ids, dev_ids = source_ids(train_examples), source_ids(dev_examples)
    if train_ids is not None and dev_ids is not None and set(train_ids) & set(dev_ids):
        raise ValueError("GRPO train/dev contain the same source example")
    if tk.name == "sql":
        from llm_rlvr.data.paths import resolve_database
        from llm_rlvr.data.splits import assert_train_split
        from llm_rlvr.tasks.sql.verifier import GoldExecutionError, validate_reference

        sql_examples = [item for item in examples if isinstance(item, Example)]
        if len(sql_examples) != len(examples):
            raise ValueError("SQL GRPO requires typed Example inputs")
        assert_train_split(sql_examples, context="SQL GRPO input")
        for example in sql_examples:
            try:
                validate_reference(example.gold_sql, resolve_database(example.db_id, context_root))
            except GoldExecutionError as exc:
                identity = example_identity(example)
                raise ValueError(f"SQL GRPO reference failed for source {identity}: {exc}") from exc
    execution = execute_prepared_grpo(
        rows,
        tk.make_reward_fn(context_root, config=rw),
        output_dir,
        base_model=base_model,
        base_model_revision=base_model_revision,
        warm_start_adapter=warm_start_adapter,
        requested_config=grpo_config_kwargs(hp, output_dir),
        request_metadata={
            "dataset": dataset,
            "dataset_revision": dataset_revision,
            "hyperparams": hp.model_dump(mode="json"),
            "reward": rw.model_dump(mode="json"),
            "train_source_ids": list(train_ids) if train_ids is not None else None,
            "dev_source_ids": list(dev_ids) if dev_ids is not None else None,
            "source_identity_policy": EXAMPLE_IDENTITY_POLICY if train_ids is not None else None,
        },
        on_begin=on_begin,
    )

    provenance = build_grpo_provenance(
        base_model=base_model,
        base_model_revision=execution.base_model_revision,
        warm_start_adapter=warm_start_adapter,
        hyperparams=hp,
        reward=rw,
        dataset=dataset,
        dataset_revision=dataset_revision,
        n_train=len(train_examples),
        n_dev=len(dev_examples),
        runtime_config=execution.configuration,
        warm_start_identity=execution.warm_start,
        output_adapter_identity=execution.output_adapter,
        requested_base_model_revision=base_model_revision,
        train_source_ids=train_ids,
        dev_source_ids=dev_ids,
        sft_train_source_ids=(
            execution.warm_start.sft_train_source_ids
            if execution.warm_start.source_identity_policy == EXAMPLE_IDENTITY_POLICY
            else None
        ),
        commit=commit,
        dirty=dirty,
    )
    return publish_training_run(
        output_dir,
        provenance,
        base_model=base_model,
        base_model_revision=execution.base_model_revision,
        training_population_ids=train_ids,
        sft_training_population_ids=execution.warm_start.sft_train_source_ids,
    )
