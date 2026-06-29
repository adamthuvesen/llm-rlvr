"""Single-turn GRPO LoRA training (Stage 2, design D1/D3/D4).

TRL `GRPOTrainer` + PEFT LoRA, bf16, vLLM colocate rollouts scored by the execution-grounded reward
(`llm_rlvr_sql.rewards`). The policy is **warm-started from the Stage-1 SFT adapter** — loaded as a
trainable PEFT model so the saved adapter is the SFT→GRPO continuation, evaluated as a single
adapter over the base (the adapter-eval path needs no change). The base model is config-parametric
(Qwen default; Llama-3.2-3B for the cross-family control).

As in `sft.py`, the kwargs builders are pure (no trl/peft import) so the decided hyperparameters are
unit-testable on the CPU box; `run_grpo` is the GPU-only driver Modal calls. The `GRPOConfig` /
`LoraConfig` kwargs are validated for real against the pinned TRL↔vLLM pair on Modal (task 1.2), not
at unit-test time — exactly like `sft_config_kwargs`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import Field

from llm_rlvr_sql._base import Frozen
from llm_rlvr_sql.data.models import DatabaseSchema, Example
from llm_rlvr_sql.eval.completions import write_json
from llm_rlvr_sql.eval.provenance import git_commit, git_is_dirty, library_versions
from llm_rlvr_sql.tasks.sql import RewardConfig, make_reward_fn, sql_task
from llm_rlvr_sql.train.lora import lora_kwargs as shared_lora_kwargs

_TRAIN_LIBRARIES = ("llm-rlvr-sql", "trl", "peft", "transformers", "torch", "accelerate", "vllm")

PROVENANCE_FILENAME = "provenance.json"


class GrpoHyperparams(Frozen):
    """The decided Stage-2 GRPO budget (design D1/D3/D4): efficient single-A100 colocate config."""

    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    # GRPO LoRA LR: the D4 default between the two literature anchors (SQL-R1 3e-7 full-FT ->
    # ~3e-6 LoRA, vs the "LoRA Without Regret" ~5e-5). Locked off the LR pilot (task 6.1).
    learning_rate: float = 1e-5
    beta: float = 0.04  # KL coefficient; start at a mild value, revisit if entropy collapses.
    num_generations: int = 4  # G=4 halves generation vs G=8; 2-GRPO shows G=2 keeps 97.6%.
    max_completion_length: int = 1024  # SQL is short; the length reward guards truncation.
    temperature: float = 1.0  # exploration during rollouts
    top_p: float = 1.0
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1  # early-stop / checkpoint-select off the dev-EX curve governs spend
    warmup_ratio: float = 0.03
    vllm_gpu_memory_utilization: float = 0.3  # colocate: leave room for the trainer on one card
    save_steps: int = 50  # periodic checkpoints so dev-EX selection (D3) has candidates to score
    eval_slice_size: int = 500  # held-out dev-EX slice for checkpoint selection
    seed: int = 0


def lora_kwargs(hyperparams: GrpoHyperparams) -> dict[str, object]:
    """PEFT `LoraConfig` kwargs: all-linear targets (incl. MLP), the decided rank/alpha/dropout."""
    return shared_lora_kwargs(
        rank=hyperparams.lora_rank,
        alpha=hyperparams.lora_alpha,
        dropout=hyperparams.lora_dropout,
    )


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


def build_lora_config(hyperparams: GrpoHyperparams):
    from peft import LoraConfig  # lazy import: train extra only

    return LoraConfig(**lora_kwargs(hyperparams))


def build_grpo_config_from_kwargs(desired: Mapping[str, object]):
    import dataclasses
    import logging

    from trl import GRPOConfig  # lazy import: train extra only

    # The TRL↔vLLM pairing is fragile and GRPOConfig fields drift across versions, so pass only the
    # kwargs this installed GRPOConfig accepts and warn loudly on any dropped — never silently.
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(set(desired) - valid)
    if dropped:
        logging.getLogger(__name__).warning(
            "GRPOConfig dropped unsupported kwargs %s on this trl version", dropped
        )
    return GRPOConfig(**{k: v for k, v in desired.items() if k in valid})


def build_grpo_config(hyperparams: GrpoHyperparams, output_dir: str | Path):
    return build_grpo_config_from_kwargs(grpo_config_kwargs(hyperparams, output_dir))


def split_train_dev(
    examples: Sequence[Example], *, dev_size: int, seed: int
) -> tuple[list[Example], list[Example]]:
    """Disjoint train/dev split for dev-EX checkpoint selection (D3) — the dev slice never trains.

    Deterministic in `seed`; `dev_size` is clamped so a tiny smoke set still yields a train split.
    """
    import random

    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    k = min(dev_size, max(0, len(examples) - 1))
    dev_idx = set(order[:k])
    dev = [examples[i] for i in sorted(dev_idx)]
    train = [examples[i] for i in range(len(examples)) if i not in dev_idx]
    return train, dev


def build_grpo_rows(
    examples: Sequence[Example],
    schemas: Mapping[str, DatabaseSchema],
    *,
    style: str = "cot",
) -> list[dict[str, object]]:
    """One GRPO dataset row per example: the chat `prompt` plus the `gold_sql` + `db_id` the reward
    needs. Pure/testable — TRL applies the chat template to `prompt` and passes the rest to the
    reward function as aligned kwargs. Delegates to the SQL task's row builder (`tasks.sql`)."""
    return sql_task().build_grpo_rows(examples, schemas, style=style)


class GrpoProvenance(Frozen):
    """Everything needed to reproduce a GRPO checkpoint: base, warm-start, reward, GRPO config."""

    base_model: str
    base_model_revision: str | None = None
    warm_start_adapter: str
    hyperparams: GrpoHyperparams
    reward: RewardConfig
    dataset: str
    dataset_revision: str
    n_train: int
    n_dev: int
    library_versions: dict[str, str] = Field(default_factory=dict)
    git_commit: str = "unknown"
    git_dirty: bool = False


def build_grpo_provenance(
    *,
    base_model: str,
    warm_start_adapter: str | Path,
    hyperparams: GrpoHyperparams,
    reward: RewardConfig,
    dataset: str,
    dataset_revision: str,
    n_train: int,
    n_dev: int,
    base_model_revision: str | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
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
        library_versions=library_versions(_TRAIN_LIBRARIES),
        git_commit=commit if commit is not None else git_commit(),
        git_dirty=dirty if dirty is not None else git_is_dirty(),
    )


def load_trainable_policy(
    base_model: str,
    base_model_revision: str | None,
    warm_start_adapter: str | Path | None,
    *,
    hyperparams: GrpoHyperparams | None = None,
):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype="bfloat16",
        **({"revision": base_model_revision} if base_model_revision else {}),
    )
    if warm_start_adapter is not None:
        return PeftModel.from_pretrained(base, str(warm_start_adapter), is_trainable=True)

    if hyperparams is None:
        raise ValueError("fresh trainable policy requires hyperparams for LoRA config")
    from peft import get_peft_model

    return get_peft_model(base, build_lora_config(hyperparams))


def run_grpo(
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
    reward: RewardConfig | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> Path:
    """Train a GRPO LoRA adapter warm-started from the SFT adapter; persist it + provenance (GPU).

    Warm-starts by loading `warm_start_adapter` over `base_model` as a *trainable* PEFT model, so
    the saved adapter is the SFT→GRPO continuation (one adapter over the base at eval). Rollouts are
    scored by the execution reward against `db_root`. A disjoint dev slice is held out for post-hoc
    dev-EX checkpoint selection; periodic checkpoints land under `output_dir`.
    """
    from datasets import Dataset  # lazy: heavy training stack, train extra only
    from trl import GRPOTrainer

    hp = hyperparams or GrpoHyperparams()
    rw = reward or RewardConfig()

    train_examples, dev_examples = split_train_dev(
        examples, dev_size=hp.eval_slice_size, seed=hp.seed
    )
    rows = build_grpo_rows(train_examples, schemas)
    train_dataset = Dataset.from_list(rows)

    policy = load_trainable_policy(
        base_model, base_model_revision, warm_start_adapter, hyperparams=hp
    )

    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=[make_reward_fn(db_root, config=rw)],
        args=build_grpo_config(hp, output_dir),
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(str(output_dir))

    provenance = build_grpo_provenance(
        base_model=base_model,
        base_model_revision=base_model_revision,
        warm_start_adapter=warm_start_adapter,
        hyperparams=hp,
        reward=rw,
        dataset=dataset,
        dataset_revision=dataset_revision,
        n_train=len(train_examples),
        n_dev=len(dev_examples),
        commit=commit,
        dirty=dirty,
    )
    write_json(provenance, Path(output_dir) / PROVENANCE_FILENAME)
    return Path(output_dir)
