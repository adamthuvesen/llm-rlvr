"""LoRA SFT on the cold-start trace dataset (design D3).

TRL `SFTTrainer` + PEFT LoRA, bf16, completion-only loss. The dataset is conversational
prompt-completion JSONL (see `traces.py`), so TRL masks the prompt and computes loss only over
the assistant `<think>`+SQL turn — `completion_only_loss=True` makes that contract explicit.

The kwargs builders are pure (no trl/peft import) so the decided hyperparameters are
unit-testable on the CPU box; `run_sft` is the GPU-only driver Modal calls.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from text2sql_rlvr._base import Frozen
from text2sql_rlvr.data.spider import sha256_of
from text2sql_rlvr.eval.completions import write_json
from text2sql_rlvr.eval.provenance import git_commit, git_is_dirty, library_versions

_TRAIN_LIBRARIES = ("text2sql-rlvr", "trl", "peft", "transformers", "torch", "accelerate")

PROVENANCE_FILENAME = "provenance.json"


class SftHyperparams(Frozen):
    """The decided Stage-1 budget: conservative — nudge behavior, don't overwrite the base."""

    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4  # SFT LoRA LR; ~10x the planned GRPO LoRA LR
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    epochs: int = 3
    max_seq_len: int = 2048
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    seed: int = 0


def lora_kwargs(hyperparams: SftHyperparams) -> dict[str, object]:
    """PEFT `LoraConfig` kwargs: all-linear targets, the decided rank/alpha/dropout."""
    return {
        "task_type": "CAUSAL_LM",
        "target_modules": "all-linear",
        "r": hyperparams.lora_rank,
        "lora_alpha": hyperparams.lora_alpha,
        "lora_dropout": hyperparams.lora_dropout,
        "bias": "none",
    }


def sft_config_kwargs(hyperparams: SftHyperparams, output_dir: str | Path) -> dict[str, object]:
    """TRL `SFTConfig` kwargs: bf16, completion-only loss, no packing, per-epoch checkpoints."""
    return {
        "output_dir": str(output_dir),
        "bf16": True,
        # Loss only over the assistant completion; the prompt (schema+question) is masked.
        "completion_only_loss": True,
        # Packing off: it complicates completion-only masking for marginal gain on short SQL.
        "packing": False,
        "max_length": hyperparams.max_seq_len,
        "learning_rate": hyperparams.learning_rate,
        "lr_scheduler_type": hyperparams.lr_scheduler,
        "warmup_ratio": hyperparams.warmup_ratio,
        "num_train_epochs": hyperparams.epochs,
        "per_device_train_batch_size": hyperparams.per_device_batch_size,
        "gradient_accumulation_steps": hyperparams.gradient_accumulation_steps,
        "seed": hyperparams.seed,
        # Per-epoch adapters so the dev-EX checkpoint selection (D3) has candidates to score.
        "save_strategy": "epoch",
        "logging_steps": 10,
        "report_to": [],
    }


def build_lora_config(hyperparams: SftHyperparams):
    from peft import LoraConfig  # lazy import: train extra only

    return LoraConfig(**lora_kwargs(hyperparams))


def build_sft_config(hyperparams: SftHyperparams, output_dir: str | Path):
    from trl import SFTConfig  # lazy import: train extra only

    return SFTConfig(**sft_config_kwargs(hyperparams, output_dir))


class AdapterProvenance(Frozen):
    """Everything needed to retrain an adapter checkpoint: base, LoRA config, data, versions."""

    base_model: str
    base_model_revision: str | None = None
    hyperparams: SftHyperparams
    trace_dataset_path: str
    trace_dataset_sha256: str
    n_traces: int
    library_versions: dict[str, str] = Field(default_factory=dict)
    git_commit: str = "unknown"
    git_dirty: bool = False


def build_adapter_provenance(
    *,
    base_model: str,
    hyperparams: SftHyperparams,
    trace_dataset_path: str | Path,
    n_traces: int,
    base_model_revision: str | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> AdapterProvenance:
    """Assemble checkpoint provenance, hashing the trace dataset it was trained on."""
    return AdapterProvenance(
        base_model=base_model,
        base_model_revision=base_model_revision,
        hyperparams=hyperparams,
        trace_dataset_path=str(trace_dataset_path),
        trace_dataset_sha256=sha256_of(trace_dataset_path),
        n_traces=n_traces,
        library_versions=library_versions(_TRAIN_LIBRARIES),
        git_commit=commit if commit is not None else git_commit(),
        git_dirty=dirty if dirty is not None else git_is_dirty(),
    )


def run_sft(
    trace_path: str | Path,
    output_dir: str | Path,
    *,
    base_model: str,
    base_model_revision: str | None = None,
    hyperparams: SftHyperparams | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> Path:
    """Train the LoRA adapter on the trace dataset and persist adapter + provenance (GPU).

    Saves the final adapter (weights + config) to `output_dir`, with per-epoch checkpoints
    under it, and writes `provenance.json` alongside. The base model is never modified.
    """
    from datasets import load_dataset  # lazy: heavy training stack, train extra only
    from trl import SFTTrainer

    hp = hyperparams or SftHyperparams()
    dataset = load_dataset("json", data_files=str(trace_path), split="train")
    # TRL's prompt-completion path wants exactly these two columns; the rest is audit metadata.
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in ("prompt", "completion")]
    )

    config = build_sft_config(hp, output_dir)
    config.model_init_kwargs = {"dtype": "bfloat16"} | (
        {"revision": base_model_revision} if base_model_revision else {}
    )
    trainer = SFTTrainer(
        model=base_model,
        args=config,
        train_dataset=dataset,
        peft_config=build_lora_config(hp),
    )
    trainer.train()
    trainer.save_model(str(output_dir))

    provenance = build_adapter_provenance(
        base_model=base_model,
        base_model_revision=base_model_revision,
        hyperparams=hp,
        trace_dataset_path=trace_path,
        n_traces=len(dataset),
        commit=commit,
        dirty=dirty,
    )
    write_json(provenance, Path(output_dir) / PROVENANCE_FILENAME)
    return Path(output_dir)
