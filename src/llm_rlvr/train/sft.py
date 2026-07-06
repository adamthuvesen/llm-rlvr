"""LoRA SFT on the cold-start trace dataset.

TRL `SFTTrainer` + PEFT LoRA, bf16, completion-only loss. The dataset is conversational
prompt-completion JSONL (see `traces.py`), so TRL masks the prompt and computes loss only over
the assistant `<think>`+SQL turn — `completion_only_loss=True` makes that contract explicit.

The kwargs builders are pure (no trl/peft import) so hyperparameters are unit-testable on CPU;
`run_sft` is the GPU-only driver Modal calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import Field

from llm_rlvr._base import Frozen
from llm_rlvr.data.paths import sha256_of
from llm_rlvr.eval.completions import write_pydantic_json
from llm_rlvr.eval.provenance import git_commit, git_is_dirty, library_versions
from llm_rlvr.train.lora import lora_kwargs as shared_lora_kwargs
from llm_rlvr.train.policy import load_trainable_lora_adapter

_TRAIN_LIBRARIES = ("llm-rlvr", "trl", "peft", "transformers", "torch", "accelerate")

PROVENANCE_FILENAME = "provenance.json"


class SftHyperparams(Frozen):
    """Conservative SFT budget: teach format without overwriting the base model."""

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
    return shared_lora_kwargs(
        rank=hyperparams.lora_rank,
        alpha=hyperparams.lora_alpha,
        dropout=hyperparams.lora_dropout,
    )


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
        # Per-epoch adapters so dev-EX checkpoint selection has candidates to score.
        "save_strategy": "epoch",
        "logging_steps": 10,
        "report_to": [],
    }


def build_lora_config(hyperparams: SftHyperparams):
    from peft import LoraConfig  # lazy import: train extra only

    return LoraConfig(**lora_kwargs(hyperparams))


def build_sft_config_from_kwargs(desired: Mapping[str, object]):
    import dataclasses
    import logging

    from trl import SFTConfig  # lazy import: train extra only

    valid = {f.name for f in dataclasses.fields(SFTConfig)}
    dropped = sorted(set(desired) - valid)
    if dropped:
        logging.getLogger(__name__).warning(
            "SFTConfig dropped unsupported kwargs %s on this trl version", dropped
        )
    return SFTConfig(**{k: v for k, v in desired.items() if k in valid})


def build_sft_config(hyperparams: SftHyperparams, output_dir: str | Path):
    return build_sft_config_from_kwargs(sft_config_kwargs(hyperparams, output_dir))


class AdapterProvenance(Frozen):
    """Everything needed to retrain an adapter checkpoint: base, LoRA config, data, versions."""

    base_model: str
    base_model_revision: str | None = None
    # Set when SFT continues an existing adapter; None means fresh LoRA from the base.
    warm_start_adapter: str | None = None
    # True when rows were pre-rendered to text with the run_sql tools block (rollout parity).
    render_tool_prompts: bool = False
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
    warm_start_adapter: str | Path | None = None,
    render_tool_prompts: bool = False,
    commit: str | None = None,
    dirty: bool | None = None,
) -> AdapterProvenance:
    """Assemble checkpoint provenance, hashing the trace dataset it was trained on."""
    return AdapterProvenance(
        base_model=base_model,
        base_model_revision=base_model_revision,
        warm_start_adapter=str(warm_start_adapter) if warm_start_adapter is not None else None,
        render_tool_prompts=render_tool_prompts,
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
    warm_start_adapter: str | Path | None = None,
    render_tool_prompts: bool = False,
    hyperparams: SftHyperparams | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> Path:
    """Train the LoRA adapter on the trace dataset and persist adapter + provenance (GPU).

    Saves the final adapter (weights + config) to `output_dir`, with per-epoch checkpoints
    under it, and writes `provenance.json` alongside. The base model is never modified.
    `warm_start_adapter` continues an existing LoRA adapter instead of creating a fresh one.
    `render_tool_prompts` pre-renders every conversational row to text through the real chat
    template with the `run_sql` tools block — byte parity with the rollout prompt.
    """
    from datasets import load_dataset  # lazy: heavy training stack, train extra only
    from trl import SFTTrainer

    hp = hyperparams or SftHyperparams()
    dataset = load_dataset("json", data_files=str(trace_path), split="train")
    # TRL's prompt-completion path wants exactly these two columns; the rest is audit metadata.
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in ("prompt", "completion")]
    )
    if render_tool_prompts:
        from datasets import Features, Value
        from transformers import AutoTokenizer
        from transformers.utils import get_json_schema

        from llm_rlvr.env.text2sql_env import SqliteText2SqlEnv
        from llm_rlvr.train.tooluse import render_text_row

        tokenizer = AutoTokenizer.from_pretrained(
            base_model, **({"revision": base_model_revision} if base_model_revision else {})
        )
        # The same schema-derivation TRL applies to the env's tool at rollout time.
        tools = [get_json_schema(SqliteText2SqlEnv(".").run_sql)]
        # remove_columns + explicit string features: without them, datasets silently CASTS the
        # rendered strings back into the original List(Json) column feature ("C" -> ["C"]) and
        # TRL's add_eos crashes on a list — observed on the first tooluse-sft2 launch.
        dataset = dataset.map(
            lambda row: render_text_row(
                row["prompt"], row["completion"], tokenizer.apply_chat_template, tools
            ),
            remove_columns=dataset.column_names,
            features=Features({"prompt": Value("string"), "completion": Value("string")}),
        )
        if not dataset[0]["completion"].endswith(tokenizer.eos_token):
            raise ValueError(
                "rendered completion does not end on the EOS token — chat template drift?"
            )

    config = build_sft_config(hp, output_dir)
    if warm_start_adapter is None:
        config.model_init_kwargs = {"dtype": "bfloat16"} | (
            {"revision": base_model_revision} if base_model_revision else {}
        )
        trainer = SFTTrainer(
            model=base_model,
            args=config,
            train_dataset=dataset,
            peft_config=build_lora_config(hp),
        )
    else:
        policy = load_trainable_lora_adapter(base_model, base_model_revision, warm_start_adapter)
        trainer = SFTTrainer(model=policy, args=config, train_dataset=dataset)
    trainer.train()
    trainer.save_model(str(output_dir))

    provenance = build_adapter_provenance(
        base_model=base_model,
        base_model_revision=base_model_revision,
        warm_start_adapter=warm_start_adapter,
        render_tool_prompts=render_tool_prompts,
        hyperparams=hp,
        trace_dataset_path=trace_path,
        n_traces=len(dataset),
        commit=commit,
        dirty=dirty,
    )
    write_pydantic_json(provenance, Path(output_dir) / PROVENANCE_FILENAME)
    return Path(output_dir)
