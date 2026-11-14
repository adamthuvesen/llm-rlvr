"""LoRA SFT on the cold-start trace dataset.

TRL `SFTTrainer` + PEFT LoRA, bf16, completion-only loss. The dataset is conversational
prompt-completion JSONL (see `traces.py`), so TRL masks the prompt and computes loss only over
the assistant `<think>`+SQL turn — `completion_only_loss=True` makes that contract explicit.

The kwargs builders are pure (no trl/peft import) so hyperparameters are unit-testable on CPU;
`run_sft` is the GPU-only driver Modal calls.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict

from pydantic import Field

from llm_rlvr._base import Frozen
from llm_rlvr.artifacts.io import sha256_of
from llm_rlvr.artifacts.provenance import git_commit, git_is_dirty, library_versions
from llm_rlvr.artifacts.runs import require_complete
from llm_rlvr.data.models import EXAMPLE_IDENTITY_POLICY, example_identity
from llm_rlvr.train.configuration import (
    AdapterIdentity,
    RuntimeConfigRecord,
    build_training_config,
)
from llm_rlvr.train.prepared import (
    PreparedSftRow,
    TrainingEligibility,
    execute_prepared_sft,
    publish_training_run,
)
from llm_rlvr.train.traces import TraceDatasetManifest, read_trace_rows

_TRAIN_LIBRARIES = ("llm-rlvr", "trl", "peft", "transformers", "torch", "accelerate")

PROVENANCE_FILENAME = "provenance.json"


class SftHyperparams(Frozen):
    """Conservative SFT budget: teach format without overwriting the base model."""

    lora_rank: int = Field(default=32, gt=0, strict=True)
    lora_alpha: int = Field(default=64, gt=0, strict=True)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1, allow_inf_nan=False)
    learning_rate: float = Field(default=1e-4, gt=0, allow_inf_nan=False)
    lr_scheduler: str = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0, le=1, allow_inf_nan=False)
    epochs: int = Field(default=3, gt=0, strict=True)
    max_seq_len: int = Field(default=2048, gt=0, strict=True)
    per_device_batch_size: int = Field(default=4, gt=0, strict=True)
    gradient_accumulation_steps: int = Field(default=4, gt=0, strict=True)
    seed: int = 0


class _LoraKwargs(TypedDict):
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: Literal["all-linear"]
    bias: Literal["none"]
    task_type: Literal["CAUSAL_LM"]


def lora_kwargs(hyperparams: SftHyperparams) -> _LoraKwargs:
    """PEFT `LoraConfig` kwargs: all-linear targets, the decided rank/alpha/dropout."""
    return {
        "r": hyperparams.lora_rank,
        "lora_alpha": hyperparams.lora_alpha,
        "lora_dropout": hyperparams.lora_dropout,
        "target_modules": "all-linear",
        "bias": "none",
        "task_type": "CAUSAL_LM",
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
        # Per-epoch adapters so dev-EX checkpoint selection has candidates to score.
        "save_strategy": "epoch",
        "logging_steps": 10,
        "report_to": [],
    }


def build_lora_config(hyperparams: SftHyperparams) -> Any:
    from peft import LoraConfig  # lazy import: train extra only

    return LoraConfig(**lora_kwargs(hyperparams))


def build_sft_config_from_kwargs(desired: Mapping[str, object]) -> Any:
    from trl import SFTConfig  # lazy import: train extra only

    return build_training_config(SFTConfig, desired)


class AdapterProvenance(Frozen):
    """Everything needed to retrain an adapter checkpoint: base, LoRA config, data, versions."""

    base_model: str
    base_model_revision: str | None = None
    # Retained because existing adapter provenance files serialize these fields.
    warm_start_adapter: str | None = None
    render_tool_prompts: bool = False
    hyperparams: SftHyperparams
    trace_dataset_path: str
    trace_dataset_sha256: str
    n_traces: int
    runtime_config: RuntimeConfigRecord | None = None
    output_adapter_identity: AdapterIdentity | None = None
    requested_base_model_revision: str | None = None
    trace_manifest_sha256: str | None = None
    train_source_ids: tuple[str, ...] | None = None
    source_identity_policy: str | None = None
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
    runtime_config: RuntimeConfigRecord | None = None,
    output_adapter_identity: AdapterIdentity | None = None,
    requested_base_model_revision: str | None = None,
    trace_manifest_sha256: str | None = None,
    train_source_ids: tuple[str, ...] | None = None,
) -> AdapterProvenance:
    """Assemble checkpoint provenance, hashing the trace dataset it was trained on."""
    return AdapterProvenance(
        base_model=base_model,
        base_model_revision=base_model_revision,
        hyperparams=hyperparams,
        trace_dataset_path=str(trace_dataset_path),
        trace_dataset_sha256=sha256_of(trace_dataset_path),
        n_traces=n_traces,
        runtime_config=runtime_config,
        output_adapter_identity=output_adapter_identity,
        requested_base_model_revision=requested_base_model_revision,
        trace_manifest_sha256=trace_manifest_sha256,
        train_source_ids=train_source_ids,
        source_identity_policy=EXAMPLE_IDENTITY_POLICY if train_source_ids is not None else None,
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
    trace_manifest_path: str | Path | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
    on_begin: Callable[[], None] | None = None,
) -> Path:
    """Train the LoRA adapter on the trace dataset and persist adapter + provenance (GPU).

    Saves the final adapter (weights + config) to `output_dir`, with per-epoch checkpoints
    under it, and writes `provenance.json` alongside. The base model is never modified.
    """
    hp = hyperparams or SftHyperparams()
    if Path(output_dir).exists():
        raise FileExistsError(f"training output already exists: {output_dir}")
    require_complete(trace_path)
    rows = read_trace_rows(trace_path, require_train_lineage=True)
    if not rows:
        raise ValueError("SFT training traces must not be empty")
    if trace_manifest_path is None:
        raise ValueError("SFT training requires trace_manifest_path with verified filter counts")
    require_complete(trace_manifest_path)
    manifest = TraceDatasetManifest.model_validate_json(
        Path(trace_manifest_path).read_text(encoding="utf-8")
    )
    if manifest.counts.n_gold_errors:
        raise ValueError("SFT training requires n_gold_errors=0 in the trace manifest")
    if manifest.source_examples is None:
        raise ValueError("SFT training requires source_examples lineage in the trace manifest")
    if manifest.counts.n_kept != len(rows):
        raise ValueError("trace manifest n_kept differs from training row count")
    if any(row.source_example not in manifest.source_examples for row in rows):
        raise ValueError("trace source example is absent from trace manifest")
    trace_hash, manifest_hash = sha256_of(trace_path), sha256_of(trace_manifest_path)
    if manifest.trace_dataset_sha256 != trace_hash:
        raise ValueError("trace manifest trace_dataset_sha256 is missing or differs from traces")
    train_ids = tuple(
        example_identity(row.source_example) for row in rows if row.source_example is not None
    )
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("SFT training contains duplicate source examples")
    execution = execute_prepared_sft(
        [PreparedSftRow(prompt=row.prompt, completion=row.completion) for row in rows],
        output_dir,
        base_model=base_model,
        base_model_revision=base_model_revision,
        requested_config=sft_config_kwargs(hp, output_dir),
        lora_config=build_lora_config(hp),
        requested_lora=lora_kwargs(hp),
        eligibility=TrainingEligibility(
            verification_policy="sql-traces-zero-gold-errors-v1",
            source_artifact_sha256=trace_hash,
            proof_sha256=manifest_hash,
            source_ids=train_ids,
        ),
        request_metadata={
            "hyperparams": hp.model_dump(mode="json"),
            "trace_dataset_path": str(trace_path),
            "trace_manifest_path": str(trace_manifest_path),
            "source_identity_policy": EXAMPLE_IDENTITY_POLICY,
        },
        on_begin=on_begin,
    )
    if sha256_of(trace_path) != trace_hash or sha256_of(trace_manifest_path) != manifest_hash:
        raise ValueError("trace dataset or manifest changed during SFT training")

    provenance = build_adapter_provenance(
        base_model=base_model,
        base_model_revision=execution.base_model_revision,
        hyperparams=hp,
        trace_dataset_path=trace_path,
        n_traces=len(rows),
        runtime_config=execution.configuration,
        output_adapter_identity=execution.output_adapter,
        requested_base_model_revision=base_model_revision,
        trace_manifest_sha256=manifest_hash,
        train_source_ids=train_ids,
        commit=commit,
        dirty=dirty,
    )
    return publish_training_run(
        output_dir,
        provenance,
        base_model=base_model,
        base_model_revision=execution.base_model_revision,
        training_population_ids=train_ids,
        sft_training_population_ids=train_ids,
    )
