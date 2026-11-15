"""GRPO execution over prepared rows and an already bound task reward."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Self

from pydantic import Field, JsonValue, field_validator, model_validator

from llm_rlvr._base import Frozen
from llm_rlvr.artifacts.io import atomic_write_json
from llm_rlvr.artifacts.runs import begin_run, publish_complete, require_complete
from llm_rlvr.harness.messages import ChatMessage, validate_prompt
from llm_rlvr.train.configuration import (
    AdapterIdentity,
    AdapterSourceProvenance,
    RuntimeConfigRecord,
    configuration_values,
    effective_values,
    inspect_adapter,
    resolve_adapter_base,
    resolve_model_revision,
)


class TrainingExecution(Frozen):
    base_model: str
    base_model_revision: str | None
    warm_start: AdapterIdentity
    output_adapter: AdapterIdentity
    configuration: RuntimeConfigRecord


class PreparedGrpoRequest(Frozen):
    base_model: str
    requested_base_model_revision: str | None
    resolved_base_model_revision: str | None
    warm_start: AdapterIdentity
    configuration: RuntimeConfigRecord
    training_rows_sha256: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class PreparedSftRow(Frozen):
    prompt: tuple[ChatMessage, ...]
    completion: tuple[ChatMessage, ...] = Field(min_length=1, max_length=1)

    _prompt = field_validator("prompt")(validate_prompt)

    @model_validator(mode="after")
    def assistant_completion(self) -> Self:
        if self.completion[0].role != "assistant":
            raise ValueError("SFT completion role must be assistant")
        return self


class TrainingEligibility(Frozen):
    verification_policy: str = Field(min_length=1)
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    proof_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ids: tuple[str, ...] = Field(min_length=1)


class PreparedSftRequest(Frozen):
    base_model: str
    requested_base_model_revision: str | None
    resolved_base_model_revision: str | None
    configuration: RuntimeConfigRecord
    eligibility: TrainingEligibility
    training_rows_sha256: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SftTrainingExecution(Frozen):
    base_model: str
    base_model_revision: str | None
    configuration: RuntimeConfigRecord
    output_adapter: AdapterIdentity


def execute_prepared_sft(
    rows: Sequence[PreparedSftRow],
    output_dir: str | Path,
    *,
    base_model: str,
    base_model_revision: str | None,
    requested_config: Mapping[str, object],
    lora_config: object,
    requested_lora: Mapping[str, object],
    eligibility: TrainingEligibility,
    request_metadata: Mapping[str, object] | None = None,
    on_begin: Callable[[], None] | None = None,
) -> SftTrainingExecution:
    """Train validated conversational rows with a task-supplied eligibility proof."""
    if not rows or len(rows) != len(eligibility.source_ids):
        raise ValueError("SFT rows must be nonempty and match eligibility source count")
    if len(set(eligibility.source_ids)) != len(eligibility.source_ids):
        raise ValueError("SFT eligibility source identities must be unique")
    if Path(output_dir).exists():
        raise FileExistsError(f"training output already exists: {output_dir}")
    from trl import SFTConfig

    from llm_rlvr.train.configuration import build_training_config

    model_kwargs = requested_config.get("model_init_kwargs", {})
    if not isinstance(model_kwargs, Mapping):
        raise ValueError("model_init_kwargs must be a mapping")
    if "revision" in model_kwargs and model_kwargs["revision"] != base_model_revision:
        raise ValueError("model_init_kwargs revision differs from base_model_revision")
    model_kwargs = {"dtype": "bfloat16"} | dict(model_kwargs)
    desired = dict(requested_config) | {
        "output_dir": str(output_dir),
        "model_init_kwargs": model_kwargs,
    }
    config = build_training_config(SFTConfig, desired)
    revision = resolve_model_revision(base_model, base_model_revision)
    desired["model_init_kwargs"] = dict(model_kwargs) | ({"revision": revision} if revision else {})
    config.model_init_kwargs = desired["model_init_kwargs"]
    runtime = RuntimeConfigRecord(
        requested_trl=configuration_values(desired),
        effective_trl=effective_values(config, desired),
        requested_lora=configuration_values(requested_lora),
        effective_lora=effective_values(lora_config, requested_lora),
    )
    dataset_rows = [row.model_dump(mode="json") for row in rows]
    rows_json = json.dumps(dataset_rows, sort_keys=True)
    begin_run(
        output_dir,
        kind="sft",
        request=PreparedSftRequest(
            base_model=base_model,
            requested_base_model_revision=base_model_revision,
            resolved_base_model_revision=revision,
            configuration=runtime,
            eligibility=eligibility,
            training_rows_sha256=hashlib.sha256(rows_json.encode()).hexdigest(),
            metadata=configuration_values(request_metadata or {}),
        ),
    )
    if on_begin is not None:
        on_begin()
    from datasets import Dataset
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=base_model,
        args=config,
        train_dataset=Dataset.from_list(dataset_rows),
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    saved = inspect_adapter(output_dir, base_model=base_model, base_model_revision=revision)
    runtime = runtime.model_copy(update={"effective_lora": saved.configuration})
    return SftTrainingExecution(
        base_model=base_model,
        base_model_revision=revision,
        configuration=runtime,
        output_adapter=saved,
    )


def execute_prepared_grpo(
    rows: Sequence[dict[str, object]],
    reward_fn: Callable[..., list[float]],
    output_dir: str | Path,
    *,
    base_model: str,
    base_model_revision: str | None,
    warm_start_adapter: str | Path,
    requested_config: Mapping[str, object],
    request_metadata: Mapping[str, object] | None = None,
    on_begin: Callable[[], None] | None = None,
) -> TrainingExecution:
    if not rows:
        raise ValueError("GRPO training rows must not be empty")
    if Path(output_dir).exists():
        raise FileExistsError(f"training output already exists: {output_dir}")
    require_complete(warm_start_adapter)
    from trl import GRPOConfig, GRPOTrainer

    from llm_rlvr.train.configuration import build_training_config

    desired = dict(requested_config) | {"output_dir": str(output_dir)}
    config = build_training_config(GRPOConfig, desired)
    revision, warm_start = resolve_adapter_base(
        warm_start_adapter, base_model=base_model, requested_revision=base_model_revision
    )
    runtime = RuntimeConfigRecord(
        requested_trl=configuration_values(desired),
        effective_trl=effective_values(config, desired),
        requested_lora={},
        effective_lora=warm_start.configuration,
    )
    rows_json = json.dumps(configuration_values({"rows": list(rows)}), sort_keys=True)
    begin_run(
        output_dir,
        kind="grpo",
        request=PreparedGrpoRequest(
            base_model=base_model,
            requested_base_model_revision=base_model_revision,
            resolved_base_model_revision=revision,
            warm_start=warm_start,
            configuration=runtime,
            training_rows_sha256=hashlib.sha256(rows_json.encode()).hexdigest(),
            metadata=configuration_values(request_metadata or {}),
        ),
    )
    if on_begin is not None:
        on_begin()

    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    train_dataset = Dataset.from_list(list(rows))
    revision_kwargs = {"revision": revision} if revision else {}
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype="bfloat16", **revision_kwargs)
    policy = PeftModel.from_pretrained(base, str(warm_start_adapter), is_trainable=True)
    if revision:
        # PEFT saves the warm start's declared ref, which may be a branch or tag, not the commit.
        for adapter_config in policy.peft_config.values():
            adapter_config.revision = revision
    trainer = GRPOTrainer(
        model=policy, reward_funcs=[reward_fn], args=config, train_dataset=train_dataset
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    saved = inspect_adapter(output_dir, base_model=base_model, base_model_revision=revision)
    runtime = runtime.model_copy(update={"effective_lora": saved.configuration})
    return TrainingExecution(
        base_model=base_model,
        base_model_revision=revision,
        warm_start=warm_start,
        output_adapter=saved,
        configuration=runtime,
    )


def publish_training_run(
    output_dir: str | Path,
    provenance: Frozen,
    *,
    base_model: str,
    base_model_revision: str | None,
    training_population_ids: Sequence[str] | None,
    sft_training_population_ids: Sequence[str] | None = None,
) -> Path:
    """Publish final provenance, independently verifiable checkpoints, then run completion."""
    from llm_rlvr.train.checkpoints import publish_checkpoints

    root = Path(output_dir).resolve()
    payload = provenance.model_dump(mode="json")
    if "base_model_revision" not in payload:
        raise ValueError("training provenance must include base_model_revision explicitly")
    source = AdapterSourceProvenance.model_validate(payload)
    if source.base_model != base_model or source.base_model_revision != base_model_revision:
        raise ValueError(
            "training provenance base model or revision differs from the saved adapter"
        )
    identity = inspect_adapter(root, base_model=base_model, base_model_revision=base_model_revision)
    provenance_path = atomic_write_json(provenance, root / "provenance.json")
    publish_checkpoints(
        root,
        base_model=base_model,
        base_model_revision=base_model_revision,
        training_population_ids=training_population_ids,
        sft_training_population_ids=sft_training_population_ids,
    )
    publish_complete(
        root, artifacts=[provenance_path, *(root / file.path for file in identity.files)]
    )
    return Path(output_dir)
