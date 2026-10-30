"""Publish independently verifiable periodic adapters after training succeeds."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, JsonValue, TypeAdapter

from llm_rlvr._base import Frozen
from llm_rlvr.artifacts.io import atomic_write_json, ensure_new_output, sha256_of
from llm_rlvr.artifacts.runs import (
    COMPLETION_FILENAME,
    REQUEST_FILENAME,
    RunRequestManifest,
    publish_complete,
    request_digest,
)
from llm_rlvr.train.configuration import AdapterIdentity, inspect_adapter


class _TrainerState(Frozen):
    model_config = ConfigDict(frozen=True, extra="ignore")
    global_step: int = Field(ge=0, strict=True)
    epoch: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class CheckpointProvenance(Frozen):
    base_model: str
    base_model_revision: str | None
    parent_request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    checkpoint_name: str
    saved_step: int = Field(ge=0, strict=True)
    saved_step_source: Literal["trainer_state.global_step", "directory_name"]
    trainer_state_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    epoch: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    output_adapter_identity: AdapterIdentity
    training_population_ids: tuple[str, ...] | None
    sft_training_population_ids: tuple[str, ...] | None = None
    training_exposure: Literal["unknown"] = "unknown"


def publish_checkpoints(
    output_dir: str | Path,
    *,
    base_model: str,
    base_model_revision: str | None,
    training_population_ids: Sequence[str] | None,
    sft_training_population_ids: Sequence[str] | None = None,
) -> tuple[Path, ...]:
    """Validate every periodic adapter, then publish each checkpoint with its own digests.

    Population IDs describe available training inputs. They do not identify examples already seen
    at an intermediate step. Evaluation selects a checkpoint through its explicit adapter path.
    """
    root = Path(output_dir).resolve()
    parent = RunRequestManifest.model_validate_json(
        (root / REQUEST_FILENAME).read_text(encoding="utf-8")
    )
    if parent.kind not in {"sft", "grpo"}:
        raise ValueError("checkpoint publication requires an SFT or GRPO parent run")
    expected_revision = parent.request.get(
        "resolved_base_model_revision", parent.request.get("base_model_revision")
    )
    if parent.request.get("base_model") != base_model or expected_revision != base_model_revision:
        raise ValueError(
            "checkpoint base model or revision differs from the parent training request"
        )
    drafts: list[tuple[Path, CheckpointProvenance, list[Path]]] = []
    for checkpoint in sorted(root.glob("checkpoint-*")):
        if not checkpoint.is_dir() or checkpoint.is_symlink():
            raise ValueError(
                f"checkpoint must be a directory inside the training run: {checkpoint}"
            )
        match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
        if match is None:
            raise ValueError(f"invalid checkpoint step directory: {checkpoint.name}")
        named_step = int(match.group(1))
        for filename in (REQUEST_FILENAME, COMPLETION_FILENAME, "provenance.json"):
            ensure_new_output(checkpoint / filename)
        state_path = checkpoint / "trainer_state.json"
        state = (
            _TrainerState.model_validate_json(state_path.read_text(encoding="utf-8"))
            if state_path.exists()
            else None
        )
        if state is not None and state.global_step != named_step:
            raise ValueError(f"trainer state step differs from checkpoint directory: {checkpoint}")
        identity = inspect_adapter(
            checkpoint, base_model=base_model, base_model_revision=base_model_revision
        )
        files = [checkpoint / file.path for file in identity.files]
        if state is not None:
            files.append(state_path)
        for artifact in files:
            if (
                not artifact.resolve().is_relative_to(checkpoint.resolve())
                or not artifact.is_file()
                or artifact.stat().st_size == 0
            ):
                raise ValueError(f"checkpoint artifact is missing or empty: {artifact}")
        provenance = CheckpointProvenance(
            base_model=base_model,
            base_model_revision=base_model_revision,
            parent_request_sha256=parent.request_sha256,
            checkpoint_name=checkpoint.name,
            saved_step=state.global_step if state is not None else named_step,
            saved_step_source="trainer_state.global_step"
            if state is not None
            else "directory_name",
            trainer_state_sha256=sha256_of(state_path) if state is not None else None,
            epoch=state.epoch if state is not None else None,
            output_adapter_identity=identity,
            training_population_ids=(
                tuple(training_population_ids) if training_population_ids is not None else None
            ),
            sft_training_population_ids=(
                tuple(sft_training_population_ids)
                if sft_training_population_ids is not None
                else None
            ),
        )
        drafts.append((checkpoint, provenance, files))

    for checkpoint, provenance, files in drafts:
        request = RunRequestManifest(
            kind="checkpoint",
            request_sha256=request_digest(provenance),
            request=TypeAdapter(dict[str, JsonValue]).validate_python(
                provenance.model_dump(mode="json")
            ),
        )
        atomic_write_json(request, checkpoint / REQUEST_FILENAME)
        provenance_path = atomic_write_json(provenance, checkpoint / "provenance.json")
        publish_complete(checkpoint, artifacts=[*files, provenance_path])
    return tuple(checkpoint for checkpoint, _, _ in drafts)
