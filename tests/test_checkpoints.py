"""Periodic checkpoints carry independent artifact identity and honest training lineage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_rlvr._base import Frozen
from llm_rlvr.artifacts.io import sha256_of
from llm_rlvr.artifacts.runs import begin_run, require_complete
from llm_rlvr.train.checkpoints import CheckpointProvenance, publish_checkpoints
from llm_rlvr.train.configuration import inspect_adapter

REVISION = "a" * 40


class TrainingRequest(Frozen):
    base_model: str = "org/model"
    base_model_revision: str | None = REVISION


def _parent(tmp_path: Path) -> Path:
    root = tmp_path / "training"
    begin_run(root, kind="sft", request=TrainingRequest())
    return root


def _checkpoint(root: Path, step: int, *, trainer_state: bool = True) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "org/model",
                "revision": REVISION,
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0,
            }
        )
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(f"step {step} weights".encode())
    if trainer_state:
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step, "epoch": 0.5, "other_trainer_field": True})
        )
    return checkpoint


def _publish(root: Path) -> tuple[Path, ...]:
    return publish_checkpoints(
        root,
        base_model="org/model",
        base_model_revision=REVISION,
        training_population_ids=("source-a", "source-b"),
        sft_training_population_ids=("warm-source",),
    )


def test_periodic_checkpoints_are_independently_selectable_and_identified(tmp_path: Path) -> None:
    root = _parent(tmp_path)
    first = _checkpoint(root, 10)
    second = _checkpoint(root, 20, trainer_state=False)
    parent_request = json.loads((root / "request.json").read_text())
    assert _publish(root) == (first, second)
    first_metadata = CheckpointProvenance.model_validate_json(
        (first / "provenance.json").read_text()
    )
    second_metadata = CheckpointProvenance.model_validate_json(
        (second / "provenance.json").read_text()
    )
    assert first_metadata.saved_step == 10
    assert first_metadata.saved_step_source == "trainer_state.global_step"
    assert first_metadata.trainer_state_sha256 == sha256_of(first / "trainer_state.json")
    assert first_metadata.epoch == 0.5
    assert second_metadata.saved_step == 20
    assert second_metadata.saved_step_source == "directory_name"
    assert second_metadata.trainer_state_sha256 is None
    assert second_metadata.epoch is None
    assert first_metadata.parent_request_sha256 == parent_request["request_sha256"]
    assert first_metadata.base_model_revision == REVISION
    assert (
        first_metadata.output_adapter_identity.sha256
        != second_metadata.output_adapter_identity.sha256
    )
    assert first_metadata.training_population_ids == ("source-a", "source-b")
    assert first_metadata.sft_training_population_ids == ("warm-source",)
    assert first_metadata.training_exposure == "unknown"
    identity = inspect_adapter(first, base_model="org/model", base_model_revision=REVISION)
    assert identity.train_source_ids is None
    assert identity.sft_train_source_ids is None
    assert identity.sha256 == first_metadata.output_adapter_identity.sha256
    require_complete(first)
    require_complete(second)
    with pytest.raises(ValueError, match="incomplete"):
        require_complete(root)
    (first / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        require_complete(first)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_weights",
        "empty_weights",
        "step_mismatch",
        "malformed_state",
        "invalid_name",
        "wrong_base",
    ],
)
def test_invalid_checkpoint_prevents_all_publication(tmp_path: Path, problem: str) -> None:
    root = _parent(tmp_path)
    first = _checkpoint(root, 10)
    second = _checkpoint(root, 20)
    if problem == "missing_weights":
        (second / "adapter_model.safetensors").unlink()
    elif problem == "empty_weights":
        (second / "adapter_model.safetensors").write_bytes(b"")
    elif problem == "step_mismatch":
        (second / "trainer_state.json").write_text('{"global_step": 21}')
    elif problem == "malformed_state":
        (second / "trainer_state.json").write_text('{"global_step": "20"}')
    elif problem == "invalid_name":
        second = second.rename(root / "checkpoint-latest")
    else:
        configuration = json.loads((second / "adapter_config.json").read_text())
        configuration["base_model_name_or_path"] = "wrong/model"
        (second / "adapter_config.json").write_text(json.dumps(configuration))
    with pytest.raises(ValueError):
        _publish(root)
    for checkpoint in (first, second):
        assert not (checkpoint / "request.json").exists()
        assert not (checkpoint / "complete.json").exists()
        assert not (checkpoint / "provenance.json").exists()
        with pytest.raises(ValueError):
            require_complete(checkpoint)


def test_checkpoint_cannot_overwrite_existing_metadata(tmp_path: Path) -> None:
    root = _parent(tmp_path)
    checkpoint = _checkpoint(root, 10)
    (checkpoint / "provenance.json").write_text("user metadata")
    with pytest.raises(FileExistsError):
        _publish(root)
    assert (checkpoint / "provenance.json").read_text() == "user metadata"
    assert not (checkpoint / "complete.json").exists()


def test_empty_periodic_checkpoint_set_is_valid(tmp_path: Path) -> None:
    assert _publish(_parent(tmp_path)) == ()


def test_parent_request_tampering_cannot_certify_checkpoints(tmp_path: Path) -> None:
    root = _parent(tmp_path)
    checkpoint = _checkpoint(root, 10)
    manifest = json.loads((root / "request.json").read_text())
    manifest["request"]["base_model"] = "changed/model"
    (root / "request.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="request digest mismatch"):
        _publish(root)
    assert not (checkpoint / "complete.json").exists()


def test_checkpoint_publication_never_resolves_model_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "training"
    begin_run(root, kind="sft", request=TrainingRequest(base_model_revision=None))
    checkpoint = _checkpoint(root, 10)
    configuration = json.loads((checkpoint / "adapter_config.json").read_text())
    configuration["revision"] = None
    (checkpoint / "adapter_config.json").write_text(json.dumps(configuration))
    monkeypatch.setattr(
        "llm_rlvr.train.configuration.resolve_model_revision",
        lambda *_: pytest.fail("model metadata fetched"),
    )
    publish_checkpoints(
        root, base_model="org/model", base_model_revision=None, training_population_ids=None
    )
    provenance = CheckpointProvenance.model_validate_json(
        (checkpoint / "provenance.json").read_text()
    )
    assert provenance.base_model_revision is None
    assert provenance.training_population_ids is None
    require_complete(checkpoint)
