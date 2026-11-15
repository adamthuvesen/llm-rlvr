"""Managed outputs become consumable only after complete, verified publication."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_rlvr._base import Frozen
from llm_rlvr.artifacts.io import atomic_write_text
from llm_rlvr.artifacts.runs import (
    ArtifactDigest,
    RunCompletionManifest,
    begin_run,
    publish_complete,
    require_complete,
)


class Request(Frozen):
    model: str = "model"
    revision: str = "abc"


def test_incomplete_managed_output_rejected_then_completed(tmp_path: Path) -> None:
    root = tmp_path / "run"
    request = begin_run(root, kind="generation", request=Request())
    artifact = atomic_write_text(root / "output.json", '{"valid": true}')
    with pytest.raises(ValueError, match="incomplete"):
        require_complete(artifact)
    completion = publish_complete(root, artifacts=[artifact])
    assert completion.request_sha256 == request.request_sha256
    require_complete(artifact)
    artifact.write_text("tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        require_complete(artifact)


def test_existing_run_rejected_without_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "run"
    begin_run(root, kind="generation", request=Request())
    before = (root / "request.json").read_bytes()
    with pytest.raises(FileExistsError):
        begin_run(root, kind="generation", request=Request(model="other"))
    assert (root / "request.json").read_bytes() == before


def test_unmanaged_legacy_artifacts_remain_readable(tmp_path: Path) -> None:
    artifact = atomic_write_text(tmp_path / "legacy.json", "{}")
    require_complete(artifact)


def test_unlisted_checkpoint_cannot_inherit_parent_completion(tmp_path: Path) -> None:
    root = tmp_path / "run"
    begin_run(root, kind="sft", request=Request())
    artifact = atomic_write_text(root / "provenance.json", "{}")
    checkpoint = root / "checkpoint-1"
    checkpoint.mkdir()
    atomic_write_text(checkpoint / "adapter_model.safetensors", "unlisted")
    publish_complete(root, artifacts=[artifact])
    with pytest.raises(ValueError, match="own completion manifest"):
        require_complete(checkpoint)
    with pytest.raises(ValueError, match="not a published"):
        require_complete(checkpoint / "adapter_model.safetensors")


def test_unlisted_adapter_weights_rejected_at_completed_root(tmp_path: Path) -> None:
    root = tmp_path / "run"
    begin_run(root, kind="sft", request=Request())
    artifact = atomic_write_text(root / "provenance.json", "{}")
    publish_complete(root, artifacts=[artifact])
    atomic_write_text(root / "adapter_model.safetensors", "unlisted")
    with pytest.raises(ValueError, match="not a published"):
        require_complete(root)


def test_relative_output_paths_do_not_duplicate_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path("runs/test")
    begin_run(root, kind="generation", request=Request())
    artifact = atomic_write_text(root / "output.json", "{}")
    publish_complete(root, artifacts=[artifact])
    require_complete(artifact)


@pytest.mark.parametrize("path", ["../outside", "/outside"])
def test_manifest_artifacts_reject_escaping_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactDigest(path=path, sha256="a" * 64, size_bytes=1)


def test_manifest_rejects_duplicate_artifacts_and_bad_digests() -> None:
    artifact = ArtifactDigest(path="output", sha256="a" * 64, size_bytes=1)
    with pytest.raises(ValidationError, match="duplicate"):
        RunCompletionManifest(request_sha256="b" * 64, artifacts=(artifact, artifact))
    with pytest.raises(ValidationError):
        ArtifactDigest(path="output", sha256="incorrect", size_bytes=1)


def test_failed_output_validation_never_publishes_completion(tmp_path: Path) -> None:
    root = tmp_path / "run"
    begin_run(root, kind="sft", request=Request())
    atomic_write_text(root / "empty", "")
    with pytest.raises(ValidationError):
        publish_complete(root, artifacts=["empty"])
    assert not (root / "complete.json").exists()


def test_atomic_outputs_preserve_existing_file_and_clean_staging(tmp_path: Path) -> None:
    path = atomic_write_text(tmp_path / "output", "old")
    with pytest.raises(FileExistsError):
        atomic_write_text(path, "new")
    assert path.read_text() == "old"
    atomic_write_text(path, "new", overwrite=True)
    assert path.read_text() == "new"
    assert list(tmp_path.iterdir()) == [path]


def test_checkpoint_symlink_cannot_bypass_managed_parent(tmp_path: Path) -> None:
    root = tmp_path / "run"
    begin_run(root, kind="sft", request=Request())
    checkpoint = root / "checkpoint-1"
    checkpoint.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(checkpoint, target_is_directory=True)
    with pytest.raises(ValueError, match="own completion manifest"):
        require_complete(alias)
