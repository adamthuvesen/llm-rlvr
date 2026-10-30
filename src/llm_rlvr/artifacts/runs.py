"""Record requests before execution and publish completion after validating all outputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, field_validator, model_validator

from llm_rlvr._base import Frozen
from llm_rlvr.artifacts.io import atomic_write_json, sha256_of

REQUEST_FILENAME = "request.json"
COMPLETION_FILENAME = "complete.json"


class RunRequestManifest(Frozen):
    format_version: Literal[1] = 1
    kind: str
    request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    request: dict[str, JsonValue]

    @model_validator(mode="after")
    def valid_request_digest(self) -> Self:
        canonical = json.dumps(self.request, sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != self.request_sha256:
            raise ValueError("run request digest mismatch")
        return self


class ArtifactDigest(Frozen):
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(gt=0)

    @field_validator("path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        path = Path(value)
        if not value or path.is_absolute() or ".." in path.parts or path == Path():
            raise ValueError("artifact path must be relative without traversal")
        if value in (REQUEST_FILENAME, COMPLETION_FILENAME):
            raise ValueError("run control files cannot be output artifacts")
        return value


class RunCompletionManifest(Frozen):
    format_version: Literal[1] = 1
    request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifacts: tuple[ArtifactDigest, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_artifacts(self) -> Self:
        if len({artifact.path for artifact in self.artifacts}) != len(self.artifacts):
            raise ValueError("duplicate run artifact paths")
        return self


def request_digest(request: BaseModel) -> str:
    canonical = json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def begin_run(output_dir: str | Path, *, kind: str, request: BaseModel) -> RunRequestManifest:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    manifest = RunRequestManifest(
        kind=kind,
        request_sha256=request_digest(request),
        request=TypeAdapter(dict[str, JsonValue]).validate_python(request.model_dump(mode="json")),
    )
    atomic_write_json(manifest, root / REQUEST_FILENAME)
    return manifest


def _artifact(root: Path, path: str | Path) -> ArtifactDigest:
    supplied = Path(path)
    if supplied.is_absolute() or (
        supplied.exists() and supplied.resolve().is_relative_to(root.resolve())
    ):
        target = supplied
    else:
        target = root / supplied
    resolved = target.resolve()
    if not resolved.is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError(f"run artifact must be a file inside {root}: {path}")
    relative = resolved.relative_to(root.resolve()).as_posix()
    if relative in (REQUEST_FILENAME, COMPLETION_FILENAME):
        raise ValueError("run control files cannot be output artifacts")
    return ArtifactDigest(path=relative, sha256=sha256_of(target), size_bytes=target.stat().st_size)


def publish_complete(
    output_dir: str | Path, *, artifacts: Sequence[str | Path]
) -> RunCompletionManifest:
    root = Path(output_dir)
    request = RunRequestManifest.model_validate_json(
        (root / REQUEST_FILENAME).read_text(encoding="utf-8")
    )
    outputs = tuple(_artifact(root, path) for path in artifacts)
    if len({artifact.path for artifact in outputs}) != len(outputs):
        raise ValueError("duplicate run artifact paths")
    manifest = RunCompletionManifest(request_sha256=request.request_sha256, artifacts=outputs)
    atomic_write_json(manifest, root / COMPLETION_FILENAME)
    return manifest


def require_complete(path: str | Path) -> None:
    """Validate managed inputs. Existing unmanaged artifacts remain readable."""
    target = Path(path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"input not found: {target}")
    first = target if target.is_dir() else target.parent
    for root in (first, *first.parents):
        if not (root / REQUEST_FILENAME).is_file():
            continue
        if target.is_dir() and target.resolve() != root.resolve():
            raise ValueError(f"managed subdirectory needs its own completion manifest: {target}")
        completion_path = root / COMPLETION_FILENAME
        if not completion_path.is_file():
            raise ValueError(f"managed run is incomplete: {root}")
        request = RunRequestManifest.model_validate_json(
            (root / REQUEST_FILENAME).read_text(encoding="utf-8")
        )
        completion = RunCompletionManifest.model_validate_json(
            completion_path.read_text(encoding="utf-8")
        )
        canonical = json.dumps(request.request, sort_keys=True, separators=(",", ":"))
        if (
            hashlib.sha256(canonical.encode("utf-8")).hexdigest() != request.request_sha256
            or completion.request_sha256 != request.request_sha256
        ):
            raise ValueError(f"run request digest mismatch: {root}")
        for expected in completion.artifacts:
            if _artifact(root, expected.path) != expected:
                raise ValueError(f"run artifact digest mismatch: {root / expected.path}")
        if target.is_dir():
            published = {item.path for item in completion.artifacts}
            model_files = [*target.glob("adapter_model*"), target / "adapter_config.json"]
            for model_file in model_files:
                if (
                    model_file.is_file()
                    and model_file.resolve().relative_to(root.resolve()).as_posix() not in published
                ):
                    raise ValueError(f"adapter input is not a published run artifact: {model_file}")
        if target.is_file() and target.resolve().relative_to(root.resolve()).as_posix() not in {
            item.path for item in completion.artifacts
        }:
            raise ValueError(f"input is not a published run artifact: {target}")
        return
