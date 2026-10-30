"""Training preflight and the configuration actually passed to training libraries."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from importlib import metadata
from math import isfinite
from pathlib import Path
from typing import TypeVar

from pydantic import ConfigDict, Field, JsonValue, field_validator

from llm_rlvr._base import Frozen
from llm_rlvr.artifacts.io import sha256_of

ConfigT = TypeVar("ConfigT")


class RuntimeConfigRecord(Frozen):
    requested_trl: dict[str, JsonValue]
    effective_trl: dict[str, JsonValue]
    requested_lora: dict[str, JsonValue]
    effective_lora: dict[str, JsonValue]

    @field_validator("requested_trl", "effective_trl", "requested_lora", "effective_lora")
    @classmethod
    def finite_values(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return configuration_values(value)


class AdapterFile(Frozen):
    path: str
    sha256: str


class AdapterIdentity(Frozen):
    base_model: str
    base_model_revision: str | None = None
    configuration: dict[str, JsonValue]
    files: tuple[AdapterFile, ...]
    sha256: str
    provenance_sha256: str | None = None
    train_source_ids: tuple[str, ...] | None = None
    sft_train_source_ids: tuple[str, ...] | None = None
    source_identity_policy: str | None = None


class _AdapterConfiguration(Frozen):
    model_config = ConfigDict(frozen=True, extra="ignore")
    base_model_name_or_path: str = Field(min_length=1)
    revision: str | None = None
    r: int = Field(gt=0, strict=True)
    lora_alpha: float = Field(gt=0, allow_inf_nan=False)
    lora_dropout: float = Field(ge=0, lt=1, allow_inf_nan=False)


class AdapterSourceProvenance(Frozen):
    model_config = ConfigDict(frozen=True, extra="ignore")
    base_model: str = Field(min_length=1)
    base_model_revision: str | None = None
    train_source_ids: tuple[str, ...] | None = None
    trace_dataset_path: str | None = None
    sft_train_source_ids: tuple[str, ...] | None = None
    source_identity_policy: str | None = None


class _AdapterWeightIndex(Frozen):
    model_config = ConfigDict(frozen=True, extra="ignore")
    weight_map: dict[str, str] = Field(min_length=1)


def build_training_config(
    config_type: type[ConfigT], desired: Mapping[str, object], *, package: str = "trl"
) -> ConfigT:
    valid = {field.name for field in dataclasses.fields(config_type)}  # type: ignore[arg-type]
    unsupported = sorted(set(desired) - valid)
    if unsupported:
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "not-installed"
        raise ValueError(
            f"{config_type.__name__} on {package}={version} does not support requested fields "
            f"{unsupported}. Install the project's supported train dependencies or update "
            "the requested configuration explicitly. No fields were dropped."
        )
    return config_type(**desired)


def json_value(value: object) -> JsonValue:
    if isinstance(value, Enum):
        return json_value(value.value)
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("configuration values must be finite")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("configuration keys must be strings")
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [json_value(item) for item in sorted(value)]
    raise TypeError(f"configuration value {type(value).__name__} is not JSON serializable")


def configuration_values(values: Mapping[str, object]) -> dict[str, JsonValue]:
    return {key: json_value(value) for key, value in values.items()}


def effective_values(config: object, requested: Mapping[str, object]) -> dict[str, JsonValue]:
    return {key: json_value(getattr(config, key)) for key in requested}


def resolve_model_revision(model: str, revision: str | None) -> str | None:
    """Resolve remote refs once, then pass the resulting commit to the actual model load."""
    if Path(model).is_dir():
        return revision
    if revision and len(revision) == 40 and all(char in "0123456789abcdef" for char in revision):
        return revision
    from huggingface_hub import HfApi

    resolved = HfApi().model_info(model, revision=revision).sha
    if not resolved:
        raise ValueError(f"could not resolve a commit for model {model!r}, revision={revision!r}")
    return resolved


def inspect_adapter(
    adapter_path: str | Path,
    *,
    base_model: str,
    base_model_revision: str | None,
) -> AdapterIdentity:
    """Check declared base compatibility and hash the actual config and weight files."""
    root = Path(adapter_path)
    config_path = root / "adapter_config.json"
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(configuration, dict):
        raise ValueError(f"adapter configuration must be an object: {config_path}")
    parsed = _AdapterConfiguration.model_validate(configuration)
    declared_model = parsed.base_model_name_or_path
    if declared_model != base_model:
        raise ValueError(
            f"adapter base model {declared_model!r} differs from requested base {base_model!r}"
        )
    declared_revision = parsed.revision
    provenance_path = root / "provenance.json"
    provenance: AdapterSourceProvenance | None = None
    if provenance_path.is_file():
        provenance = AdapterSourceProvenance.model_validate_json(
            provenance_path.read_text(encoding="utf-8")
        )
        if provenance.base_model != base_model:
            raise ValueError("adapter provenance base model differs from requested base")
        provenance_revision = provenance.base_model_revision
        if declared_revision and provenance_revision and declared_revision != provenance_revision:
            raise ValueError("adapter config and provenance declare different base revisions")
        declared_revision = provenance_revision or declared_revision
    if declared_revision and base_model_revision and declared_revision != base_model_revision:
        raise ValueError(
            f"adapter base revision {declared_revision!r} differs from requested "
            f"base revision {base_model_revision!r}. Use the adapter's original base revision"
        )
    weights = {
        path for path in root.glob("adapter_model*") if path.suffix in {".safetensors", ".bin"}
    }
    indices = sorted(root.glob("adapter_model*.index.json"))
    for index_path in indices:
        index = _AdapterWeightIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
        for name in index.weight_map.values():
            weight_path = root / name
            if not weight_path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"adapter weight index points outside adapter directory: {name!r}")
            if not weight_path.is_file():
                raise ValueError(f"adapter weight index references missing weights: {name!r}")
            weights.add(weight_path)
    if not weights:
        raise ValueError(f"adapter has no adapter_model weights: {root}")
    files = tuple(
        AdapterFile(path=path.relative_to(root).as_posix(), sha256=sha256_of(path))
        for path in [config_path, *indices, *sorted(weights)]
    )
    digest = hashlib.sha256(
        json.dumps([file.model_dump() for file in files], sort_keys=True).encode()
    ).hexdigest()
    return AdapterIdentity(
        base_model=base_model,
        base_model_revision=declared_revision,
        configuration=configuration_values(configuration),
        files=files,
        sha256=digest,
        provenance_sha256=sha256_of(provenance_path) if provenance is not None else None,
        train_source_ids=provenance.train_source_ids if provenance is not None else None,
        sft_train_source_ids=(
            provenance.train_source_ids
            if provenance is not None and provenance.trace_dataset_path is not None
            else provenance.sft_train_source_ids
            if provenance is not None
            else None
        ),
        source_identity_policy=provenance.source_identity_policy
        if provenance is not None
        else None,
    )


def resolve_adapter_base(
    adapter_path: str | Path,
    *,
    base_model: str,
    requested_revision: str | None,
) -> tuple[str | None, AdapterIdentity]:
    """Use the declared adapter base by default and compare explicitly requested refs by commit."""
    identity = inspect_adapter(adapter_path, base_model=base_model, base_model_revision=None)
    revision = resolve_model_revision(
        base_model, requested_revision or identity.base_model_revision
    )
    if identity.base_model_revision:
        expected = (
            revision
            if requested_revision is None or requested_revision == identity.base_model_revision
            else resolve_model_revision(base_model, identity.base_model_revision)
        )
        if revision != expected:
            raise ValueError("resolved base revision differs from warm-start adapter revision")
    return revision, identity
