"""Runtime support for Modal entrypoints: provenance, dataset contexts, and Volume dispatch."""

from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from llm_rlvr._base import Frozen
from llm_rlvr.artifacts.provenance import git_commit, git_is_dirty
from llm_rlvr.artifacts.runs import request_digest
from llm_rlvr.data.artifacts import BIRD_ARCHIVE, SPIDER_HF_REVISION
from llm_rlvr.data.bird import ensure_bird_dev, load_bird_examples
from llm_rlvr.data.models import DatabaseSchema, Example
from llm_rlvr.data.spider import ensure_spider_databases, load_examples, load_schemas


@dataclass(frozen=True)
class EvalContext:
    examples: list[Example]
    schemas: dict[str, DatabaseSchema]
    train_pool: list[Example]
    dataset: str
    dataset_revision: str
    db_archive_sha256: str | None
    data_source: Literal["archive", "override", "legacy"]
    database_root: Path
    split: str


@dataclass(frozen=True)
class SpiderTrainContext:
    examples: list[Example]
    schemas: dict[str, DatabaseSchema]
    dataset_revision: str
    db_archive_sha256: str | None
    data_source: Literal["archive", "override", "legacy"]
    database_root: Path


class CommitVolume(Protocol):
    def commit(self) -> None: ...


class RemoteFunction(Protocol):
    def spawn(self, **kwargs: Any) -> Any: ...

    def remote(self, **kwargs: Any) -> Any: ...


def git_kwargs() -> dict[str, str | bool]:
    """Git commit + dirty flag for a local entrypoint to pass through to the GPU function
    (``.git`` is stripped from the Modal image, so the remote side cannot read them itself)."""
    return {"commit": git_commit(), "dirty": git_is_dirty()}


def _validate_limit(limit: int | None) -> None:
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")


def load_eval_context(
    *,
    dataset: str,
    split: str,
    data_root: Path,
    few_shot_k: int = 0,
    limit: int | None = None,
) -> EvalContext:
    _validate_limit(limit)
    if few_shot_k < 0:
        raise ValueError("few_shot_k must be nonnegative")
    if dataset == "bird" and few_shot_k:
        raise ValueError("BIRD few-shot generation is unsupported")
    if dataset == "spider" and split not in ("dev", "validation"):
        raise ValueError("Spider evaluation requires dev or validation split")
    if dataset == "spider":
        dbs = ensure_spider_databases(data_root / "spider")
        schemas = load_schemas(dbs.tables_json)
        examples = list(load_examples(split))
        train_pool = list(load_examples("train")) if few_shot_k else []
        dataset_revision = SPIDER_HF_REVISION
        db_sha = dbs.archive_sha256
        data_source = dbs.source
        database_root = dbs.database_root
    elif dataset == "bird":
        # BIRD ships only a dev split; loading it under any other `split` label would stamp
        # wrong provenance on the artifact.
        if split != "dev":
            raise ValueError(f"bird has only a 'dev' split, got {split!r}")
        bird = ensure_bird_dev(data_root / "bird")
        schemas = load_schemas(bird.tables_json)
        examples = list(load_bird_examples(bird.dev_json))
        train_pool = []
        dataset_revision = BIRD_ARCHIVE.revision
        db_sha = bird.archive_sha256
        data_source = bird.source
        database_root = bird.database_root
    else:
        raise ValueError(f"dataset must be 'spider' or 'bird', got {dataset!r}")

    if limit is not None:
        examples = examples[:limit]

    return EvalContext(
        examples=examples,
        schemas=schemas,
        train_pool=train_pool,
        dataset=dataset,
        dataset_revision=dataset_revision,
        db_archive_sha256=db_sha,
        data_source=data_source,
        database_root=database_root,
        split=split,
    )


def load_spider_train_context(*, data_root: Path, limit: int | None = None) -> SpiderTrainContext:
    _validate_limit(limit)
    dbs = ensure_spider_databases(data_root / "spider")
    schemas = load_schemas(dbs.tables_json)
    examples = list(load_examples("train"))
    if limit is not None:
        examples = examples[:limit]
    return SpiderTrainContext(
        examples=examples,
        schemas=schemas,
        dataset_revision=SPIDER_HF_REVISION,
        db_archive_sha256=dbs.archive_sha256,
        data_source=dbs.source,
        database_root=dbs.database_root,
    )


def volume_path(root: str | Path, *parts: str | Path) -> Path:
    base = Path(root).resolve()
    for part in parts:
        validate_volume_relative(str(part))
    target = base.joinpath(*parts).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"Volume path escapes {base}")
    return target


def commit_volume(volume: CommitVolume, message: str) -> None:
    volume.commit()
    print(message)


LAUNCH_APPROVAL_FLAG = "--approved"


def approval_refusal(label: str, argv: list[str]) -> str:
    approved_command = shlex.join(["modal", *argv[1:], LAUNCH_APPROVAL_FLAG])
    return (
        f"refusing to launch {label}: this run spends money on Modal GPUs.\n"
        "Stop and ask Adam to approve this run and its rough cost.\n"
        f"Once approved, run: {approved_command}"
    )


def dispatch_remote(
    fn: RemoteFunction,
    kwargs: Frozen,
    *,
    approved: bool,
    spawn: bool,
    label: str,
    app_name: str,
) -> None:
    if not approved:
        raise SystemExit(approval_refusal(label, sys.argv))
    if spawn:
        call = fn.spawn(**kwargs.model_dump())
        print(f"{label} dispatched (spawn): function call {call.object_id}")
        print(f"monitor: modal app logs {app_name}  (artifacts land on the text2sql-runs Volume)")
    else:
        print(f"{label} complete: {fn.remote(**kwargs.model_dump())}")


def validate_volume_relative(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path == Path() or "\x00" in value:
        raise ValueError(f"expected a relative Volume path without traversal, got {value!r}")
    return value


class ModelRequest(Frozen):
    base_model: str = Field(min_length=1)
    model_revision: str | None = Field(default=None, min_length=1)
    commit: str | None = None
    dirty: bool | None = None

    @field_validator("base_model")
    @classmethod
    def nonempty_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("base_model must not be blank")
        return value


class GenerationRequest(ModelRequest):
    dataset: Literal["spider", "bird"] = "spider"
    split: Literal["dev", "validation"] = "dev"
    few_shot_k: int = Field(default=0, ge=0)
    few_shot_seed: int = 0
    prompt_style: Literal["cot", "direct"] = "cot"
    adapter: str | None = None
    limit: int | None = Field(default=None, gt=0)
    max_new_tokens: int = Field(default=512, gt=0)
    temperature: float = Field(default=0, ge=0, allow_inf_nan=False)
    sampling_seed: int = 0
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

    @field_validator("adapter")
    @classmethod
    def valid_adapter(cls, value: str | None) -> str | None:
        return validate_volume_relative(value) if value is not None else None

    @model_validator(mode="after")
    def bird_options(self) -> Self:
        if self.dataset == "bird" and (self.split != "dev" or self.few_shot_k):
            raise ValueError("BIRD requires dev split and does not support few-shot generation")
        return self


class TraceRequest(ModelRequest):
    n: int = Field(default=8, gt=0)
    temperature: float = Field(default=0.8, ge=0, allow_inf_nan=False)
    top_p: float = Field(default=0.95, gt=0, le=1)
    seed: int = 0
    max_new_tokens: int = Field(default=1024, gt=0)
    limit: int | None = Field(default=None, gt=0)
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class SftRequest(ModelRequest):
    traces: str = "sft/traces.jsonl"
    trace_manifest: str = "sft/traces.manifest.json"
    run_id: str = Field(default="sft-r32", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    epochs: int = Field(default=3, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0, allow_inf_nan=False)
    seed: int = 0

    _paths = field_validator("traces", "trace_manifest")(validate_volume_relative)


class GrpoRequest(Frozen):
    run_id: str = Field(default="grpo-s0", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    warm_start: str = "sft/sft-r32"
    base_model: str = Field(min_length=1)
    base_model_revision: str | None = Field(default=None, min_length=1)
    reward_arm: Literal["shaped", "near_binary"] = "shaped"
    learning_rate: float = Field(default=1e-5, gt=0, allow_inf_nan=False)
    seed: int = 0
    dev_size: int | None = Field(default=None, ge=0)
    limit: int | None = Field(default=None, gt=0)
    commit: str | None = None
    dirty: bool | None = None

    _warm_start = field_validator("warm_start")(validate_volume_relative)

    @model_validator(mode="after")
    def nonempty_training_split(self) -> Self:
        if not self.base_model.strip():
            raise ValueError("base_model must not be blank")
        if self.limit is not None:
            from llm_rlvr.train.grpo import GrpoHyperparams

            dev_size = GrpoHyperparams().eval_slice_size if self.dev_size is None else self.dev_size
            if dev_size >= self.limit:
                raise ValueError("dev_size must leave at least one training example under limit")
        return self


class GenerationRunRequest(Frozen):
    generation: GenerationRequest
    resolved_model_revision: str | None = None
    adapter_provenance_sha256: str | None = None
    adapter_sha256: str | None = None
    selection_policy: Literal["all", "prefix"] = "all"
    data_source: Literal["archive", "override", "legacy"]
    db_archive_sha256: str | None


class TraceRunRequest(Frozen):
    sampling: TraceRequest
    resolved_model_revision: str | None = None
    selection_policy: Literal["all", "prefix"] = "all"
    data_source: Literal["archive", "override", "legacy"]
    db_archive_sha256: str | None


def managed_run_path(root: str | Path, kind: str, request: ModelRequest) -> Path:
    import re

    tag = re.sub(r"[^a-z0-9-]+", "-", request.base_model.rsplit("/", 1)[-1].lower()).strip("-")
    run_id = getattr(request, "run_id", None) or f"{tag}-{request_digest(request)[:16]}"
    return volume_path(root, kind, run_id)
