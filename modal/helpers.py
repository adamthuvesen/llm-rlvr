"""Shared helpers for ``modal_app.py`` entrypoints."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from llm_rlvr.data.artifacts import BIRD_ARCHIVE, SPIDER_DB_ARCHIVE, SPIDER_HF_REVISION
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
    db_archive_sha256: str
    database_root: Path
    split: str


@dataclass(frozen=True)
class SpiderTrainContext:
    examples: list[Example]
    schemas: dict[str, DatabaseSchema]
    dataset_revision: str
    db_archive_sha256: str
    database_root: Path


class CommitVolume(Protocol):
    def commit(self) -> None: ...


def git_kwargs() -> dict[str, str | bool]:
    commit, dirty = git_provenance()
    return {"commit": commit or "unknown", "dirty": False if dirty is None else dirty}


def git_provenance() -> tuple[str | None, bool | None]:
    """Git commit and dirty flag for Modal provenance (``.git`` is stripped from the image)."""
    return _local_git_commit(), _local_git_dirty()


def _local_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or None
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _local_git_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_eval_context(
    *,
    dataset: str,
    split: str,
    data_root: Path,
    few_shot_k: int = 0,
    limit: int | None = None,
) -> EvalContext:
    if dataset == "spider":
        dbs = ensure_spider_databases(data_root / "spider")
        schemas = load_schemas(dbs.tables_json)
        examples = list(load_examples(split))
        train_pool = list(load_examples("train")) if few_shot_k else []
        dataset_revision = SPIDER_HF_REVISION
        db_sha = SPIDER_DB_ARCHIVE.sha256
        database_root = dbs.database_root
    elif dataset == "bird":
        bird = ensure_bird_dev(data_root / "bird")
        schemas = load_schemas(bird.tables_json)
        examples = list(load_bird_examples(bird.dev_json))
        train_pool = []
        dataset_revision = BIRD_ARCHIVE.revision
        db_sha = BIRD_ARCHIVE.sha256
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
        database_root=database_root,
        split=split,
    )


def load_spider_train_context(*, data_root: Path, limit: int | None = None) -> SpiderTrainContext:
    dbs = ensure_spider_databases(data_root / "spider")
    schemas = load_schemas(dbs.tables_json)
    examples = list(load_examples("train"))
    if limit is not None:
        examples = examples[:limit]
    return SpiderTrainContext(
        examples=examples,
        schemas=schemas,
        dataset_revision=SPIDER_HF_REVISION,
        db_archive_sha256=SPIDER_DB_ARCHIVE.sha256,
        database_root=dbs.database_root,
    )


def volume_path(root: str | Path, *parts: str | Path) -> Path:
    return Path(root).joinpath(*parts)


def commit_volume(volume: CommitVolume, message: str) -> None:
    volume.commit()
    print(message)


def dispatch_remote(
    fn: Callable[..., Any],
    kwargs: Mapping[str, Any],
    *,
    spawn: bool,
    label: str,
    app_name: str,
) -> None:
    if spawn:
        call = fn.spawn(**dict(kwargs))
        print(f"{label} dispatched (spawn): function call {call.object_id}")
        print(f"monitor: modal app logs {app_name}  (artifacts land on the text2sql-runs Volume)")
    else:
        print(f"{label} complete: {fn.remote(**dict(kwargs))}")
