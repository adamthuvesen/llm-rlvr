"""Spider data layer: typed examples, schema parsing, and checksum-gated DB resolution.

`load_dataset("xlangai/spider")` returns only question/SQL parquet; the SQLite databases are a
separate, checksummed artifact (design D2). This module parses Spider's `tables.json` into the
shared `DatabaseSchema`, loads examples at the pinned revision, and resolves a `db_id` to its
verified `.sqlite` file.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any

from llm_rlvr_sql._base import Frozen
from llm_rlvr_sql.data.artifacts import (
    SPIDER_DB_ARCHIVE,
    SPIDER_HF_REPO,
    SPIDER_HF_REVISION,
)
from llm_rlvr_sql.data.models import Column, DatabaseSchema, Example, ForeignKey, Table
from llm_rlvr_sql.data.paths import verify_sha256

# Spider's parquet "dev" split is named `validation`.
_SPLIT_ALIASES = {"dev": "validation", "validation": "validation", "train": "train"}

# Points at an already-extracted `spider_data/` dir — an offline escape hatch that skips the
# download (e.g. for the very first baseline run before the HF mirror is uploaded).
_SPIDER_ROOT_ENV = "TEXT2SQL_SPIDER_ROOT"
_DEFAULT_CACHE = Path("data/spider")


def parse_schema(entry: dict[str, Any]) -> DatabaseSchema:
    """Parse one Spider `tables.json` entry into a `DatabaseSchema`.

    Spider indexes a global column list whose element 0 is the special ``*`` column (table -1);
    `primary_keys` / `foreign_keys` reference that global index.
    """
    table_names = entry["table_names_original"]
    columns = entry["column_names_original"]  # [[table_idx, name], ...]
    column_types = entry["column_types"]

    per_table: list[list[Column]] = [[] for _ in table_names]
    for (table_idx, name), col_type in zip(columns, column_types, strict=True):
        if table_idx < 0:  # the "*" column
            continue
        per_table[table_idx].append(Column(name=name, type=col_type))

    pk_by_table: dict[int, list[str]] = {}
    for pk in entry.get("primary_keys", []):
        for col_idx in pk if isinstance(pk, list) else [pk]:
            table_idx, name = columns[col_idx]
            pk_by_table.setdefault(table_idx, []).append(name)

    tables = tuple(
        Table(
            name=table_names[i],
            columns=tuple(per_table[i]),
            primary_key=tuple(pk_by_table.get(i, [])),
        )
        for i in range(len(table_names))
    )

    foreign_keys = []
    for from_idx, to_idx in entry.get("foreign_keys", []):
        from_table_idx, from_col = columns[from_idx]
        to_table_idx, to_col = columns[to_idx]
        foreign_keys.append(
            ForeignKey(
                from_table=table_names[from_table_idx],
                from_column=from_col,
                to_table=table_names[to_table_idx],
                to_column=to_col,
            )
        )

    return DatabaseSchema(db_id=entry["db_id"], tables=tables, foreign_keys=tuple(foreign_keys))


def load_schemas(tables_json: str | Path) -> dict[str, DatabaseSchema]:
    """Parse a Spider `tables.json` file into `{db_id: DatabaseSchema}`."""
    entries = json.loads(Path(tables_json).read_text(encoding="utf-8"))
    return {entry["db_id"]: parse_schema(entry) for entry in entries}


def load_examples(split: str, *, revision: str = SPIDER_HF_REVISION) -> tuple[Example, ...]:
    """Load Spider examples at the pinned revision (network; integration-tested).

    `split` accepts `train` or `dev`/`validation`. Returns strictly-typed `Example`s.
    """
    if split not in _SPLIT_ALIASES:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(_SPLIT_ALIASES)}")
    from datasets import load_dataset  # lazy: keeps import light for non-loading callers

    rows = load_dataset(SPIDER_HF_REPO, revision=revision, split=_SPLIT_ALIASES[split])
    return tuple(
        Example(
            db_id=row["db_id"],
            question=row["question"],
            gold_sql=row["query"],
            split=split,
            dataset="spider",
        )
        for row in rows
    )


class SpiderDatabases(Frozen):
    """The local Spider data tree: the extracted root, its `database/` dir, and `tables.json`."""

    root: Path
    database_root: Path
    tables_json: Path


def ensure_spider_databases(cache_dir: str | Path | None = None) -> SpiderDatabases:
    """Return the local Spider databases, fetching + verifying + extracting if needed.

    Resolution order:
      1. ``$TEXT2SQL_SPIDER_ROOT`` — an already-extracted ``spider_data/`` dir (offline hatch);
      2. the pinned, checksummed ``SPIDER_DB_ARCHIVE`` from HF, downloaded + verified + extracted.
    """
    override = os.environ.get(_SPIDER_ROOT_ENV)
    if override:
        return _spider_databases_at(Path(override))

    cache = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
    extracted = cache / "spider_data"
    if not extracted.exists():
        from huggingface_hub import hf_hub_download

        archive = hf_hub_download(
            repo_id=SPIDER_DB_ARCHIVE.repo,
            filename=SPIDER_DB_ARCHIVE.filename,
            revision=SPIDER_DB_ARCHIVE.revision,
            repo_type="dataset",
        )
        verify_sha256(archive, SPIDER_DB_ARCHIVE.sha256)
        cache.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(cache)
    return _spider_databases_at(extracted)


def _spider_databases_at(root: Path) -> SpiderDatabases:
    database_root = root / "database"
    tables_json = root / "tables.json"
    if not database_root.is_dir() or not tables_json.is_file():
        raise FileNotFoundError(
            f"expected a spider_data layout at {root} (database/ + tables.json)"
        )
    return SpiderDatabases(root=root, database_root=database_root, tables_json=tables_json)
