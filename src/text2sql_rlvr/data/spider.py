"""Spider data layer: typed examples, schema parsing, and checksum-gated DB resolution.

`load_dataset("xlangai/spider")` returns only question/SQL parquet; the SQLite databases are a
separate, checksummed artifact (design D2). This module parses Spider's `tables.json` into the
shared `DatabaseSchema`, loads examples at the pinned revision, and resolves a `db_id` to its
verified `.sqlite` file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from text2sql_rlvr.data.artifacts import SPIDER_HF_REPO, SPIDER_HF_REVISION
from text2sql_rlvr.data.models import Column, DatabaseSchema, Example, ForeignKey, Table

# Spider's parquet "dev" split is named `validation`.
_SPLIT_ALIASES = {"dev": "validation", "validation": "validation", "train": "train"}


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


def sha256_of(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA256 of a file (archives are large)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> None:
    """Raise loudly if a file's checksum does not match the pinned value (the trust anchor)."""
    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(
            f"checksum mismatch for {path}: expected {expected}, got {actual}. "
            "Refusing to use an unverified Spider database archive."
        )


def resolve_database(db_id: str, db_root: str | Path) -> Path:
    """Return the `.sqlite` path for `db_id`, failing loudly if it is missing.

    `db_root` is the extracted Spider `database/` directory (`<db_root>/<db_id>/<db_id>.sqlite`).
    """
    path = Path(db_root) / db_id / f"{db_id}.sqlite"
    if not path.exists():
        raise FileNotFoundError(
            f"database for db_id={db_id!r} not found at {path}. The Spider SQLite databases are a "
            "separate checksummed artifact — see RUNBOOK.md / design R1."
        )
    return path
