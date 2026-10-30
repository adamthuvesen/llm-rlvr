"""Strict, typed records for examples and database schemas.

Loader-agnostic: a Spider/BIRD loader parses its own on-disk format into these models, so
everything downstream (prompt building, serialization, scoring) sees one shape.
"""

from __future__ import annotations

import hashlib
import json
from typing import Self

from pydantic import model_validator

from llm_rlvr._base import Frozen


class Column(Frozen):
    """A single column: its name and SQL-ish type as the dataset declares it."""

    name: str
    type: str


class Table(Frozen):
    """A table, its columns (in declared order), and its primary-key column names."""

    name: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...] = ()

    @model_validator(mode="after")
    def valid_columns(self) -> Self:
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate column names in table {self.name!r}")
        if len(self.primary_key) != len(set(self.primary_key)) or not set(self.primary_key) <= set(
            names
        ):
            raise ValueError(f"invalid primary-key columns in table {self.name!r}")
        return self


class ForeignKey(Frozen):
    """A foreign-key edge: (from_table.from_column) references (to_table.to_column)."""

    from_table: str
    from_column: str
    to_table: str
    to_column: str


class DatabaseSchema(Frozen):
    """A database's tables (in declared order) and its foreign-key edges."""

    db_id: str
    tables: tuple[Table, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()

    @model_validator(mode="after")
    def valid_references(self) -> Self:
        columns = {table.name: {column.name for column in table.columns} for table in self.tables}
        if len(columns) != len(self.tables):
            raise ValueError(f"duplicate table names in database {self.db_id!r}")
        for edge in self.foreign_keys:
            if edge.from_column not in columns.get(
                edge.from_table, set()
            ) or edge.to_column not in columns.get(edge.to_table, set()):
                raise ValueError(f"invalid foreign-key reference in database {self.db_id!r}")
        return self


class Example(Frozen):
    """One NL→SQL example. `db_id` resolves to a schema + a `.sqlite` file at load time.

    `evidence` is BIRD's external-knowledge hint (None for Spider); when present it is included in
    the prompt.
    """

    db_id: str
    question: str
    gold_sql: str
    split: str
    dataset: str = "spider"
    evidence: str | None = None


EXAMPLE_IDENTITY_POLICY = "sql-example-v1"


def example_identity(example: Example) -> str:
    """Hash the versioned semantic fields used to identify SQL training populations."""
    payload = {
        "policy": EXAMPLE_IDENTITY_POLICY,
        "dataset": example.dataset,
        "split": example.split,
        "db_id": example.db_id,
        "question": example.question,
        "gold_sql": example.gold_sql,
        "evidence": example.evidence,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
