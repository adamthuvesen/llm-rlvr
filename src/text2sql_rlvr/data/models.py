"""Strict, typed records for examples and database schemas.

Loader-agnostic: a Spider/BIRD loader parses its own on-disk format into these models, so
everything downstream (prompt building, serialization, scoring) sees one shape.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Column(_Frozen):
    """A single column: its name and SQL-ish type as the dataset declares it."""

    name: str
    type: str


class Table(_Frozen):
    """A table, its columns (in declared order), and its primary-key column names."""

    name: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...] = ()


class ForeignKey(_Frozen):
    """A foreign-key edge: (from_table.from_column) references (to_table.to_column)."""

    from_table: str
    from_column: str
    to_table: str
    to_column: str


class DatabaseSchema(_Frozen):
    """A database's tables (in declared order) and its foreign-key edges."""

    db_id: str
    tables: tuple[Table, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()


class Example(_Frozen):
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
