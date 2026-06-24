"""Strict, typed records for examples and database schemas.

Loader-agnostic: a Spider/BIRD loader parses its own on-disk format into these models, so
everything downstream (prompt building, serialization, scoring) sees one shape.
"""

from __future__ import annotations

from llm_rlvr_sql._base import Frozen


class Column(Frozen):
    """A single column: its name and SQL-ish type as the dataset declares it."""

    name: str
    type: str


class Table(Frozen):
    """A table, its columns (in declared order), and its primary-key column names."""

    name: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...] = ()


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
