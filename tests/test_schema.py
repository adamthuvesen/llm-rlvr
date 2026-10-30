"""spider-dataset: schema serialization is deterministic and complete."""

from __future__ import annotations

import pytest

from llm_rlvr.data import (
    Column,
    DatabaseSchema,
    ForeignKey,
    Table,
    serialize_schema,
)

SCHEMA = DatabaseSchema(
    db_id="concert",
    tables=(
        Table(
            name="stadium",
            columns=(Column(name="Stadium_ID", type="int"), Column(name="Name", type="text")),
            primary_key=("Stadium_ID",),
        ),
        Table(
            name="singer",
            columns=(Column(name="Singer_ID", type="int"), Column(name="Stadium_ID", type="int")),
            primary_key=("Singer_ID",),
        ),
    ),
    foreign_keys=(
        ForeignKey(
            from_table="singer",
            from_column="Stadium_ID",
            to_table="stadium",
            to_column="Stadium_ID",
        ),
    ),
)


def test_serialization_is_deterministic() -> None:
    assert serialize_schema(SCHEMA) == serialize_schema(SCHEMA)


def test_serialization_includes_tables_columns_pk_fk() -> None:
    text = serialize_schema(SCHEMA)
    assert 'CREATE TABLE "stadium"' in text
    assert 'CREATE TABLE "singer"' in text
    assert '"Stadium_ID" int' in text
    assert 'PRIMARY KEY ("Stadium_ID")' in text
    assert 'FOREIGN KEY ("Stadium_ID") REFERENCES "stadium"("Stadium_ID")' in text


def test_tables_appear_in_declared_order() -> None:
    text = serialize_schema(SCHEMA)
    assert text.index('"stadium"') < text.index('"singer"')


def test_schema_rejects_duplicate_table_names() -> None:
    with pytest.raises(ValueError, match="duplicate table names"):
        DatabaseSchema(db_id="db", tables=(SCHEMA.tables[0], SCHEMA.tables[0]))


def test_schema_rejects_duplicate_column_names() -> None:
    column = Column(name="id", type="int")
    with pytest.raises(ValueError, match="duplicate column names"):
        Table(name="t", columns=(column, column))


def test_schema_rejects_missing_primary_key_column() -> None:
    with pytest.raises(ValueError, match="invalid primary-key"):
        Table(name="t", columns=(), primary_key=("missing",))


def test_schema_rejects_missing_foreign_key_target() -> None:
    with pytest.raises(ValueError, match="invalid foreign-key"):
        DatabaseSchema(db_id="db", tables=(SCHEMA.tables[1],), foreign_keys=SCHEMA.foreign_keys)
