"""spider-dataset: schema serialization is deterministic and complete."""

from __future__ import annotations

from text2sql_rlvr.data import (
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
