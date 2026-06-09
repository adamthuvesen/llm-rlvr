"""spider-dataset: tables.json parsing, checksum gate, DB resolution (no network)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from text2sql_rlvr.data import ForeignKey, resolve_database, serialize_schema, verify_sha256
from text2sql_rlvr.data.spider import load_schemas, parse_schema

TABLES_ENTRY = {
    "db_id": "concert",
    "table_names_original": ["stadium", "singer"],
    "column_names_original": [
        [-1, "*"],
        [0, "Stadium_ID"],
        [0, "Name"],
        [1, "Singer_ID"],
        [1, "Stadium_ID"],
    ],
    "column_types": ["text", "number", "text", "number", "number"],
    "primary_keys": [1, 3],
    "foreign_keys": [[4, 1]],
}


def test_parse_schema_tables_columns_pk_fk() -> None:
    schema = parse_schema(TABLES_ENTRY)
    assert [t.name for t in schema.tables] == ["stadium", "singer"]
    stadium, singer = schema.tables
    assert [(c.name, c.type) for c in stadium.columns] == [
        ("Stadium_ID", "number"),
        ("Name", "text"),
    ]
    assert stadium.primary_key == ("Stadium_ID",)
    assert singer.primary_key == ("Singer_ID",)
    assert schema.foreign_keys == (
        ForeignKey(
            from_table="singer",
            from_column="Stadium_ID",
            to_table="stadium",
            to_column="Stadium_ID",
        ),
    )


def test_parse_schema_round_trips_through_serialization() -> None:
    text = serialize_schema(parse_schema(TABLES_ENTRY))
    assert 'CREATE TABLE "stadium"' in text
    assert 'FOREIGN KEY ("Stadium_ID") REFERENCES "stadium"("Stadium_ID")' in text


def test_load_schemas_from_file(tmp_path: Path) -> None:
    tables = tmp_path / "tables.json"
    tables.write_text(json.dumps([TABLES_ENTRY]), encoding="utf-8")
    schemas = load_schemas(tables)
    assert set(schemas) == {"concert"}


def test_verify_sha256_roundtrip_and_mismatch(tmp_path: Path) -> None:
    blob = tmp_path / "archive.bin"
    blob.write_bytes(b"spider-databases")
    import hashlib

    good = hashlib.sha256(b"spider-databases").hexdigest()
    verify_sha256(blob, good)  # no raise
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_sha256(blob, "0" * 64)


def test_resolve_database_present_and_missing(tmp_path: Path) -> None:
    db_root = tmp_path / "database"
    db_dir = db_root / "concert"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "concert.sqlite"
    sqlite3.connect(db_path).close()

    assert resolve_database("concert", db_root) == db_path
    with pytest.raises(FileNotFoundError, match=r"not found"):
        resolve_database("nonexistent", db_root)
