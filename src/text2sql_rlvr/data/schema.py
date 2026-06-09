"""Deterministic schema → prompt-text serialization.

A pure function of the schema: the same `DatabaseSchema` always serializes to byte-identical
text. Kept swappable (one function, fixed contract) so the serialization style can be A/B'd
later without touching callers. Style: a compact `CREATE TABLE` block per table — the format
SQL-R1 / OmniSQL-class models expect.
"""

from __future__ import annotations

from text2sql_rlvr.data.models import DatabaseSchema


def serialize_schema(schema: DatabaseSchema) -> str:
    """Render a `CREATE TABLE`-style block per table, with PK and FK constraints.

    Tables and columns appear in declared order; foreign keys are grouped under their
    originating table. Output is deterministic.
    """
    fks_by_table: dict[str, list[str]] = {}
    for fk in schema.foreign_keys:
        line = f'  FOREIGN KEY ("{fk.from_column}") REFERENCES "{fk.to_table}"("{fk.to_column}")'
        fks_by_table.setdefault(fk.from_table, []).append(line)

    blocks: list[str] = []
    for table in schema.tables:
        lines = [f'CREATE TABLE "{table.name}" (']
        body = [f'  "{col.name}" {col.type}' for col in table.columns]
        if table.primary_key:
            pk = ", ".join(f'"{name}"' for name in table.primary_key)
            body.append(f"  PRIMARY KEY ({pk})")
        body.extend(fks_by_table.get(table.name, []))
        lines.append(",\n".join(body))
        lines.append(");")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)
