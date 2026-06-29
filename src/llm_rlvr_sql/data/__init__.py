"""Data layer: Spider/BIRD loaders, typed examples, and schema serialization."""

from llm_rlvr_sql.data.bird import BirdDev, ensure_bird_dev, load_bird_examples
from llm_rlvr_sql.data.models import Column, DatabaseSchema, Example, ForeignKey, Table
from llm_rlvr_sql.data.paths import resolve_database, sha256_of, verify_sha256
from llm_rlvr_sql.data.schema import serialize_schema
from llm_rlvr_sql.data.spider import (
    SpiderDatabases,
    ensure_spider_databases,
    load_examples,
    load_schemas,
    parse_schema,
)

__all__ = [
    "BirdDev",
    "Column",
    "DatabaseSchema",
    "Example",
    "ForeignKey",
    "SpiderDatabases",
    "Table",
    "ensure_bird_dev",
    "ensure_spider_databases",
    "load_bird_examples",
    "load_examples",
    "load_schemas",
    "parse_schema",
    "resolve_database",
    "serialize_schema",
    "sha256_of",
    "verify_sha256",
]
