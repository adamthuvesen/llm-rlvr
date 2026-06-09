"""Data layer: Spider/BIRD loaders, typed examples, and schema serialization."""

from text2sql_rlvr.data.bird import BirdDev, ensure_bird_dev, load_bird_examples
from text2sql_rlvr.data.models import Column, DatabaseSchema, Example, ForeignKey, Table
from text2sql_rlvr.data.schema import serialize_schema
from text2sql_rlvr.data.spider import (
    SpiderDatabases,
    ensure_spider_databases,
    load_examples,
    load_schemas,
    parse_schema,
    resolve_database,
    verify_sha256,
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
    "verify_sha256",
]
