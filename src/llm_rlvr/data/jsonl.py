"""JSONL helpers for dataset artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path


def write_jsonl(records: Iterable[Mapping[str, object]], path: str | Path) -> Path:
    """Write JSONL records with a trailing newline when non-empty."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out
