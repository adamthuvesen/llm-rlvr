"""bird-dataset: example parsing (with evidence) and bundle layout resolution (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_rlvr_sql.data.bird import _bird_dev_at, load_bird_examples

BIRD_ROWS = [
    {
        "question_id": 0,
        "db_id": "california_schools",
        "question": "Q1",
        "evidence": "hint A",
        "SQL": "SELECT 1",
    },
    {"question_id": 1, "db_id": "card_games", "question": "Q2", "evidence": "", "SQL": "SELECT 2"},
]


def test_load_bird_examples_preserves_evidence(tmp_path: Path) -> None:
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps(BIRD_ROWS), encoding="utf-8")
    examples = load_bird_examples(dev)
    assert examples[0].dataset == "bird"
    assert examples[0].gold_sql == "SELECT 1"
    assert examples[0].evidence == "hint A"
    assert examples[1].evidence is None  # empty evidence normalizes to None


def test_bird_dev_layout_resolves(tmp_path: Path) -> None:
    root = tmp_path / "dev_20240627"
    (root / "dev_databases" / "california_schools").mkdir(parents=True)
    (root / "dev.json").write_text("[]", encoding="utf-8")
    (root / "dev_tables.json").write_text("[]", encoding="utf-8")
    bird = _bird_dev_at(root)
    assert bird.database_root == root / "dev_databases"
    assert bird.dev_json == root / "dev.json"


def test_bird_dev_missing_layout_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="BIRD dev layout"):
        _bird_dev_at(tmp_path)
