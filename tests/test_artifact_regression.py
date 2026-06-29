"""Regression guard: re-scoring committed CompletionSets must match committed ScoreReports."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from llm_rlvr_sql.eval.completions import read_completion_set
from llm_rlvr_sql.eval.score import score_completion_set

REPO = Path(__file__).resolve().parents[1]
N50_COMPLETION = REPO / "results/baseline/spider-dev-fs3-n50.json"
N50_SCORE = REPO / "results/baseline/spider-dev-fs3-n50.score.json"


def _spider_database_root() -> Path | None:
    env_root = os.environ.get("TEXT2SQL_SPIDER_ROOT")
    if env_root:
        db = Path(env_root) / "database"
        if db.is_dir():
            return db
    default = REPO / "data/spider/spider_data/database"
    if default.is_dir():
        return default
    return None


@pytest.fixture
def spider_db_root() -> Path:
    root = _spider_database_root()
    if root is None:
        pytest.skip("Spider databases not present (run make data or set TEXT2SQL_SPIDER_ROOT)")
    return root


def test_committed_n50_slice_rescores_to_committed_score_report(spider_db_root: Path) -> None:
    committed = json.loads(N50_SCORE.read_text(encoding="utf-8"))
    completion_set = read_completion_set(N50_COMPLETION)
    report = score_completion_set(completion_set, spider_db_root)

    assert report.execution_accuracy == committed["execution_accuracy"]
    assert report.n_correct == committed["n_correct"]
    assert report.n == committed["n"]
    assert report.n_gold_errors == committed["n_gold_errors"]


def test_committed_score_report_has_ex_fields() -> None:
    data = json.loads(N50_SCORE.read_text(encoding="utf-8"))
    for key in ("execution_accuracy", "n_correct", "n", "n_gold_errors"):
        assert key in data
