"""Revision probe tests: builders, gating logic, reference-free prompts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from llm_rlvr.data.models import Column, DatabaseSchema, Example, Table
from llm_rlvr.eval.completions import CompletionSet, GenerationSample, ProblemCompletions
from llm_rlvr.eval.provenance import DecodingConfig, GenerationProvenance
from llm_rlvr.revision.probe import (
    build_revision_prompts,
    read_prompt_rows,
    score_revision_probe,
    write_prompt_rows,
)

SCHEMA = DatabaseSchema(
    db_id="shop",
    tables=(
        Table(
            name="orders",
            columns=(Column(name="order_id", type="int"), Column(name="amount", type="int")),
            primary_key=("order_id",),
        ),
    ),
)


@pytest.fixture()
def db_root(tmp_path: Path) -> Path:
    root = tmp_path / "database"
    (root / "shop").mkdir(parents=True)
    with sqlite3.connect(root / "shop" / "shop.sqlite") as conn:
        conn.execute("CREATE TABLE orders (order_id int PRIMARY KEY, amount int)")
        conn.executemany("INSERT INTO orders VALUES (?, ?)", [(1, 10), (2, 25)])
    return root


def _completion_set(samples: list[tuple[str, str, str]]) -> CompletionSet:
    """(question, gold, predicted) triples → a minimal CompletionSet."""
    return CompletionSet(
        provenance=GenerationProvenance(
            model="m",
            backend="vllm",
            dataset="spider",
            dataset_revision="r",
            split="dev",
            n_examples=len(samples),
            decoding=DecodingConfig(),
        ),
        items=tuple(
            ProblemCompletions(
                example=Example(db_id="shop", question=q, gold_sql=gold, split="dev"),
                samples=(GenerationSample(text=f"```sql\n{pred}\n```", sql=pred),),
            )
            for q, gold, pred in samples
        ),
    )


WRONG = ("How many orders?", "SELECT COUNT(*) FROM orders", "SELECT COUNT(*) FROM order")
RIGHT = ("Total amount?", "SELECT SUM(amount) FROM orders", "SELECT SUM(amount) FROM orders")


def test_build_splits_wrong_and_right_with_real_feedback(db_root: Path) -> None:
    rows, n_skipped = build_revision_prompts(
        _completion_set([WRONG, RIGHT]), {"shop": SCHEMA}, db_root
    )
    assert n_skipped == 0 and len(rows) == 2
    wrong_row = next(r for r in rows if not r.was_correct)
    right_row = next(r for r in rows if r.was_correct)
    assert wrong_row.feedback.startswith("Execution feedback: error")
    assert right_row.feedback.startswith("Execution feedback: query executed")
    # The reference query never appears anywhere in the prompt the model sees.
    for row in rows:
        prompt_text = " ".join(str(m["content"]) for m in row.prompt)
        assert row.gold_sql not in prompt_text or row.previous_sql == row.gold_sql
    assert "Your previous query" in str(wrong_row.prompt[1]["content"])


def test_round_trip(db_root: Path, tmp_path: Path) -> None:
    rows, _ = build_revision_prompts(_completion_set([WRONG, RIGHT]), {"shop": SCHEMA}, db_root)
    path = write_prompt_rows(rows, tmp_path / "rows.jsonl")
    assert read_prompt_rows(path) == rows


def test_scoring_judges_pre_registered_gates(db_root: Path, tmp_path: Path) -> None:
    rows, n_skipped = build_revision_prompts(
        _completion_set([WRONG, RIGHT]), {"shop": SCHEMA}, db_root
    )
    rows = tuple(sorted(rows, key=lambda r: r.was_correct))  # wrong first, right second
    src = tmp_path / "src.json"
    src.write_text("{}", encoding="utf-8")
    # Revision fixes the wrong one and keeps the right one → fix 1/1, harm 0/1 → PASS.
    report = score_revision_probe(
        rows,
        ["```sql\nSELECT COUNT(*) FROM orders\n```", "```sql\nSELECT SUM(amount) FROM orders\n```"],
        db_root,
        source_completions=src,
        n_source=2,
        n_right_total=1,
        n_skipped=n_skipped,
        commit="abc",
    )
    assert (report.n_fixed, report.n_harmed) == (1, 0)
    assert report.fix_rate == 1.0 and report.harm_rate == 0.0
    assert report.n_fixed_from_error_feedback == 1
    assert report.passed and report.g1_passed and report.g2_passed
    # Revision breaks the right one and leaves the wrong one wrong → FAIL both directions.
    report2 = score_revision_probe(
        rows,
        ["```sql\nSELECT COUNT(*) FROM order\n```", "```sql\nSELECT 999\n```"],
        db_root,
        source_completions=src,
        n_source=2,
        n_right_total=1,
        n_skipped=n_skipped,
        commit="abc",
    )
    assert (report2.n_fixed, report2.n_harmed) == (0, 1)
    assert not report2.passed
    assert report2.net_ex_delta_estimate < 0


def test_unparseable_sql_is_skipped(db_root: Path) -> None:
    cs = _completion_set([WRONG])
    no_sql = ProblemCompletions(
        example=Example(db_id="shop", question="q", gold_sql="SELECT 1", split="dev"),
        samples=(GenerationSample(text="no sql here", sql=None),),
    )
    cs = cs.model_copy(update={"items": (*cs.items, no_sql)})
    rows, n_skipped = build_revision_prompts(cs, {"shop": SCHEMA}, db_root)
    assert len(rows) == 1 and n_skipped == 1


def test_right_sample_is_seeded_and_capped(db_root: Path) -> None:
    samples = [
        (f"Total {i}?", "SELECT SUM(amount) FROM orders", "SELECT SUM(amount) FROM orders")
        for i in range(8)
    ]
    cs = _completion_set(samples)
    a, _ = build_revision_prompts(cs, {"shop": SCHEMA}, db_root, right_sample_size=3, seed=1)
    b, _ = build_revision_prompts(cs, {"shop": SCHEMA}, db_root, right_sample_size=3, seed=1)
    assert a == b and sum(1 for r in a if r.was_correct) == 3
