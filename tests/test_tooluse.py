"""Tool-use SFT dataset (Stage-3 rung 2): corruption, transcript shapes, real-feedback fidelity."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from text2sql_rlvr.data.models import Column, DatabaseSchema, Example, Table
from text2sql_rlvr.env.text2sql_env import FeedbackConfig, SqliteText2SqlEnv
from text2sql_rlvr.eval.prompt import MULTITURN_INSTRUCTION
from text2sql_rlvr.train.tooluse import (
    ToolUseMix,
    build_tooluse_manifest,
    build_tooluse_rows,
    corruption_candidates,
    write_tooluse_dataset,
)
from text2sql_rlvr.train.traces import SftTrace

SCHEMA = DatabaseSchema(
    db_id="shop",
    tables=(
        Table(
            name="orders",
            columns=(
                Column(name="order_id", type="int"),
                Column(name="amount", type="int"),
            ),
            primary_key=("order_id",),
        ),
        Table(
            name="customers",
            columns=(
                Column(name="customer_id", type="int"),
                Column(name="region", type="text"),
            ),
            primary_key=("customer_id",),
        ),
    ),
)


@pytest.fixture()
def db_root(tmp_path: Path) -> Path:
    root = tmp_path / "database"
    db_dir = root / "shop"
    db_dir.mkdir(parents=True)
    with sqlite3.connect(db_dir / "shop.sqlite") as conn:
        conn.execute("CREATE TABLE orders (order_id int PRIMARY KEY, amount int)")
        conn.execute("CREATE TABLE customers (customer_id int PRIMARY KEY, region text)")
        conn.executemany("INSERT INTO orders VALUES (?, ?)", [(1, 10), (2, 25)])
        conn.execute("INSERT INTO customers VALUES (1, 'north')")
    return root


def _example(question: str = "How many orders?") -> Example:
    return Example(
        db_id="shop", question=question, gold_sql="SELECT COUNT(*) FROM orders", split="train"
    )


TRACES = (
    SftTrace(
        db_id="shop",
        question="Total amount?",
        prompt=(
            {"role": "system", "content": "old cot instruction"},
            {"role": "user", "content": "Database schema:\nS\n\nQuestion: Total amount?"},
        ),
        completion=(
            "<think>Sum the amount column.</think>\n```sql\nSELECT SUM(amount) FROM orders\n```"
        ),
    ),
)


def test_corruption_candidates_prefer_cross_table_columns() -> None:
    first = next(corruption_candidates("SELECT amount FROM orders", SCHEMA))
    # `amount` is replaced by a column from the unreferenced `customers` table.
    assert "customer_id" in first and "amount" not in first


def test_revise_transcript_rows_carry_real_feedback(db_root: Path) -> None:
    rows, counts = build_tooluse_rows(
        [_example()],
        {"shop": SCHEMA},
        db_root,
        (),
        mix=ToolUseMix(n_revise=1, n_check=0, n_direct=0),
    )
    assert counts.n_revise == 1 and counts.n_rows == 3
    first, second, final = rows
    # Row 1: bare prompt → corrupted tool call.
    assert [m["role"] for m in first.prompt] == ["system", "user"]
    assert first.prompt[0]["content"] == MULTITURN_INSTRUCTION
    call = first.completion[0]["tool_calls"][0]["function"]
    assert call["name"] == "run_sql"
    corrupted = call["arguments"]["query"]
    assert corrupted != _example().gold_sql
    # Row 2: the error feedback the model saw is what the env would really say for that query.
    env = SqliteText2SqlEnv(db_root)
    env.reset(db_id="shop")
    assert second.prompt[3]["content"] == env.run_sql(corrupted)
    assert second.prompt[3]["content"].startswith("Execution feedback: error")
    # Row 2 completion revises to gold; row 3 ends with the gold answer in the mt final format.
    assert second.completion[0]["tool_calls"][0]["function"]["arguments"]["query"] == (
        _example().gold_sql
    )
    assert "<think>" in final.completion[0]["content"]
    assert f"```sql\n{_example().gold_sql}\n```" in final.completion[0]["content"]
    # Success feedback in row 3's prompt matches real execution too.
    env.reset(db_id="shop")
    assert final.prompt[5]["content"] == env.run_sql(_example().gold_sql)


def test_check_and_direct_shapes(db_root: Path) -> None:
    rows, counts = build_tooluse_rows(
        [_example()],
        {"shop": SCHEMA},
        db_root,
        TRACES,
        mix=ToolUseMix(n_revise=0, n_check=1, n_direct=1),
    )
    assert (counts.n_check, counts.n_direct, counts.n_rows) == (1, 1, 3)
    check_rows = [r for r in rows if r.shape == "check"]
    assert len(check_rows) == 2
    assert check_rows[0].completion[0]["tool_calls"][0]["function"]["arguments"]["query"] == (
        _example().gold_sql
    )
    # Direct rows re-home the real Stage-1 trace under the mt instruction, keeping its completion.
    (direct,) = [r for r in rows if r.shape == "direct"]
    assert direct.prompt[0] == {"role": "system", "content": MULTITURN_INSTRUCTION}
    assert direct.completion[0]["content"] == TRACES[0].completion
    assert "tool_calls" not in direct.completion[0]


def test_gold_error_questions_are_skipped(db_root: Path) -> None:
    broken = Example(
        db_id="shop", question="Broken?", gold_sql="SELECT nope FROM orders", split="train"
    )
    rows, counts = build_tooluse_rows(
        [broken], {"shop": SCHEMA}, db_root, (), mix=ToolUseMix(n_revise=1, n_check=1, n_direct=0)
    )
    assert rows == () and counts.n_gold_errors == 1


def test_non_train_split_raises(db_root: Path) -> None:
    dev = Example(db_id="shop", question="q", gold_sql="SELECT 1", split="dev")
    with pytest.raises(ValueError, match="train-split only"):
        build_tooluse_rows([dev], {"shop": SCHEMA}, db_root, ())


def test_dataset_jsonl_round_trip_and_manifest(db_root: Path, tmp_path: Path) -> None:
    rows, counts = build_tooluse_rows(
        [_example()],
        {"shop": SCHEMA},
        db_root,
        TRACES,
        mix=ToolUseMix(n_revise=1, n_check=0, n_direct=1),
    )
    out = write_tooluse_dataset(rows, tmp_path / "tooluse.jsonl")
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == counts.n_rows
    assert all(
        set(line) == {"prompt", "completion", "db_id", "question", "shape"} for line in lines
    )
    # The serialized completion is a single assistant message (TRL prompt-completion contract).
    assert all(
        len(line["completion"]) == 1 and line["completion"][0]["role"] == "assistant"
        for line in lines
    )
    traces_file = tmp_path / "traces.jsonl"
    traces_file.write_text("{}", encoding="utf-8")
    manifest = build_tooluse_manifest(
        dataset="spider",
        dataset_revision="rev",
        mix=ToolUseMix(n_revise=1, n_check=0, n_direct=1),
        feedback=FeedbackConfig(),
        trace_dataset_path=traces_file,
        counts=counts,
        commit="abc",
        dirty=False,
    )
    assert manifest.counts.n_rows == counts.n_rows and len(manifest.trace_dataset_sha256) == 64


def test_direct_rows_strip_preambles_and_skip_thinkless(db_root: Path) -> None:
    preambled = SftTrace(
        db_id="shop",
        question="Total?",
        prompt=TRACES[0].prompt,
        completion="To answer this, we sum.\n\n<think>Sum it.</think>\n```sql\nSELECT 1\n```",
    )
    thinkless = SftTrace(
        db_id="shop",
        question="No think?",
        prompt=TRACES[0].prompt,
        completion="```sql\nSELECT 2\n```",
    )
    rows, counts = build_tooluse_rows(
        [],
        {"shop": SCHEMA},
        db_root,
        (preambled, thinkless),
        mix=ToolUseMix(n_revise=0, n_check=0, n_direct=2),
    )
    assert counts.n_direct == 1 and counts.n_direct_skipped == 1
    (direct,) = rows
    # The preamble is gone; the completion now LEADS with <think> (the Stage-2 format).
    assert direct.completion[0]["content"].startswith("<think>Sum it.</think>")


def test_no_completion_ever_quotes_the_feedback_marker(db_root: Path) -> None:
    """Attempt-4 regression: a think template quoting 'Execution feedback' taught the policy to
    generate it. No training target may contain the marker."""
    rows, _ = build_tooluse_rows(
        [_example()],
        {"shop": SCHEMA},
        db_root,
        TRACES,
        mix=ToolUseMix(n_revise=1, n_check=1, n_direct=1),
    )
    for row in rows:
        assert "Execution feedback" not in str(row.completion[0].get("content", ""))


def test_render_text_row_prefix_split() -> None:
    from text2sql_rlvr.train.tooluse import render_text_row

    def fake_template(messages, tools, add_generation_prompt, tokenize):
        text = f"[tools:{len(tools)}]" + "".join(
            f"<{m['role']}>{m.get('content', '')}" for m in messages
        )
        return text + ("<assistant>" if add_generation_prompt else "\n")

    row = render_text_row(
        [{"role": "user", "content": "q"}],
        [{"role": "assistant", "content": "a"}],
        fake_template,
        [{"name": "run_sql"}],
    )
    assert row == {"prompt": "[tools:1]<user>q<assistant>", "completion": "a"}


def test_render_text_row_raises_when_not_prefix_stable() -> None:
    import pytest

    from text2sql_rlvr.train.tooluse import render_text_row

    def broken_template(messages, tools, add_generation_prompt, tokenize):
        return "X" if add_generation_prompt else "Y"

    with pytest.raises(ValueError, match="prefix-stable"):
        render_text_row([{"role": "user", "content": "q"}], [], broken_template, [])


def test_build_is_deterministic(db_root: Path) -> None:
    mix = ToolUseMix(n_revise=1, n_check=0, n_direct=1, seed=7)
    a = build_tooluse_rows([_example()], {"shop": SCHEMA}, db_root, TRACES, mix=mix)
    b = build_tooluse_rows([_example()], {"shop": SCHEMA}, db_root, TRACES, mix=mix)
    assert a == b
