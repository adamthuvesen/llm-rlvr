"""baseline-evaluation: prompt construction + deterministic few-shot selection."""

from __future__ import annotations

from text2sql_rlvr.data.models import Column, DatabaseSchema, Example, Table
from text2sql_rlvr.eval.prompt import INSTRUCTION, build_messages, select_exemplars


def _schema(db_id: str) -> DatabaseSchema:
    return DatabaseSchema(
        db_id=db_id,
        tables=(Table(name="t", columns=(Column(name="a", type="int"),)),),
    )


SCHEMAS = {f"db{i}": _schema(f"db{i}") for i in range(6)}
TRAIN = tuple(
    Example(db_id=f"db{i}", question=f"q{i}", gold_sql=f"SELECT {i}", split="train")
    for i in range(6)
)


def test_messages_structure_with_exemplars() -> None:
    exemplars = select_exemplars(TRAIN, SCHEMAS, k=2, seed=0)
    messages = build_messages("SCHEMA-TEXT", "How many?", exemplars)
    assert messages[0] == {"role": "system", "content": INSTRUCTION}
    assert messages[-1]["role"] == "user"
    assert "SCHEMA-TEXT" in messages[-1]["content"]
    assert "How many?" in messages[-1]["content"]
    # system + (user, assistant) * k + final user
    assert len(messages) == 1 + 2 * 2 + 1


def test_exemplar_selection_is_deterministic() -> None:
    assert select_exemplars(TRAIN, SCHEMAS, k=3, seed=7) == select_exemplars(
        TRAIN, SCHEMAS, k=3, seed=7
    )


def test_exemplars_render_schema_and_come_from_pool() -> None:
    exemplars = select_exemplars(TRAIN, SCHEMAS, k=3, seed=1)
    assert len(exemplars) == 3
    pool_questions = {e.question for e in TRAIN}
    for exemplar in exemplars:
        assert exemplar.question in pool_questions
        assert 'CREATE TABLE "t"' in exemplar.schema_text


def test_zero_shot_has_no_exemplars() -> None:
    assert select_exemplars(TRAIN, SCHEMAS, k=0, seed=0) == ()
    assert len(build_messages("S", "Q")) == 2  # system + user


def test_exemplars_carry_db_id() -> None:
    exemplars = select_exemplars(TRAIN, SCHEMAS, k=2, seed=0)
    assert all(e.db_id in SCHEMAS for e in exemplars)


def test_select_exemplars_rejects_non_train_pool() -> None:
    import pytest

    dev_pool = (Example(db_id="db0", question="q", gold_sql="SELECT 0", split="dev"),)
    with pytest.raises(ValueError, match="train-only"):
        select_exemplars(dev_pool, SCHEMAS, k=1, seed=0)
