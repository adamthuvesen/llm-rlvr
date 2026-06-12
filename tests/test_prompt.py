"""baseline-evaluation: prompt construction + deterministic few-shot selection."""

from __future__ import annotations

from text2sql_rlvr.data.models import Column, DatabaseSchema, Example, Table
from text2sql_rlvr.eval.prompt import (
    INSTRUCTION,
    FewShotExemplar,
    build_messages,
    select_exemplars,
)


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


def test_evidence_included_in_prompt_when_present() -> None:
    target = build_messages("SCHEMA", "Q", evidence="charge = 0 means free")
    assert "Evidence: charge = 0 means free" in target[-1]["content"]
    exemplar = FewShotExemplar(
        db_id="d", schema_text="S", question="EQ", gold_sql="G", evidence="exemplar hint"
    )
    with_exemplar = build_messages("SCHEMA", "Q", (exemplar,))
    assert "Evidence: exemplar hint" in with_exemplar[1]["content"]


def test_no_evidence_leaves_prompt_unchanged() -> None:
    # Spider (no evidence) must be byte-identical to the pre-BIRD behavior.
    content = build_messages("SCHEMA-TEXT", "How many?")[-1]["content"]
    assert content == "Database schema:\nSCHEMA-TEXT\n\nQuestion: How many?"
    assert "Evidence:" not in content


def test_prompt_style_selects_instruction() -> None:
    from text2sql_rlvr.eval.prompt import DIRECT_INSTRUCTION

    assert build_messages("S", "Q", style="cot")[0]["content"] == INSTRUCTION
    assert build_messages("S", "Q", style="direct")[0]["content"] == DIRECT_INSTRUCTION


def test_unknown_prompt_style_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown prompt style"):
        build_messages("S", "Q", style="nope")


# --- mt_demo: the in-context tool-protocol demonstration (Stage-3 smoke rung 1) ---


def test_mt_demo_inserts_demo_between_system_and_target() -> None:
    from text2sql_rlvr.eval.prompt import MULTITURN_INSTRUCTION, tool_demo_messages

    messages = build_messages("SCHEMA-TEXT", "How many?", style="mt_demo")
    assert messages[0] == {"role": "system", "content": MULTITURN_INSTRUCTION}
    assert messages[1:-1] == tool_demo_messages()
    assert messages[-1]["role"] == "user"
    assert "SCHEMA-TEXT" in messages[-1]["content"]
    # The other styles are byte-identical to before — no demo anywhere.
    for style in ("cot", "direct", "mt"):
        assert len(build_messages("S", "Q", style=style)) == 2


def test_mt_demo_shows_two_tool_turns_then_final_answer() -> None:
    from text2sql_rlvr.eval.prompt import (
        DEMO_ERROR_FEEDBACK,
        DEMO_FIXED_SQL,
        DEMO_SUCCESS_FEEDBACK,
        DEMO_WRONG_SQL,
        tool_demo_messages,
    )

    demo = tool_demo_messages()
    roles = [m["role"] for m in demo]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    calls = [m["tool_calls"][0]["function"] for m in demo if "tool_calls" in m]
    assert [c["arguments"]["query"] for c in calls] == [DEMO_WRONG_SQL, DEMO_FIXED_SQL]
    assert [m["content"] for m in demo if m["role"] == "tool"] == [
        DEMO_ERROR_FEEDBACK,
        DEMO_SUCCESS_FEEDBACK,
    ]
    # The final message is an answer (not a tool call), in the mt final-answer format, and its
    # SQL is the query the demonstration just checked.
    final = demo[-1]
    assert "tool_calls" not in final
    assert "<think>" in final["content"] and f"```sql\n{DEMO_FIXED_SQL}\n```" in final["content"]


def test_mt_demo_tool_calls_match_the_env_tool() -> None:
    import inspect

    from text2sql_rlvr.env.text2sql_env import SqliteText2SqlEnv
    from text2sql_rlvr.eval.prompt import tool_demo_messages

    params = set(inspect.signature(SqliteText2SqlEnv.run_sql).parameters) - {"self"}
    for message in tool_demo_messages():
        for call in message.get("tool_calls", []):
            assert call["function"]["name"] == SqliteText2SqlEnv.run_sql.__name__
            assert set(call["function"]["arguments"]) == params


def test_mt_demo_feedback_byte_matches_real_execution(tmp_path) -> None:
    """The demonstrated feedback must be what the env would actually say — execute the demo
    queries in the real sandbox against the demo schema and compare byte-for-byte."""
    import sqlite3

    from text2sql_rlvr.env.sandbox import SqliteSandbox
    from text2sql_rlvr.env.text2sql_env import FeedbackConfig, format_feedback
    from text2sql_rlvr.eval.prompt import (
        DEMO_ERROR_FEEDBACK,
        DEMO_FIXED_SQL,
        DEMO_SUCCESS_FEEDBACK,
        DEMO_WRONG_SQL,
    )

    db = tmp_path / "office_plants.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE plants (id int PRIMARY KEY, species text, room text)")
        conn.executemany(
            "INSERT INTO plants VALUES (?, ?, ?)",
            [
                (1, "monstera", "Atrium"),
                (2, "fern", "Atrium"),
                (3, "ficus", "Atrium"),
                (4, "cactus", "Lobby"),
            ],
        )
    sandbox = SqliteSandbox(db)
    config = FeedbackConfig()
    assert format_feedback(sandbox.execute(DEMO_WRONG_SQL), config) == DEMO_ERROR_FEEDBACK
    assert format_feedback(sandbox.execute(DEMO_FIXED_SQL), config) == DEMO_SUCCESS_FEEDBACK


def test_mt_demo_is_gold_free_and_synthetic() -> None:
    """The demo must never leak dataset content: synthetic db, and no verdict/gold tokens
    (the same invariant the env's feedback tests pin)."""
    from text2sql_rlvr.eval.prompt import DEMO_SCHEMA_TEXT, tool_demo_messages

    assert 'CREATE TABLE "plants"' in DEMO_SCHEMA_TEXT
    text = " ".join(str(m.get("content", "")) for m in tool_demo_messages()).lower()
    assert "gold" not in text and "correct answer" not in text and "verdict" not in text
