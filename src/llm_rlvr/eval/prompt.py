"""Prompt construction: schema + question → chat messages, with a fixed few-shot policy.

Few-shot policy (design R2): k=3 seeded random exemplars drawn from the *train* split only
(cross-schema, no dev/test leakage), recorded in provenance. No retriever — the baseline is a
clean, reproducible reference, not an optimized pipeline.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from llm_rlvr._base import Frozen
from llm_rlvr.data.models import DatabaseSchema, Example
from llm_rlvr.data.schema import serialize_schema
from llm_rlvr.data.splits import assert_train_split
from llm_rlvr.eval.messages import user_content
from llm_rlvr.eval.prompts import multiturn as _mt

MULTITURN_INSTRUCTION = _mt.MULTITURN_INSTRUCTION
tool_demo_messages = _mt.tool_demo_messages
DEMO_ERROR_FEEDBACK = _mt.DEMO_ERROR_FEEDBACK
DEMO_FIXED_SQL = _mt.DEMO_FIXED_SQL
DEMO_SCHEMA_TEXT = _mt.DEMO_SCHEMA_TEXT
DEMO_SUCCESS_FEEDBACK = _mt.DEMO_SUCCESS_FEEDBACK
DEMO_WRONG_SQL = _mt.DEMO_WRONG_SQL

INSTRUCTION = (
    "You are an expert data analyst. Given a SQLite database schema and a question, write a "
    "single SQLite query that answers it. Reason step by step inside <think></think>, then give "
    "the final query in a ```sql ... ``` code block."
)
# A direct prompt (no chain-of-thought) — the standard base-model text-to-SQL setup, for a fair,
# literature-comparable base number. The `<think>` style above is matched to the RL model.
DIRECT_INSTRUCTION = (
    "You are an expert data analyst. Given a SQLite database schema and a question, write a "
    "single SQLite query that answers it. Output only the query in a ```sql ... ``` code block, "
    "with no explanation."
)
# Multi-turn styles — see docs/status.md for current scope.
INSTRUCTIONS = {
    "cot": INSTRUCTION,
    "direct": DIRECT_INSTRUCTION,
    "mt": MULTITURN_INSTRUCTION,
    "mt_demo": MULTITURN_INSTRUCTION,
}


class FewShotExemplar(Frozen):
    """A rendered few-shot demonstration: db_id, schema text, question, gold SQL, and evidence."""

    db_id: str
    schema_text: str
    question: str
    gold_sql: str
    evidence: str | None = None


def build_messages(
    schema_text: str,
    question: str,
    exemplars: Sequence[FewShotExemplar] = (),
    evidence: str | None = None,
    style: str = "cot",
) -> list[dict[str, object]]:
    """Chat messages: system, the tool demonstration (`mt_demo` only), few-shot user/assistant
    pairs, then the target question."""
    if style not in INSTRUCTIONS:
        raise ValueError(f"unknown prompt style {style!r}; expected one of {sorted(INSTRUCTIONS)}")
    messages: list[dict[str, object]] = [{"role": "system", "content": INSTRUCTIONS[style]}]
    if style == "mt_demo":
        messages.extend(tool_demo_messages())
    for exemplar in exemplars:
        messages.append(
            {
                "role": "user",
                "content": user_content(exemplar.schema_text, exemplar.question, exemplar.evidence),
            }
        )
        messages.append({"role": "assistant", "content": f"```sql\n{exemplar.gold_sql}\n```"})
    messages.append({"role": "user", "content": user_content(schema_text, question, evidence)})
    return messages


def select_exemplars(
    train_pool: Sequence[Example],
    schemas: Mapping[str, DatabaseSchema],
    *,
    k: int,
    seed: int,
) -> tuple[FewShotExemplar, ...]:
    """Deterministically pick k cross-schema exemplars from the train pool (seeded; no leakage).

    Enforces the R2 invariant: every pooled example MUST be from the train split, so dev/test
    rows can never leak into a prompt.
    """
    if k <= 0 or not train_pool:
        return ()
    assert_train_split(train_pool, context="few-shot pool")
    rng = random.Random(seed)
    chosen = rng.sample(list(train_pool), min(k, len(train_pool)))
    return tuple(
        FewShotExemplar(
            db_id=example.db_id,
            schema_text=serialize_schema(schemas[example.db_id]),
            question=example.question,
            gold_sql=example.gold_sql,
            evidence=example.evidence,
        )
        for example in chosen
    )
