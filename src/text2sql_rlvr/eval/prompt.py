"""Prompt construction: schema + question → chat messages, with a fixed few-shot policy.

Few-shot policy (design R2): k=3 seeded random exemplars drawn from the *train* split only
(cross-schema, no dev/test leakage), recorded in provenance. No retriever — the baseline is a
clean, reproducible reference, not an optimized pipeline.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from text2sql_rlvr._base import Frozen
from text2sql_rlvr.data.models import DatabaseSchema, Example
from text2sql_rlvr.data.schema import serialize_schema

INSTRUCTION = (
    "You are an expert data analyst. Given a SQLite database schema and a question, write a "
    "single SQLite query that answers it. Reason step by step inside <think></think>, then give "
    "the final query in a ```sql ... ``` code block."
)


class FewShotExemplar(Frozen):
    """A rendered few-shot demonstration: source db_id, schema text, question, and gold SQL."""

    db_id: str
    schema_text: str
    question: str
    gold_sql: str


def _user_content(schema_text: str, question: str) -> str:
    return f"Database schema:\n{schema_text}\n\nQuestion: {question}"


def build_messages(
    schema_text: str,
    question: str,
    exemplars: Sequence[FewShotExemplar] = (),
) -> list[dict[str, str]]:
    """Chat messages: system, few-shot user/assistant pairs, then the target question."""
    messages = [{"role": "system", "content": INSTRUCTION}]
    for exemplar in exemplars:
        messages.append(
            {"role": "user", "content": _user_content(exemplar.schema_text, exemplar.question)}
        )
        messages.append({"role": "assistant", "content": f"```sql\n{exemplar.gold_sql}\n```"})
    messages.append({"role": "user", "content": _user_content(schema_text, question)})
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
    leaked = [example.split for example in train_pool if example.split != "train"]
    if leaked:
        raise ValueError(
            f"few-shot pool must be train-only (R2 no-leakage); got splits {sorted(set(leaked))}"
        )
    rng = random.Random(seed)
    chosen = rng.sample(list(train_pool), min(k, len(train_pool)))
    return tuple(
        FewShotExemplar(
            db_id=example.db_id,
            schema_text=serialize_schema(schemas[example.db_id]),
            question=example.question,
            gold_sql=example.gold_sql,
        )
        for example in chosen
    )
