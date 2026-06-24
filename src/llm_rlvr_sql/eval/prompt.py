"""Prompt construction: schema + question → chat messages, with a fixed few-shot policy.

Few-shot policy (design R2): k=3 seeded random exemplars drawn from the *train* split only
(cross-schema, no dev/test leakage), recorded in provenance. No retriever — the baseline is a
clean, reproducible reference, not an optimized pipeline.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from llm_rlvr_sql._base import Frozen
from llm_rlvr_sql.data.models import Column, DatabaseSchema, Example, Table
from llm_rlvr_sql.data.schema import serialize_schema

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
# The multi-turn (Stage 3) instruction: same final-answer format as `cot`, plus the
# check-then-answer protocol. The first attempt-1 smoke showed the Stage-2 policy never calls a
# tool it is not told about (0/96 episodes) — the affordance must be in the instruction; the
# *quality* of revision is what RL trains. Gold-free: it describes the protocol, never the answer.
MULTITURN_INSTRUCTION = (
    "You are an expert data analyst. Given a SQLite database schema and a question, write a "
    "single SQLite query that answers it. Before committing to an answer, check your query by "
    "calling the run_sql tool and reading the execution feedback; if it errors or the result "
    "does not answer the question, revise and check again (you have a limited number of checks). "
    "Then reason step by step inside <think></think> and give the final query in a "
    "```sql ... ``` code block."
)
# `mt_demo` = the `mt` instruction + a fixed in-context demonstration of the tool protocol
# (smoke attempt 3, the rung after attempt 2 showed the instruction alone leaves p(tool_call)
# at zero: the policy has never seen the tool-call wire format it must emit). The demonstration
# is structured chat messages (assistant `tool_calls` + `tool` feedback), so the chat template
# itself renders the exact format the trainer's parser expects — nothing is hand-rolled.
INSTRUCTIONS = {
    "cot": INSTRUCTION,
    "direct": DIRECT_INSTRUCTION,
    "mt": MULTITURN_INSTRUCTION,
    "mt_demo": MULTITURN_INSTRUCTION,
}

# The demonstration database is synthetic — not from any dataset split, so nothing can leak.
# Schema text goes through `serialize_schema` like every real example's.
_DEMO_SCHEMA = DatabaseSchema(
    db_id="office_plants",
    tables=(
        Table(
            name="plants",
            columns=(
                Column(name="id", type="int"),
                Column(name="species", type="text"),
                Column(name="room", type="text"),
            ),
            primary_key=("id",),
        ),
    ),
)
DEMO_SCHEMA_TEXT = serialize_schema(_DEMO_SCHEMA)
DEMO_QUESTION = "How many plants are in the Atrium?"
DEMO_WRONG_SQL = "SELECT COUNT(*) FROM plants WHERE location = 'Atrium'"
DEMO_FIXED_SQL = "SELECT COUNT(*) FROM plants WHERE room = 'Atrium'"
# Both feedback strings byte-match what `format_feedback` produces for these queries on the
# demonstration database — a unit test executes them in the real sandbox and asserts equality,
# so drift in the env's wording breaks the build, not the demonstration.
DEMO_ERROR_FEEDBACK = "Execution feedback: error (syntax): no such column: location"
DEMO_SUCCESS_FEEDBACK = "Execution feedback: query executed, returned 1 row. First rows: ((3,),)"
DEMO_FINAL_ANSWER = (
    '<think>My first query failed because the rooms are stored in the "room" column, not '
    '"location". The corrected query executed and returned a count, which is what the question '
    "asks for, so I can commit to it.</think>\n"
    f"```sql\n{DEMO_FIXED_SQL}\n```"
)


def _tool_call_message(query: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": "run_sql", "arguments": {"query": query}}}
        ],
    }


def tool_demo_messages() -> list[dict[str, object]]:
    """The fixed tool-protocol demonstration: wrong column → read the error → revise → success →
    final `<think>` + ```sql``` answer (two tool turns, the shape the smoke asserts). Fresh
    structures on every call so callers can never mutate shared state."""
    return [
        {"role": "user", "content": _user_content(DEMO_SCHEMA_TEXT, DEMO_QUESTION)},
        _tool_call_message(DEMO_WRONG_SQL),
        {"role": "tool", "content": DEMO_ERROR_FEEDBACK},
        _tool_call_message(DEMO_FIXED_SQL),
        {"role": "tool", "content": DEMO_SUCCESS_FEEDBACK},
        {"role": "assistant", "content": DEMO_FINAL_ANSWER},
    ]


class FewShotExemplar(Frozen):
    """A rendered few-shot demonstration: db_id, schema text, question, gold SQL, and evidence."""

    db_id: str
    schema_text: str
    question: str
    gold_sql: str
    evidence: str | None = None


def _user_content(schema_text: str, question: str, evidence: str | None = None) -> str:
    content = f"Database schema:\n{schema_text}\n\nQuestion: {question}"
    if evidence:
        content += f"\nEvidence: {evidence}"
    return content


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
                "content": _user_content(
                    exemplar.schema_text, exemplar.question, exemplar.evidence
                ),
            }
        )
        messages.append({"role": "assistant", "content": f"```sql\n{exemplar.gold_sql}\n```"})
    messages.append({"role": "user", "content": _user_content(schema_text, question, evidence)})
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
            evidence=example.evidence,
        )
        for example in chosen
    )
