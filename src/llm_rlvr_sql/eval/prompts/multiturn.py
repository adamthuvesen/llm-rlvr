"""Multiturn prompt styles and the fixed tool-protocol demonstration."""

from __future__ import annotations

from llm_rlvr_sql.data.models import Column, DatabaseSchema, Table
from llm_rlvr_sql.data.schema import serialize_schema
from llm_rlvr_sql.eval.messages import tool_call_message, user_content

MULTITURN_INSTRUCTION = (
    "You are an expert data analyst. Given a SQLite database schema and a question, write a "
    "single SQLite query that answers it. Before committing to an answer, check your query by "
    "calling the run_sql tool and reading the execution feedback; if it errors or the result "
    "does not answer the question, revise and check again (you have a limited number of checks). "
    "Then reason step by step inside <think></think> and give the final query in a "
    "```sql ... ``` code block."
)

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
DEMO_ERROR_FEEDBACK = "Execution feedback: error (syntax): no such column: location"
DEMO_SUCCESS_FEEDBACK = "Execution feedback: query executed, returned 1 row. First rows: ((3,),)"
DEMO_FINAL_ANSWER = (
    '<think>My first query failed because the rooms are stored in the "room" column, not '
    '"location". The corrected query executed and returned a count, which is what the question '
    "asks for, so I can commit to it.</think>\n"
    f"```sql\n{DEMO_FIXED_SQL}\n```"
)


def tool_demo_messages() -> list[dict[str, object]]:
    """Fixed in-context tool-protocol demonstration for ``mt_demo`` prompts."""
    return [
        {"role": "user", "content": user_content(DEMO_SCHEMA_TEXT, DEMO_QUESTION)},
        tool_call_message(DEMO_WRONG_SQL),
        {"role": "tool", "content": DEMO_ERROR_FEEDBACK},
        tool_call_message(DEMO_FIXED_SQL),
        {"role": "tool", "content": DEMO_SUCCESS_FEEDBACK},
        {"role": "assistant", "content": DEMO_FINAL_ANSWER},
    ]
