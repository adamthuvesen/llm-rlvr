"""Gold-free execution feedback for forced-revision rollouts and eval."""

from __future__ import annotations

from pathlib import Path

from llm_rlvr.env.sandbox import SqliteSandbox
from llm_rlvr.env.text2sql_env import (
    AGENT_ROW_CAP,
    AGENT_TIMEOUT_S,
    FeedbackConfig,
    format_feedback,
)
from llm_rlvr.eval.parsing import extract_sql

UNPARSEABLE_FEEDBACK = (
    "Execution feedback: error (parse): no SQL code block found in your answer. "
    "Write a single SQLite query in a ```sql ... ``` code block."
)

REVISE_INSTRUCTION = (
    "Above is the execution feedback for your query. If the query errored or its result does "
    "not answer the question, revise it; if it is already correct, you may keep it. Reason "
    "step by step inside <think></think>, then give the final query in a "
    "```sql ... ``` code block."
)


def revision_user_message(feedback: str) -> str:
    return f"{feedback}\n\n{REVISE_INSTRUCTION}"


def is_error_feedback(feedback: str) -> bool:
    """Whether a feedback turn reports an execution or parse error."""
    return feedback.startswith("Execution feedback: error")


def attempt_feedback(
    attempt_text: str,
    db_path: str | Path,
    config: FeedbackConfig | None = None,
) -> str:
    sql = extract_sql(attempt_text)
    if sql is None:
        return UNPARSEABLE_FEEDBACK
    sandbox = SqliteSandbox(db_path, timeout_s=AGENT_TIMEOUT_S, row_cap=AGENT_ROW_CAP)
    return format_feedback(sandbox.execute(sql), config or FeedbackConfig())
