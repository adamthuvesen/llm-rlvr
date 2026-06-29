"""Shared chat message builders for eval and training."""

from __future__ import annotations


def user_content(schema_text: str, question: str, evidence: str | None = None) -> str:
    content = f"Database schema:\n{schema_text}\n\nQuestion: {question}"
    if evidence:
        content += f"\nEvidence: {evidence}"
    return content


def tool_call_message(query: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": "run_sql", "arguments": {"query": query}}}
        ],
    }
