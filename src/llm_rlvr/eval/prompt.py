"""Compatibility imports for SQL prompts."""

from llm_rlvr.tasks.sql.prompt import (
    DIRECT_INSTRUCTION,
    INSTRUCTION,
    INSTRUCTIONS,
    FewShotExemplar,
    build_messages,
    select_exemplars,
    user_content,
)

__all__ = [
    "DIRECT_INSTRUCTION",
    "INSTRUCTION",
    "INSTRUCTIONS",
    "FewShotExemplar",
    "build_messages",
    "select_exemplars",
    "user_content",
]
