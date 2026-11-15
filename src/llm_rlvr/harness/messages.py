"""Validated chat messages shared by task traces and prepared training data."""

from __future__ import annotations

from typing import Literal

from pydantic import field_validator

from llm_rlvr._base import Frozen


class ChatMessage(Frozen):
    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def nonempty_content(cls, content: str) -> str:
        if not content.strip():
            raise ValueError("message content must not be empty")
        return content


def validate_prompt(prompt: tuple[ChatMessage, ...]) -> tuple[ChatMessage, ...]:
    roles = [message.role for message in prompt]
    if roles and roles[0] == "system":
        roles = roles[1:]
    if (
        not roles
        or roles[-1] != "user"
        or any(
            role != ("user" if index % 2 == 0 else "assistant") for index, role in enumerate(roles)
        )
    ):
        raise ValueError("prompt must alternate user/assistant turns and end with user")
    return prompt
