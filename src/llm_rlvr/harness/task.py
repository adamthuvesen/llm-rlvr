"""The `Task` protocol: the surface implemented by a concrete RLVR task.

Problem-specific verification and GRPO inputs are defined here. The shared GRPO driver consumes
`make_reward_fn` and `build_grpo_rows`; `build_prompts` is available to task-specific callers.
Text-to-SQL (`llm_rlvr.tasks.sql`) is the reference implementation, and `docs/harness.md` is the
step-by-step guide.

The split:
  - **verification** (`parser`, `make_verifier`, `make_reward_fn`): answers are parsed, checked, and
    turned into training rewards.
  - **data to prompt** (`build_prompts`, `build_grpo_rows`): examples are rendered into chat prompts
    and GRPO rows with the per-example `reference` and `context` required by the reward.

`resources` is whatever per-task lookup the prompt/reward need (for SQL: the database schemas);
`context_root` is where the verifier resolves its context from (for SQL: the database root).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from llm_rlvr.harness.answer import AnswerParser
from llm_rlvr.harness.reward import RewardConfig
from llm_rlvr.harness.verifier import Verifier

# A TRL-style reward function: called with `completions=` plus the dataset columns as aligned
# keyword lists, returns one float per completion.
RewardFn = Callable[..., list[float]]

# Chat messages, as consumed by a tokenizer's chat template.
Messages = list[dict[str, str]]


@runtime_checkable
class Task(Protocol):
    """The full extension surface for one verifiable task."""

    name: str
    parser: AnswerParser

    def make_verifier(self) -> Verifier:
        """The evaluation verifier (official, unguarded semantics)."""
        ...

    def make_reward_fn(
        self, context_root: str | Path, *, config: RewardConfig | None = None
    ) -> RewardFn:
        """The GRPO training reward closure (resolves per-example context under `context_root`)."""
        ...

    def build_prompts(self, examples: Any, resources: Any, **kwargs: object) -> list[Messages]:
        """Render chat prompts for `examples`."""
        ...

    def build_grpo_rows(
        self, examples: Any, resources: Any, **kwargs: object
    ) -> list[dict[str, object]]:
        """One GRPO row per example: `prompt` plus the `reference` + `context` for the reward."""
        ...
