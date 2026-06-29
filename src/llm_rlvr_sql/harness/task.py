"""The `Task` protocol — the surface a concrete RLVR task implements.

A task bundles everything the shared SFT / GRPO / eval machinery needs that is *specific* to a
problem, so the drivers stay task-agnostic. To add a task (e.g. your own model on your own problem),
implement these members; text-to-SQL (`llm_rlvr_sql.tasks.sql`) is the reference implementation;
`docs/harness.md` is the step-by-step guide.

The split:
  - **verification** (`parser`, `make_verifier`, `make_reward_fn`) — the RLVR core: parse the
    answer, check it, turn the check into a training reward. This makes the reward verifiable.
  - **data → prompt** (`build_prompts`, `build_grpo_rows`) — render examples into chat prompts and
    into GRPO rows that carry the per-example `reference` + `context` the reward needs.

`resources` is whatever per-task lookup the prompt/reward need (for SQL: the database schemas);
`context_root` is where the verifier resolves its context from (for SQL: the database root).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from llm_rlvr_sql.harness.answer import AnswerParser
from llm_rlvr_sql.harness.reward import RewardConfig
from llm_rlvr_sql.harness.verifier import Verifier

# A TRL-style reward function: called with `completions=` plus the dataset columns as aligned
# keyword lists, returns one float per completion.
RewardFn = Callable[..., list[float]]

# Chat messages, as consumed by a tokenizer's chat template.
Messages = list[dict[str, object]]


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

    def build_prompts(
        self, examples: Sequence[object], resources: Mapping[str, object], **kwargs: object
    ) -> list[Messages]:
        """Render chat prompts for `examples`."""
        ...

    def build_grpo_rows(
        self, examples: Sequence[object], resources: Mapping[str, object], **kwargs: object
    ) -> list[dict[str, object]]:
        """One GRPO row per example: `prompt` plus the `reference` + `context` for the reward."""
        ...
