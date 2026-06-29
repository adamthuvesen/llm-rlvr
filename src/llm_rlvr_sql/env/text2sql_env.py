"""The multi-turn text-to-SQL tool environment (Stage 3, design D2).

Conforms to TRL v1.5.1's `environment_factory` contract (verified against the tagged source):
the trainer instantiates one environment per rollout slot via a zero-arg factory, calls
`reset(**dataset_row)` before each episode, and exposes **every public method** as a model-
callable tool (schema derived from type hints + the Google-style docstring). The trainer owns
the loop: the model calls `run_sql` to try a query, reads the execution feedback, and ends the
episode by giving its final answer instead of another tool call — bounded by
`GRPOConfig.max_tool_calling_iterations` *and* this environment's own turn budget (defense in
depth; ours is the unit-tested one). Tool-output tokens are loss-masked by the trainer's
`tool_mask`; the reward function reads per-episode state via the `environments=` kwarg.

Feedback is deterministic and gold-free: error message (classified + truncated) on failure, row
count + a capped preview on success. It NEVER contains gold SQL, gold rows, or a verifier
verdict — only what executing the model's own query returns. Instances are reused across
episodes, so `reset` fully restores state.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from llm_rlvr_sql._base import Frozen
from llm_rlvr_sql.data.paths import resolve_database
from llm_rlvr_sql.env.sandbox import ExecutionError, ExecutionOutcome, SqliteSandbox

# Agent-facing execution is deliberately tighter than verification: the model only needs a
# preview, and a rollout must not stall training on a pathological query. Reward scoring runs
# verification-grade settings separately — these caps never decide reward.
AGENT_ROW_CAP = 50
AGENT_TIMEOUT_S = 10.0

BUDGET_EXHAUSTED_FEEDBACK = (
    "Execution feedback: turn budget exhausted — no more queries can be run. "
    "Give your final answer now."
)


class FeedbackConfig(Frozen):
    """The environment knobs: turn budget + deterministic feedback formatting."""

    max_turns: int = 3
    error_message_chars: int = 300
    # Success feedback: row count always; leading-row preview only when `include_rows` (the
    # error-only variant is the cheaper prompt ablated at the pilot — design Open Question).
    include_rows: bool = True
    preview_rows: int = 5
    preview_chars: int = 500


class TurnRecord(Frozen):
    """One executed tool turn: the query the model ran and the feedback it saw."""

    query: str
    feedback: str


def format_feedback(outcome: ExecutionOutcome, config: FeedbackConfig) -> str:
    """Render an execution outcome as the model-facing observation. Deterministic, gold-free."""
    if isinstance(outcome, ExecutionError):
        message = outcome.message[: config.error_message_chars]
        return f"Execution feedback: error ({outcome.error_type}): {message}"

    count = f"{outcome.row_count} row{'s' if outcome.row_count != 1 else ''}"
    if outcome.truncated:
        count += " (truncated)"
    if not config.include_rows or outcome.row_count == 0:
        return f"Execution feedback: query executed, returned {count}."
    preview = repr(outcome.rows[: config.preview_rows])[: config.preview_chars]
    return f"Execution feedback: query executed, returned {count}. First rows: {preview}"


class SqliteText2SqlEnv:
    """One rollout slot's environment; `reset` rebinds it to an episode's database.

    Public methods other than `reset` become model-callable tools under TRL's contract — keep
    everything else underscore-private.
    """

    def __init__(
        self,
        db_root: str | Path,
        *,
        config: FeedbackConfig | None = None,
        resolver: Callable[[str, str | Path], Path] = resolve_database,
    ) -> None:
        if (config or FeedbackConfig()).max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self._db_root = db_root
        self._config = config or FeedbackConfig()
        self._resolver = resolver
        self._sandbox: SqliteSandbox | None = None
        self._turns: list[TurnRecord] = []

    @property
    def turns(self) -> tuple[TurnRecord, ...]:
        """The episode's executed turns — read by the reward function via `environments=`."""
        return tuple(self._turns)

    def reset(self, **kwargs: object) -> str | None:
        """Bind to the episode's database and clear state. Receives the whole dataset row;
        requires its `db_id` column. Returns None (the prompt already carries the schema)."""
        db_id = kwargs.get("db_id")
        if not isinstance(db_id, str) or not db_id:
            raise ValueError("reset requires a non-empty `db_id` dataset column")
        db_path = self._resolver(db_id, self._db_root)
        self._sandbox = SqliteSandbox(db_path, timeout_s=AGENT_TIMEOUT_S, row_cap=AGENT_ROW_CAP)
        self._turns = []
        return None

    def run_sql(self, query: str) -> str:
        """Run one read-only SQL query against the question's database and see what it returns.

        Use this to check your SQL before giving the final answer: you will see the error
        message if it fails, or the row count and first rows if it succeeds. If the result is
        not what the question asks for, fix the query and try again.

        Args:
            query: A single SQLite SELECT statement.

        Returns:
            The execution feedback: an error description, or the row count plus leading rows.
        """
        if self._sandbox is None:
            raise RuntimeError("reset() must run before run_sql()")
        if len(self._turns) >= self._config.max_turns:
            return BUDGET_EXHAUSTED_FEEDBACK
        feedback = format_feedback(self._sandbox.execute(query), self._config)
        self._turns.append(TurnRecord(query=query, feedback=feedback))
        return feedback


def make_environment_factory(
    db_root: str | Path,
    *,
    config: FeedbackConfig | None = None,
) -> Callable[[], SqliteText2SqlEnv]:
    """The zero-arg factory `GRPOTrainer(environment_factory=...)` expects."""

    def factory() -> SqliteText2SqlEnv:
        return SqliteText2SqlEnv(db_root, config=config)

    return factory
