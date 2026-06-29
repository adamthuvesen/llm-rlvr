"""Anti-collapse training diagnostics for self-correction analysis.

SCoRe's failure mode is invisible in aggregate reward until late: the policy sandbags turn 1 (or
stops genuinely revising) while final reward keeps rising. These pure functions compute the
collapse signature from a batch of scored trajectories so the trainer can log it per step:

  - turn-1 progress vs final reward (turn-1 falling while final rises = sandbagging),
  - fraction of episodes that actually revise, and the SQL edit distance when they do
    (near-zero distance = trivial-edit collapse),
  - the improvement-delta distribution (where the bonus is actually being earned).

CPU-pure over `(episode SQLs, TrajectoryBreakdown)` pairs — the SQLs are the `run_sql` tool
queries plus the final answer's extracted SQL, in order. Unit-testable without a trainer.
"""

from __future__ import annotations

from collections.abc import Sequence
from difflib import SequenceMatcher
from itertools import pairwise

from llm_rlvr._base import Frozen
from llm_rlvr.rewards.multiturn import TrajectoryBreakdown


class CollapseDiagnostics(Frozen):
    """One logging-step snapshot of the collapse-signature metrics."""

    n_episodes: int
    mean_turns_used: float
    frac_revising: float
    mean_edit_distance: float  # over consecutive revised-SQL pairs; 0.0 when nothing revises
    mean_turn1_progress: float  # execution tier of each episode's first query
    mean_final_reward: float
    mean_delta_bonus: float


def _sql_edit_distance(a: str, b: str) -> float:
    """Normalized SQL edit distance in [0, 1]: 0 = identical, 1 = nothing in common."""
    return 1.0 - SequenceMatcher(a=a.strip(), b=b.strip()).ratio()


def compute_diagnostics(
    episodes: Sequence[tuple[Sequence[str | None], TrajectoryBreakdown]],
) -> CollapseDiagnostics:
    """Aggregate the collapse signature over one batch of `(sqls, breakdown)` episodes."""
    if not episodes:
        raise ValueError("diagnostics need at least one episode")

    revising = 0
    distances: list[float] = []
    for sqls, _ in episodes:
        pairs = [(a, b) for a, b in pairwise(sqls) if a is not None and b is not None]
        revised = [(a, b) for a, b in pairs if a.strip() != b.strip()]
        if revised:
            revising += 1
            distances.extend(_sql_edit_distance(a, b) for a, b in revised)

    breakdowns = [breakdown for _, breakdown in episodes]
    n = len(episodes)
    return CollapseDiagnostics(
        n_episodes=n,
        mean_turns_used=sum(b.turns_used for b in breakdowns) / n,
        frac_revising=revising / n,
        mean_edit_distance=(sum(distances) / len(distances)) if distances else 0.0,
        mean_turn1_progress=sum(b.turn_progress[0] for b in breakdowns) / n,
        mean_final_reward=sum(b.final.total for b in breakdowns) / n,
        mean_delta_bonus=sum(b.delta_bonus for b in breakdowns) / n,
    )
