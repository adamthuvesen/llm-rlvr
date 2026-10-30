"""The base → SFT → RL progression table.

Renders the headline comparison: for each benchmark, the base and SFT execution-accuracy anchors
(single greedy numbers) and the RL policy aggregated at the **seed level** (mean + 95% CI over the
per-seed EX), plus the RL-vs-SFT gain and whether its CI excludes zero (the success bar). The
single-seed controls (cross-family, reward ablation) render as point estimates with no CI claim.

Eval strictness is explicit: the project reports **single-DB** execution accuracy, so the table
carries the documented single-DB false-positive caveat. Test-suite multi-DB accuracy would be
stricter, but it is not part of the public result.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Self

from pydantic import Field, TypeAdapter, model_validator

from llm_rlvr._base import Frozen
from llm_rlvr.eval.aggregate import SeedLevelStat, rl_vs_sft_gain, seed_level_ci

SINGLE_DB_CAVEAT = (
    "EX is single-DB execution-match (the training reward's metric). Single-DB carries a "
    "documented ~6.5% false-positive rate; test-suite (multi-DB) accuracy would be stricter, "
    "but it is not part of the public result."
)

Accuracy = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class RlRow(Frozen):
    """One RL variant on one benchmark: its per-seed EX and the gain over the SFT anchor."""

    label: str
    benchmark: str
    sft_anchor: Accuracy
    per_seed_ex: tuple[Accuracy, ...] = Field(min_length=1)
    ex: SeedLevelStat
    gain: SeedLevelStat

    @model_validator(mode="after")
    def aggregates_match_seeds(self) -> Self:
        if self.ex != seed_level_ci(self.per_seed_ex):
            raise ValueError("RL execution aggregate must match per_seed_ex")
        if self.gain != rl_vs_sft_gain(self.per_seed_ex, self.sft_anchor):
            raise ValueError("RL gain aggregate must match per_seed_ex and sft_anchor")
        return self

    @property
    def is_headline(self) -> bool:
        """A multi-seed variant makes a CI claim; a single-seed control is a point estimate only."""
        return self.ex.n >= 2


def build_rl_row(
    label: str, benchmark: str, *, sft_anchor: float, per_seed_ex: Sequence[float]
) -> RlRow:
    """Aggregate one RL variant's per-seed EX into its seed-level mean/CI and the gain over SFT."""
    return RlRow(
        label=label,
        benchmark=benchmark,
        sft_anchor=sft_anchor,
        per_seed_ex=tuple(per_seed_ex),
        ex=seed_level_ci(per_seed_ex),
        gain=rl_vs_sft_gain(per_seed_ex, sft_anchor),
    )


def _fmt_ci(stat: SeedLevelStat) -> str:
    if stat.ci_low is None or stat.ci_high is None:
        return f"{stat.mean:.3f}  (n=1, point)"
    return f"{stat.mean:.3f}  [{stat.ci_low:.3f}, {stat.ci_high:.3f}]"


def render_progression(
    benchmark: str,
    *,
    base_ex: float,
    sft_ex: float,
    rl_rows: Sequence[RlRow],
    single_db: bool = True,
) -> str:
    """Render the base → SFT → RL table for one benchmark, with the gain CIs and the caveat."""
    TypeAdapter(Accuracy).validate_python(base_ex)
    TypeAdapter(Accuracy).validate_python(sft_ex)
    for row in rl_rows:
        if row.benchmark != benchmark or row.sft_anchor != sft_ex:
            raise ValueError("RL row benchmark and SFT anchor must match the progression table")
    lines = [
        f"## {benchmark}  (execution accuracy)",
        f"{'setting':<28}  {'EX':>30}  {'Δ vs SFT (95% CI)':>26}",
        f"{'base  <think>':<28}  {base_ex:>30.3f}  {'':>26}",
        f"{'SFT   <think>':<28}  {sft_ex:>30.3f}  {'':>26}",
    ]
    for row in rl_rows:
        direction = (
            "  ✓>0"
            if row.gain.above_zero
            else "  (<0)"
            if row.gain.excludes_zero
            else "  (incl. 0)"
        )
        gain = (
            f"{row.gain.mean:+.3f} [{row.gain.ci_low:+.3f},{row.gain.ci_high:+.3f}]{direction}"
            if row.gain.ci_low is not None
            else f"{row.gain.mean:+.3f}  (n=1, point)"
        )
        lines.append(f"{row.label:<28}  {_fmt_ci(row.ex):>30}  {gain:>26}")
    if single_db:
        lines.append("")
        lines.append(f"_Caveat:_ {SINGLE_DB_CAVEAT}")
    return "\n".join(lines)
