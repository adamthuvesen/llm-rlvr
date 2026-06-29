"""The base → SFT → RL progression table (Stage 2, design D6).

Renders the headline comparison: for each benchmark, the base and SFT execution-accuracy anchors
(single greedy numbers) and the RL policy aggregated at the **seed level** (mean + 95% CI over the
per-seed EX), plus the RL-vs-SFT gain and whether its CI excludes zero (the success bar). The
single-seed controls (cross-family, reward ablation) render as point estimates with no CI claim.

Eval strictness is disclosed: the project reports **single-DB** execution accuracy, so the table
carries the documented single-DB false-positive caveat (the verifier supports test-suite multi-DB if
the distilled databases are later self-hosted — see the design Open Questions).
"""

from __future__ import annotations

from collections.abc import Sequence

from llm_rlvr._base import Frozen
from llm_rlvr.eval.aggregate import SeedLevelStat, rl_vs_sft_gain, seed_level_ci

SINGLE_DB_CAVEAT = (
    "EX is single-DB execution-match (the training reward's metric). Single-DB carries a "
    "documented ~6.5% false-positive rate; test-suite (multi-DB) accuracy would be stricter and is "
    "available if the distilled Spider DBs are self-hosted (verifier already supports `db_dir`)."
)


class RlRow(Frozen):
    """One RL variant on one benchmark: its per-seed EX and the gain over the SFT anchor."""

    label: str
    benchmark: str
    sft_anchor: float
    per_seed_ex: tuple[float, ...]
    ex: SeedLevelStat
    gain: SeedLevelStat

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
    lines = [
        f"## {benchmark}  (execution accuracy)",
        f"{'setting':<28}{'EX':>30}{'Δ vs SFT (95% CI)':>26}",
        f"{'base  <think>':<28}{base_ex:>30.3f}{'':>26}",
        f"{'SFT   <think>':<28}{sft_ex:>30.3f}{'':>26}",
    ]
    for row in rl_rows:
        gain = (
            f"{row.gain.mean:+.3f} [{row.gain.ci_low:+.3f},{row.gain.ci_high:+.3f}]"
            f"{'  ✓>0' if row.gain.excludes_zero else '  (incl. 0)'}"
            if row.gain.ci_low is not None
            else f"{row.gain.mean:+.3f}  (n=1, point)"
        )
        lines.append(f"{row.label:<28}{_fmt_ci(row.ex):>30}{gain:>26}")
    if single_db:
        lines.append("")
        lines.append(f"_Caveat:_ {SINGLE_DB_CAVEAT}")
    return "\n".join(lines)
