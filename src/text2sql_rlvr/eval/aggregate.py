"""Seed-level aggregation for the RL headline (Stage 2, design D6).

The RL gain is aggregated at the **seed level**: each seed contributes exactly one EX number (its
committed `CompletionSet`'s execution accuracy), and the confidence interval is computed over those
per-seed numbers — rollouts are never pooled across seeds (pooling inflates significance; Colas et
al.). The RL-vs-SFT gain is the per-seed RL EX minus the (deterministic, greedy) SFT anchor; its CI
excluding zero is the headline success bar.

The CI is a Student-t interval (small n — 3 seeds is the floor). With no scipy dependency, the
two-sided 95% t critical values are a small lookup table (df 1-30) with a normal fallback above;
the confidence level is fixed at 95% to match that table.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from text2sql_rlvr._base import Frozen

# Two-sided 95% Student-t critical values by degrees of freedom (df = n - 1). Above df=30 the
# t-interval is within ~2% of the normal, so 1.96 is the documented fallback.
_T_CRIT_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}
_Z_95 = 1.96


class SeedLevelStat(Frozen):
    """A seed-level mean with a 95% Student-t CI over `n` seeds. `n=1` reports no interval."""

    mean: float
    std: float
    n: int
    ci_low: float | None
    ci_high: float | None
    confidence: float = 0.95

    @property
    def excludes_zero(self) -> bool:
        """True iff the CI is defined and lies entirely above or below zero (the gain bar)."""
        if self.ci_low is None or self.ci_high is None:
            return False
        return self.ci_low > 0.0 or self.ci_high < 0.0


def _t_crit_95(df: int) -> float:
    return _T_CRIT_95.get(df, _Z_95)


def seed_level_ci(values: Sequence[float]) -> SeedLevelStat:
    """Mean + 95% Student-t CI over per-seed values. A single seed reports a point (no interval)."""
    if not values:
        raise ValueError("seed_level_ci needs at least one per-seed value")
    n = len(values)
    mean = statistics.fmean(values)
    if n == 1:
        return SeedLevelStat(mean=mean, std=0.0, n=1, ci_low=None, ci_high=None)
    std = statistics.stdev(values)  # sample std (ddof=1) — the seed-level dispersion
    half = _t_crit_95(n - 1) * std / (n**0.5)
    return SeedLevelStat(mean=mean, std=std, n=n, ci_low=mean - half, ci_high=mean + half)


def rl_vs_sft_gain(rl_per_seed: Sequence[float], sft_anchor: float) -> SeedLevelStat:
    """Seed-level CI of the RL-vs-SFT gain: each seed's RL EX minus the constant SFT anchor.

    The anchor is the deterministic (greedy) SFT EX — one number — so the gain's dispersion is the
    RL seeds' dispersion, centered at `mean(RL) - anchor`. `excludes_zero` is the headline bar.
    """
    return seed_level_ci([rl - sft_anchor for rl in rl_per_seed])
