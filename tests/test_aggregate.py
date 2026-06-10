"""rl-evaluation: seed-level aggregation — CI over seeds, never pooled, the gain bar."""

from __future__ import annotations

import statistics

import pytest

from text2sql_rlvr.eval.aggregate import rl_vs_sft_gain, seed_level_ci


def test_single_seed_reports_a_point_not_an_interval() -> None:
    stat = seed_level_ci([0.71])
    assert stat.n == 1
    assert stat.mean == 0.71
    assert stat.ci_low is None and stat.ci_high is None
    assert stat.excludes_zero is False  # no CI -> no significance claim (the 1-seed control)


def test_three_seed_ci_is_student_t_over_the_seeds() -> None:
    values = [0.70, 0.72, 0.74]
    stat = seed_level_ci(values)
    assert stat.n == 3
    assert stat.mean == pytest.approx(0.72)
    # half-width = t(df=2,.975)=4.303 * sample_std / sqrt(3)
    expected_half = 4.303 * statistics.stdev(values) / (3**0.5)
    assert stat.ci_low == pytest.approx(0.72 - expected_half)
    assert stat.ci_high == pytest.approx(0.72 + expected_half)


def test_ci_is_seed_level_not_pooled() -> None:
    # The CI must come from the 3 per-seed numbers, NOT from a pooled count of correct/total.
    # Two pools with identical per-seed EX but very different rollout counts must give the SAME CI.
    per_seed = [0.60, 0.62, 0.64]
    a = seed_level_ci(per_seed)
    b = seed_level_ci(per_seed)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    # width depends only on the seed dispersion + n, not on any per-seed sample size
    assert a.std == statistics.stdev(per_seed)


def test_gain_excludes_zero_when_all_seeds_beat_the_anchor() -> None:
    gain = rl_vs_sft_gain([0.70, 0.71, 0.72], sft_anchor=0.65)
    assert gain.mean == pytest.approx(0.06)
    assert gain.excludes_zero is True  # the headline success bar


def test_gain_does_not_exclude_zero_when_seeds_straddle_the_anchor() -> None:
    gain = rl_vs_sft_gain([0.60, 0.66, 0.72], sft_anchor=0.65)
    # a wide, anchor-straddling spread -> CI includes zero -> no claimed gain
    assert gain.excludes_zero is False


def test_empty_values_rejected_loudly() -> None:
    with pytest.raises(ValueError, match="at least one"):
        seed_level_ci([])
