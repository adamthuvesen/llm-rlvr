"""rl-evaluation report: progression table, seed-level CIs, single-seed control, caveat."""

from __future__ import annotations

from text2sql_rlvr.report import SINGLE_DB_CAVEAT, build_rl_row, render_progression


def test_headline_row_carries_seed_level_ci_and_gain_bar() -> None:
    row = build_rl_row("RL 3-seed", "BIRD-dev", sft_anchor=0.344, per_seed_ex=[0.40, 0.41, 0.42])
    assert row.is_headline  # 3 seeds -> a CI claim
    assert row.gain.ci_low is not None
    assert row.gain.excludes_zero is True  # all seeds clear the anchor
    assert abs(row.ex.mean - 0.41) < 1e-9


def test_single_seed_control_is_point_estimate_no_ci() -> None:
    row = build_rl_row("Llama control", "BIRD-dev", sft_anchor=0.344, per_seed_ex=[0.39])
    assert row.is_headline is False
    assert row.ex.ci_low is None
    assert row.gain.excludes_zero is False  # n=1 -> no significance claim


def test_render_includes_rows_gain_and_caveat() -> None:
    rows = [
        build_rl_row("RL 3-seed", "BIRD-dev", sft_anchor=0.344, per_seed_ex=[0.40, 0.41, 0.42]),
        build_rl_row("Llama 1-seed", "BIRD-dev", sft_anchor=0.344, per_seed_ex=[0.37]),
    ]
    out = render_progression("BIRD-dev", base_ex=0.313, sft_ex=0.344, rl_rows=rows, single_db=True)
    assert "BIRD-dev" in out
    assert "base  <think>" in out and "SFT   <think>" in out
    assert "RL 3-seed" in out and "Llama 1-seed" in out
    assert "✓>0" in out  # the gain bar marker for the significant headline
    assert "n=1, point" in out  # the control labelled as a point estimate
    assert SINGLE_DB_CAVEAT in out


def test_caveat_suppressed_when_not_single_db() -> None:
    rows = [build_rl_row("RL", "Spider-dev", sft_anchor=0.633, per_seed_ex=[0.70, 0.71, 0.72])]
    out = render_progression("Spider-dev", base_ex=0.6, sft_ex=0.633, rl_rows=rows, single_db=False)
    assert SINGLE_DB_CAVEAT not in out
