"""CPU tests: multiturn environment, reward, diagnostics, and provenance."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_rlvr_sql.env import (
    FeedbackConfig,
    SqliteText2SqlEnv,
    format_feedback,
    make_environment_factory,
)
from llm_rlvr_sql.env.sandbox import ExecutionResult
from llm_rlvr_sql.env.text2sql_env import BUDGET_EXHAUSTED_FEEDBACK
from llm_rlvr_sql.rewards import (
    MultiTurnRewardConfig,
    RewardConfig,
    score_trajectory,
)
from llm_rlvr_sql.train.diagnostics import compute_diagnostics
from llm_rlvr_sql.train.grpo import GrpoHyperparams
from llm_rlvr_sql.train.multiturn_provenance import (
    MultiTurnProvenance,
    build_multiturn_provenance,
)

GOLD = "SELECT a FROM t"


def _final(sql: str) -> str:
    """A well-formed final answer: leading <think> + one sql fence."""
    return f"<think>\nchecked it\n</think>\n\n```sql\n{sql}\n```"


def _env(scoring_root: Path, **config: object) -> SqliteText2SqlEnv:
    env = SqliteText2SqlEnv(scoring_root, config=FeedbackConfig(**config))  # type: ignore[arg-type]
    env.reset(db_id="concert", prompt="ignored", gold_sql=GOLD)
    return env


# --- multiturn-environment ---


def test_reset_requires_db_id(scoring_root: Path) -> None:
    env = SqliteText2SqlEnv(scoring_root)
    with pytest.raises(ValueError, match="db_id"):
        env.reset(prompt="no db column")


def test_run_sql_before_reset_is_loud(scoring_root: Path) -> None:
    env = SqliteText2SqlEnv(scoring_root)
    with pytest.raises(RuntimeError, match="reset"):
        env.run_sql("SELECT 1")


def test_success_feedback_has_count_preview_and_records_turn(scoring_root: Path) -> None:
    env = _env(scoring_root)
    feedback = env.run_sql("SELECT a FROM t")
    assert "2 rows" in feedback and "First rows:" in feedback
    assert len(env.turns) == 1 and env.turns[0].query == "SELECT a FROM t"


def test_error_feedback_is_classified_and_truncated(scoring_root: Path) -> None:
    env = _env(scoring_root, error_message_chars=20)
    feedback = env.run_sql("SELECT nope FROM missing_table")
    assert feedback.startswith("Execution feedback: error (syntax):")
    assert len(feedback) <= len("Execution feedback: error (syntax): ") + 20


def test_write_statement_is_rejected_and_db_unchanged(scoring_root: Path) -> None:
    env = _env(scoring_root)
    feedback = env.run_sql("DROP TABLE t")
    assert "error" in feedback
    assert "2 rows" in env.run_sql("SELECT a FROM t")  # the table is still there


def test_feedback_is_deterministic_and_gold_free(scoring_root: Path) -> None:
    a = _env(scoring_root).run_sql("SELECT b FROM t")
    b = _env(scoring_root).run_sql("SELECT b FROM t")
    assert a == b  # determinism: same query, same observation
    # gold-free: a wrong-column query gets no verdict, no gold SQL, no gold rows
    for token in ("correct", "incorrect", "wrong", "match", GOLD):
        assert token not in a


def test_turn_budget_stops_execution_without_recording(scoring_root: Path) -> None:
    env = _env(scoring_root, max_turns=2)
    env.run_sql("SELECT a FROM t")
    env.run_sql("SELECT b FROM t")
    assert env.run_sql("SELECT a FROM t") == BUDGET_EXHAUSTED_FEEDBACK
    assert len(env.turns) == 2


def test_reset_fully_restores_state(scoring_root: Path) -> None:
    env = _env(scoring_root, max_turns=1)
    env.run_sql("SELECT a FROM t")
    assert env.run_sql("SELECT a FROM t") == BUDGET_EXHAUSTED_FEEDBACK
    env.reset(db_id="concert")
    assert env.turns == ()  # pooled instances are reused; reset must clear everything
    assert "2 rows" in env.run_sql("SELECT a FROM t")


def test_factory_is_zero_arg_and_makes_fresh_envs(scoring_root: Path) -> None:
    factory = make_environment_factory(scoring_root, config=FeedbackConfig(max_turns=2))
    assert factory() is not factory()


def test_feedback_formatting_variants() -> None:
    cfg = FeedbackConfig()
    empty = format_feedback(ExecutionResult(rows=()), cfg)
    assert empty.endswith("returned 0 rows.") and "First rows" not in empty
    no_preview = format_feedback(ExecutionResult(rows=((1,),)), FeedbackConfig(include_rows=False))
    assert "First rows" not in no_preview and "1 row" in no_preview
    truncated = format_feedback(ExecutionResult(rows=((1,),), truncated=True), cfg)
    assert "(truncated)" in truncated


# --- multiturn-reward ---


def test_hybrid_rewards_progress_and_final(scoring_root: Path) -> None:
    db = scoring_root / "concert" / "concert.sqlite"
    # turn 1 fails, the (changed) final answer matches: both the outcome and the path count.
    b = score_trajectory(["SELECT nope FROM missing"], _final(GOLD), GOLD, db)
    assert b.final.result_match == 1.0
    assert b.turn_progress == (0.0, 1.0)
    assert b.delta_bonus == 1.0  # 0.0 -> 1.0, SQL changed
    assert b.total > b.final.total  # progress + delta add on top of the final reward
    assert b.turns_used == 2


def test_unchanged_resubmission_earns_zero_delta(scoring_root: Path) -> None:
    db = scoring_root / "concert" / "concert.sqlite"
    b = score_trajectory([GOLD], _final(GOLD), GOLD, db)
    assert b.turn_progress == (1.0, 1.0)
    assert b.delta_bonus == 0.0  # same SQL resubmitted: no improvement credit


def test_degradation_earns_no_delta(scoring_root: Path) -> None:
    db = scoring_root / "concert" / "concert.sqlite"
    b = score_trajectory([GOLD], _final("SELECT b FROM t"), GOLD, db)
    assert b.turn_progress == (1.0, 0.1)
    assert b.delta_bonus == 0.0  # only positive deltas count


def test_trajectory_arm_is_outcome_only(scoring_root: Path) -> None:
    db = scoring_root / "concert" / "concert.sqlite"
    cfg = MultiTurnRewardConfig(credit="trajectory")
    b = score_trajectory(["SELECT nope FROM missing"], _final(GOLD), GOLD, db, config=cfg)
    assert b.total == b.final.total
    assert b.delta_bonus == 0.0


def test_empty_result_guard_applies_per_turn(scoring_root: Path) -> None:
    db = scoring_root / "concert" / "concert.sqlite"
    # both pred and gold genuinely empty: the guard blocks the always-empty hack per turn too.
    b = score_trajectory(
        ["SELECT a FROM t WHERE a=88"], _final(GOLD), "SELECT a FROM t WHERE a=99", db
    )
    assert b.turn_progress[0] == 0.1  # executable, but the empty "match" earns no tier-1 credit


def test_no_tool_calls_still_scores_the_final_answer(scoring_root: Path) -> None:
    db = scoring_root / "concert" / "concert.sqlite"
    b = score_trajectory([], _final(GOLD), GOLD, db)
    assert b.turns_used == 1
    assert b.turn_progress == (1.0,)
    assert b.delta_bonus == 0.0


# --- collapse diagnostics ---


def test_diagnostics_separate_revising_from_static(scoring_root: Path) -> None:
    db = scoring_root / "concert" / "concert.sqlite"
    revising = score_trajectory(["SELECT nope FROM missing"], _final(GOLD), GOLD, db)
    static = score_trajectory([], _final(GOLD), GOLD, db)
    diag = compute_diagnostics(
        [
            (["SELECT nope FROM missing", GOLD], revising),
            ([GOLD], static),
        ]
    )
    assert diag.n_episodes == 2
    assert diag.frac_revising == 0.5
    assert diag.mean_edit_distance > 0.0
    assert diag.mean_turn1_progress == 0.5  # (0.0 + 1.0) / 2
    assert diag.mean_delta_bonus == 0.5


def test_diagnostics_zero_distance_when_nothing_revises(scoring_root: Path) -> None:
    db = scoring_root / "concert" / "concert.sqlite"
    static = score_trajectory([GOLD], _final(GOLD), GOLD, db)
    diag = compute_diagnostics([([GOLD, GOLD], static)])
    assert diag.frac_revising == 0.0
    assert diag.mean_edit_distance == 0.0


def test_diagnostics_reject_empty_batch() -> None:
    with pytest.raises(ValueError, match="at least one"):
        compute_diagnostics([])


# --- multi-turn provenance ---


def test_multiturn_provenance_round_trips_with_env_and_reward_config() -> None:
    prov = build_multiturn_provenance(
        base_model="Qwen/Qwen2.5-Coder-3B-Instruct",
        warm_start_adapter="grpo/grpo-s0",
        hyperparams=GrpoHyperparams(),
        environment=FeedbackConfig(max_turns=3, include_rows=True),
        multiturn_reward=MultiTurnRewardConfig(base=RewardConfig(arm="shaped")),
        dataset="spider",
        dataset_revision="rev",
        n_train=100,
        n_dev=10,
        commit="abc123",
        dirty=False,
    )
    restored = MultiTurnProvenance.model_validate_json(prov.model_dump_json())
    assert restored == prov
    assert restored.grpo.warm_start_adapter == "grpo/grpo-s0"
    assert restored.environment.max_turns == 3
    assert restored.multiturn_reward.first_turn_kl_anchor is True  # the guard default is ON
    assert restored.grpo.git_commit == "abc123"
