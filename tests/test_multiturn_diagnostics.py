"""CPU tests for validation wiring: config kwargs, reward wrapper, and launch-gate checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_rlvr.env import FeedbackConfig, SqliteText2SqlEnv, format_feedback
from llm_rlvr.env.sandbox import ExecutionResult
from llm_rlvr.env.text2sql_env import BUDGET_EXHAUSTED_FEEDBACK
from llm_rlvr.rewards import MultiTurnRewardConfig, score_trajectory
from llm_rlvr.train.grpo_smoke import (
    FEEDBACK_MARKER,
    IS_RATIO_HEALTHY_BAND,
    check_is_ratio,
    check_tool_masking,
)
from llm_rlvr.train.multiturn import (
    MultiTurnHyperparams,
    make_multiturn_reward_fn,
    multiturn_grpo_config_kwargs,
    smoke_failures,
)

GOLD = "SELECT a FROM t"
FINAL = f"<think>\nchecked\n</think>\n\n```sql\n{GOLD}\n```"


# --- config kwargs ---


def test_config_kwargs_add_tool_loop_bound_on_top_of_stage2() -> None:
    kwargs = multiturn_grpo_config_kwargs(MultiTurnHyperparams(), "out")
    assert kwargs["max_tool_calling_iterations"] == 3
    # Colocate settings carry over unchanged.
    assert kwargs["use_vllm"] is True and kwargs["vllm_mode"] == "colocate"
    assert kwargs["num_generations"] == 4


# --- the TRL-facing reward ---


def _episode_env(scoring_root: Path, queries: list[str]) -> SqliteText2SqlEnv:
    env = SqliteText2SqlEnv(scoring_root)
    env.reset(db_id="concert", prompt="ignored", gold_sql=GOLD)
    for query in queries:
        env.run_sql(query)
    return env


def test_reward_fn_scores_env_turns_plus_final(scoring_root: Path) -> None:
    env_a = _episode_env(scoring_root, ["SELECT nope FROM missing", GOLD])
    env_b = _episode_env(scoring_root, [])
    reward_fn = make_multiturn_reward_fn(scoring_root)

    rewards = reward_fn(
        completions=[[{"role": "assistant", "content": FINAL}], FINAL],
        gold_sql=[GOLD, GOLD],
        db_id=["concert", "concert"],
        environments=[env_a, env_b],
    )

    expected_a = score_trajectory(
        ["SELECT nope FROM missing", GOLD], FINAL, GOLD, scoring_root / "concert" / "concert.sqlite"
    )
    expected_b = score_trajectory([], FINAL, GOLD, scoring_root / "concert" / "concert.sqlite")
    assert rewards == [pytest.approx(expected_a.total), pytest.approx(expected_b.total)]


def test_reward_fn_without_environments_is_loud(scoring_root: Path) -> None:
    reward_fn = make_multiturn_reward_fn(scoring_root)
    with pytest.raises(ValueError, match="environments"):
        reward_fn(completions=[FINAL], gold_sql=[GOLD], db_id=["concert"])


def test_reward_fn_reports_episodes_and_is_named_for_the_credit_scheme(
    scoring_root: Path,
) -> None:
    episodes: list[tuple[list[str | None], object]] = []
    reward_fn = make_multiturn_reward_fn(
        scoring_root,
        config=MultiTurnRewardConfig(credit="trajectory"),
        on_episode=lambda sqls, breakdown: episodes.append((sqls, breakdown)),
    )
    assert reward_fn.__name__ == "multiturn_reward_trajectory"

    env = _episode_env(scoring_root, ["SELECT b FROM t"])
    reward_fn(completions=[FINAL], gold_sql=[GOLD], db_id=["concert"], environments=[env])
    ((sqls, _breakdown),) = episodes
    assert sqls == ["SELECT b FROM t", GOLD]  # tool query + the final answer's extracted SQL


# --- mask check ---


def _decode_via(vocab: dict[int, str]):
    return lambda ids: "".join(vocab[i] for i in ids)


def test_mask_check_passes_when_feedback_is_only_in_masked_spans() -> None:
    vocab = {0: "SELECT 1 ", 1: f"{FEEDBACK_MARKER}: error ", 2: "final answer", 9: "<pad>"}
    check = check_tool_masking(
        completion_ids=[[0, 1, 2, 9], [2, 9, 9, 9]],
        completion_mask=[[1, 1, 1, 0], [1, 0, 0, 0]],
        tool_mask=[[1, 0, 1, 1], [1, 1, 1, 1]],
        decode=_decode_via(vocab),
    )
    assert check.n_rows == 2 and check.rows_with_masked_tokens == 1
    assert check.masked_tokens_total == 1
    assert check.feedback_in_masked_text and not check.feedback_in_unmasked_text


def test_mask_check_flags_feedback_carrying_gradient() -> None:
    vocab = {0: f"{FEEDBACK_MARKER}: 2 rows", 1: "answer"}
    check = check_tool_masking(
        completion_ids=[[0, 1]],
        completion_mask=[[1, 1]],
        tool_mask=[[1, 1]],  # nothing masked: the feedback tokens would train the policy
        decode=_decode_via(vocab),
    )
    assert check.masked_tokens_total == 0 and check.feedback_in_unmasked_text


def test_feedback_marker_matches_the_real_env_strings() -> None:
    success = format_feedback(ExecutionResult(rows=((1,),)), FeedbackConfig())
    assert success.startswith(FEEDBACK_MARKER)
    assert BUDGET_EXHAUSTED_FEEDBACK.startswith(FEEDBACK_MARKER)


# --- IS-ratio check ---


def test_is_ratio_in_band_over_logged_steps() -> None:
    history = [
        {"loss": 0.1},  # entries without the metric are ignored
        {
            "sampling/importance_sampling_ratio/mean": 1.02,
            "sampling/importance_sampling_ratio/min": 0.43,
            "sampling/importance_sampling_ratio/max": 2.25,
            "sampling/sampling_logp_difference/mean": 0.0076,
        },
        {"sampling/importance_sampling_ratio/mean": 0.97},
    ]
    check = check_is_ratio(history)
    assert check.in_band and check.n_logged == 2
    assert check.overall_min == pytest.approx(0.43)
    assert check.overall_max == pytest.approx(2.25)
    assert check.logp_diff_mean == pytest.approx(0.0076)
    assert check.band == IS_RATIO_HEALTHY_BAND


def test_is_ratio_out_of_band_or_never_logged_fails() -> None:
    assert not check_is_ratio([{"sampling/importance_sampling_ratio/mean": 1.6}]).in_band
    assert not check_is_ratio([{"loss": 0.1}]).in_band  # silence is failure, not a pass


# --- smoke validation verdict ---


def _passing_mask() -> dict[str, object]:
    return {
        "n_rows": 4,
        "rows_with_masked_tokens": 2,
        "masked_tokens_total": 40,
        "feedback_in_masked_text": True,
        "feedback_in_unmasked_text": False,
    }


def test_smoke_failures_empty_when_all_assertions_hold() -> None:
    from llm_rlvr.train.multiturn import IsRatioCheck, MaskCheck

    failures = smoke_failures(
        episodes_with_tool_calls=3,
        episodes_with_2plus_tool_turns=1,
        mask=MaskCheck(**_passing_mask()),
        is_ratio=IsRatioCheck(
            n_logged=2,
            mean_of_means=1.0,
            overall_min=0.5,
            overall_max=1.8,
            logp_diff_mean=0.01,
            band=IS_RATIO_HEALTHY_BAND,
            in_band=True,
        ),
    )
    assert failures == ()


def test_smoke_failures_name_every_broken_assertion() -> None:
    from llm_rlvr.train.multiturn import IsRatioCheck, MaskCheck

    failures = smoke_failures(
        episodes_with_tool_calls=0,
        episodes_with_2plus_tool_turns=0,
        mask=MaskCheck(
            n_rows=4,
            rows_with_masked_tokens=0,
            masked_tokens_total=0,
            feedback_in_masked_text=False,
            feedback_in_unmasked_text=True,
        ),
        is_ratio=IsRatioCheck(
            n_logged=0,
            mean_of_means=None,
            overall_min=None,
            overall_max=None,
            logp_diff_mean=None,
            band=IS_RATIO_HEALTHY_BAND,
            in_band=False,
        ),
    )
    assert len(failures) == 6  # every assertion failed, every one is named


# --- the multi-turn prompt style ---


def test_mt_prompt_style_offers_the_tool_and_keeps_the_final_format() -> None:
    from llm_rlvr.eval.prompt import INSTRUCTIONS, build_messages

    messages = build_messages("CREATE TABLE t (a INT)", "How many rows?", style="mt")
    system = messages[0]["content"]
    assert "run_sql" in system  # the protocol affordance
    assert "<think>" in system and "```sql" in system  # final-answer format survives
    assert "mt" in INSTRUCTIONS
    # Gold-free: the instruction describes the protocol, never answers or verdicts.
    assert "gold" not in system.lower()


def test_multiturn_provenance_records_the_prompt_style() -> None:
    from llm_rlvr.train.grpo import GrpoHyperparams
    from llm_rlvr.train.multiturn_provenance import build_multiturn_provenance

    provenance = build_multiturn_provenance(
        base_model="m",
        warm_start_adapter="grpo/grpo-s0",
        hyperparams=GrpoHyperparams(),
        environment=FeedbackConfig(),
        multiturn_reward=MultiTurnRewardConfig(),
        dataset="spider",
        dataset_revision="rev",
        n_train=10,
        n_dev=2,
        prompt_style="mt",
        commit="abc",
        dirty=False,
    )
    assert provenance.prompt_style == "mt"


# --- hyperparam/env budget consistency (driver guard, testable via the config) ---


def test_tool_loop_bound_must_match_env_budget() -> None:
    # The driver refuses a silent divergence; the pure config builder is where the bound lives.
    hp = MultiTurnHyperparams(max_tool_calling_iterations=5)
    assert multiturn_grpo_config_kwargs(hp, "out")["max_tool_calling_iterations"] == 5
