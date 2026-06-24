"""reward-functions: execution-grounded reward, both arms, empty guard, preamble penalty."""

from __future__ import annotations

from pathlib import Path

from llm_rlvr_sql.rewards import RewardConfig, make_reward_fn, score_completion

# A completion led by <think> then one sql fence (the clean, cheap target format).
GOLD = "SELECT a FROM t"


def _wrap(sql: str, *, preamble: str = "", think: str = "count the rows") -> str:
    return f"{preamble}<think>\n{think}\n</think>\n\n```sql\n{sql}\n```"


def test_result_match_earns_full_match_credit(tiny_db: Path) -> None:
    b = score_completion(_wrap("SELECT a FROM t"), GOLD, tiny_db)
    assert b.result_match == 1.0
    assert b.executable == 1.0
    assert b.reason == "match"


def test_executable_but_wrong_earns_executable_not_match(tiny_db: Path) -> None:
    b = score_completion(_wrap("SELECT b FROM t"), GOLD, tiny_db)
    assert b.executable == 1.0
    assert b.result_match == 0.0
    assert b.reason == "denotation_mismatch"


def test_non_executing_sql_earns_nothing(tiny_db: Path) -> None:
    b = score_completion(_wrap("SELECT nope FROM missing_table"), GOLD, tiny_db)
    assert b.executable == 0.0
    assert b.result_match == 0.0
    assert b.reason == "pred_error"


def test_no_sql_is_zero_not_an_error(tiny_db: Path) -> None:
    b = score_completion("<think>no fence here</think>", GOLD, tiny_db)
    assert b.executable == 0.0
    assert b.result_match == 0.0
    assert b.reason == "no_sql"


def test_empty_pred_against_nonempty_gold_is_mismatch(tiny_db: Path) -> None:
    # gold returns rows; an empty prediction is a plain denotation mismatch (not the guard).
    b = score_completion(_wrap("SELECT a FROM t WHERE a=999"), GOLD, tiny_db)
    assert b.result_match == 0.0
    assert b.reason == "denotation_mismatch"


def test_empty_guard_blocks_empty_matching_empty(tiny_db: Path) -> None:
    # both genuinely empty: official semantics would match, but the training guard blocks the
    # always-empty reward hack.
    b = score_completion(_wrap("SELECT a FROM t WHERE a=88"), "SELECT a FROM t WHERE a=99", tiny_db)
    assert b.result_match == 0.0
    assert b.reason == "trivial_empty_guard"


def test_empty_guard_can_be_disabled(tiny_db: Path) -> None:
    cfg = RewardConfig(guard_trivial_empty=False)
    # gold genuinely empty + pred empty -> official semantics match when the guard is off.
    b = score_completion(
        _wrap("SELECT a FROM t WHERE a=88"), "SELECT a FROM t WHERE a=99", tiny_db, config=cfg
    )
    assert b.result_match == 1.0


def test_format_full_credit_only_when_think_leads(tiny_db: Path) -> None:
    led = score_completion(_wrap("SELECT a FROM t"), GOLD, tiny_db)
    preambled = score_completion(
        _wrap("SELECT a FROM t", preamble="To answer this, we count the rows. "), GOLD, tiny_db
    )
    assert led.format == 1.0
    assert preambled.format == RewardConfig().preamble_credit
    assert preambled.format < led.format
    # the preamble penalty must actually lower the shaped total, all else equal.
    assert preambled.total < led.total


def test_malformed_format_is_zero(tiny_db: Path) -> None:
    # two fences -> not well-formed -> zero format credit (the SQL still scores on execution).
    text = "<think>x</think>\n```sql\nSELECT a FROM t\n```\n```sql\nSELECT b FROM t\n```"
    b = score_completion(text, GOLD, tiny_db)
    assert b.format == 0.0


def test_near_binary_arm_tiers(tiny_db: Path) -> None:
    cfg = RewardConfig(arm="near_binary")
    exact = score_completion(_wrap("SELECT a FROM t"), GOLD, tiny_db, config=cfg)
    wrong = score_completion(_wrap("SELECT b FROM t"), GOLD, tiny_db, config=cfg)
    broken = score_completion(_wrap("SELECT x FROM nope"), GOLD, tiny_db, config=cfg)
    assert exact.total == 1.0
    assert wrong.total == 0.1
    assert broken.total == 0.0


def test_shaped_arm_weights_sum(tiny_db: Path) -> None:
    cfg = RewardConfig(arm="shaped")
    b = score_completion(_wrap("SELECT a FROM t"), GOLD, tiny_db, config=cfg)
    expected = (
        cfg.w_format * b.format
        + cfg.w_executable * b.executable
        + cfg.w_result_match * b.result_match
        + cfg.w_length * b.length
    )
    assert abs(b.total - expected) < 1e-9


def test_determinism(tiny_db: Path) -> None:
    a = score_completion(_wrap("SELECT a FROM t"), GOLD, tiny_db)
    b = score_completion(_wrap("SELECT a FROM t"), GOLD, tiny_db)
    assert a.model_dump() == b.model_dump()


def test_reward_never_mutates_db(tiny_db: Path) -> None:
    before = tiny_db.read_bytes()
    score_completion(_wrap("DELETE FROM t"), GOLD, tiny_db)
    score_completion(_wrap("DROP TABLE t"), GOLD, tiny_db)
    assert tiny_db.read_bytes() == before


def test_trl_reward_fn_batches_over_completions(tiny_db: Path) -> None:
    # the TRL closure resolves db per example; here every db_id maps to tiny_db.
    fn = make_reward_fn(tiny_db.parent, resolver=lambda _dbid, _root: tiny_db)
    completions = [_wrap("SELECT a FROM t"), _wrap("SELECT b FROM t")]
    rewards = fn(completions=completions, gold_sql=[GOLD, GOLD], db_id=["t", "t"])
    assert len(rewards) == 2
    assert rewards[0] > rewards[1]  # exact match outscores executable-but-wrong


def test_trl_reward_fn_reads_assistant_message_lists(tiny_db: Path) -> None:
    fn = make_reward_fn(tiny_db.parent, resolver=lambda _dbid, _root: tiny_db)
    conversational = [[{"role": "assistant", "content": _wrap("SELECT a FROM t")}]]
    rewards = fn(completions=conversational, gold_sql=[GOLD], db_id=["t"])
    assert rewards[0] > 0.0
