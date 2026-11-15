"""The task-agnostic harness: a toy integer task plugs into the same shaping reward as SQL."""

from __future__ import annotations

import re

import pytest

from llm_rlvr.harness import (
    ReferenceUnverifiable,
    RewardConfig,
    Verdict,
    completion_text,
    score_answer,
)

_FENCE = re.compile(r"```\n?(.*?)```", re.DOTALL)


def _extract_integer(text: str) -> str | None:
    blocks = _FENCE.findall(text)
    return blocks[-1].strip() if blocks else None


def _integer_well_formed(text: str) -> bool:
    # One <think> block then one fence. The *leading*-vs-preamble distinction is the reward's
    # job (`_format_score`), not the parser's — same split as the SQL parser.
    has_think = "<think>" in text.lower() and "</think>" in text.lower()
    return has_think and len(_FENCE.findall(text)) == 1


def _verify_integer(answer: str, reference: object) -> Verdict:
    try:
        gold = int(str(reference))
    except ValueError as exc:
        raise ReferenceUnverifiable("non-integer reference", reason="ref_broken") from exc
    try:
        pred = int(answer)
    except ValueError:
        return Verdict(matched=False, valid=False, reason="not_an_integer")
    hit = pred == gold
    return Verdict(matched=hit, valid=True, reason="match" if hit else "wrong")


def _wrap(answer: str, *, preamble: str = "") -> str:
    return f"{preamble}<think>\nadd them up\n</think>\n\n```\n{answer}\n```"


def _score(text: str, reference: object, *, config: RewardConfig | None = None):
    return score_answer(
        text,
        extract=_extract_integer,
        well_formed=_integer_well_formed,
        verify=lambda answer: _verify_integer(answer, reference),
        config=config,
    )


def test_correct_integer_earns_full_match_credit() -> None:
    b = _score(_wrap("42"), 42)
    assert b.result_match == 1.0
    assert b.executable == 1.0
    assert b.reason == "match"


def test_wrong_but_parseable_earns_valid_not_match() -> None:
    b = _score(_wrap("41"), 42)
    assert b.executable == 1.0
    assert b.result_match == 0.0
    assert b.reason == "wrong"


def test_unparseable_answer_earns_nothing() -> None:
    b = _score(_wrap("forty-two"), 42)
    assert b.executable == 0.0
    assert b.result_match == 0.0
    assert b.reason == "not_an_integer"


def test_no_answer_is_zero_not_an_error() -> None:
    b = _score("<think>no fence here</think>", 42)
    assert b.executable == 0.0
    assert b.result_match == 0.0
    assert b.reason == "no_answer"


def test_broken_reference_surfaces_as_unverifiable_reason() -> None:
    b = _score(_wrap("42"), "not-a-number")
    assert b.result_match == 0.0
    assert b.executable == 0.0
    assert b.reason == "ref_broken"


def test_format_full_credit_only_when_think_leads() -> None:
    led = _score(_wrap("42"), 42)
    preambled = _score(_wrap("42", preamble="Let me think. "), 42)
    assert led.format == 1.0
    assert preambled.format == RewardConfig().preamble_credit
    assert preambled.total < led.total


def test_near_binary_arm_tiers_are_task_agnostic() -> None:
    cfg = RewardConfig(arm="near_binary")
    assert _score(_wrap("42"), 42, config=cfg).total == 1.0
    assert _score(_wrap("41"), 42, config=cfg).total == 0.1
    assert _score(_wrap("nope"), 42, config=cfg).total == 0.0


def test_completion_text_handles_str_dict_and_message_list() -> None:
    assert completion_text("hello") == "hello"
    assert completion_text({"role": "assistant", "content": "hi"}) == "hi"
    assert completion_text([{"role": "assistant", "content": "answer"}]) == "answer"


def test_sql_training_reward_aborts_on_unverifiable_reference(scoring_root) -> None:
    from llm_rlvr.tasks.sql import make_reward_fn
    from llm_rlvr.tasks.sql.verifier import GoldExecutionError

    reward = make_reward_fn(scoring_root)
    with pytest.raises(GoldExecutionError, match="reference became unverifiable"):
        reward(
            completions=["<think>x</think>\n```sql\nSELECT a FROM t\n```"],
            gold_sql=["SELECT bad FROM t"],
            db_id=["concert"],
        )


def test_sql_training_reward_requires_empty_guard() -> None:
    from llm_rlvr.tasks.sql import SqlRewardConfig, make_reward_fn

    with pytest.raises(ValueError, match="guard_trivial_empty=True"):
        make_reward_fn("unused", config=SqlRewardConfig(guard_trivial_empty=False))


@pytest.mark.parametrize(
    "completion",
    [
        None,
        42,
        b"answer",
        [],
        {"role": "assistant"},
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": ["answer"]},
        {"role": "user", "content": "question"},
        [{"role": "assistant", "content": 42}],
        [{"role": "user", "content": "question"}],
        ["answer"],
    ],
)
def test_completion_text_rejects_malformed_values(completion) -> None:
    with pytest.raises(ValueError, match="completion"):
        completion_text(completion)


def test_completion_text_preserves_last_assistant_text_in_valid_conversation() -> None:
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "follow-up"},
        {"role": "assistant", "content": "final answer"},
    ]
    assert completion_text(messages) == "final answer"


def test_matched_verdict_must_also_be_valid() -> None:
    with pytest.raises(ValueError, match="matched verdict must be valid"):
        Verdict(matched=True, valid=False, reason="match")
