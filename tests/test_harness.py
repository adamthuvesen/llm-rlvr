"""The task-agnostic harness: a non-SQL task plugs into the same shaping reward, and the SQL task
satisfies the harness protocols.

The toy "integer" task below is the whole point of the refactor: implement `AnswerParser` +
`Verifier`, and `score_answer` gives you shaped / near-binary RLVR rewards with no SQL in sight.
"""

from __future__ import annotations

import re

import pytest

from llm_rlvr.harness import (
    AnswerParser,
    ReferenceUnverifiable,
    RewardConfig,
    Task,
    Verdict,
    Verifier,
    completion_text,
    score_answer,
)
from llm_rlvr.tasks import get_task
from llm_rlvr.tasks.sql import SQL_PARSER, SqlTask, SqlVerifier, sql_task

# --- a tiny non-SQL task: "what is the integer answer?" -----------------------------------------

_FENCE = re.compile(r"```\n?(.*?)```", re.DOTALL)


class IntegerParser:
    """Extract the answer from a fenced block; well-formed = a leading <think> then one fence."""

    def extract(self, text: str) -> str | None:
        blocks = _FENCE.findall(text)
        return blocks[-1].strip() if blocks else None

    def well_formed(self, text: str) -> bool:
        # One <think> block then one fence. The *leading*-vs-preamble distinction is the reward's
        # job (`_format_score`), not the parser's — same split as the SQL parser.
        has_think = "<think>" in text.lower() and "</think>" in text.lower()
        return has_think and len(_FENCE.findall(text)) == 1


class IntegerVerifier:
    """reference = the correct integer (any int-coercible); context is unused.

    Raises `ReferenceUnverifiable` when the reference itself is not an integer — a broken example,
    not a wrong answer.
    """

    def verify(self, answer: str, reference: object, context: object) -> Verdict:
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


INT_PARSER = IntegerParser()
INT_VERIFIER = IntegerVerifier()


def _wrap(answer: str, *, preamble: str = "") -> str:
    return f"{preamble}<think>\nadd them up\n</think>\n\n```\n{answer}\n```"


def _score(text: str, reference: object, *, config: RewardConfig | None = None):
    return score_answer(
        text, reference, None, parser=INT_PARSER, verifier=INT_VERIFIER, config=config
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


# --- a non-SQL task flows through the GRPO construction seam (rows + reward) ---------------------


class IntegerTask:
    """A minimal non-SQL `Task` — enough to prove `run_grpo`'s row/reward seam is task-agnostic."""

    name = "integer"
    parser = INT_PARSER

    def make_verifier(self):
        return INT_VERIFIER

    def make_reward_fn(self, context_root, *, config=None):
        cfg = config or RewardConfig()

        def reward(completions, reference, **_kwargs):
            return [
                score_answer(
                    completion_text(c),
                    ref,
                    None,
                    parser=INT_PARSER,
                    verifier=INT_VERIFIER,
                    config=cfg,
                ).total
                for c, ref in zip(completions, reference, strict=True)
            ]

        return reward

    def build_prompts(self, examples, resources, *, style="cot", **_kwargs):
        return [[{"role": "user", "content": f"sum: {e}"}] for e in examples]

    def build_grpo_rows(self, examples, resources, *, style="cot", **_kwargs):
        prompts = self.build_prompts(examples, resources, style=style)
        return [{"prompt": p, "reference": e} for e, p in zip(examples, prompts, strict=True)]


def test_non_sql_task_flows_through_grpo_rows_and_reward() -> None:
    task = IntegerTask()
    assert isinstance(task, Task)

    # the exact two calls run_grpo makes: task.build_grpo_rows(...) then task.make_reward_fn(...)
    rows = task.build_grpo_rows([42, 7], {})
    assert [set(r) for r in rows] == [{"prompt", "reference"}, {"prompt", "reference"}]
    assert rows[0]["reference"] == 42

    reward_fn = task.make_reward_fn("/unused")
    rewards = reward_fn(
        completions=[_wrap("42"), _wrap("41")],
        reference=[r["reference"] for r in rows],
    )
    assert len(rewards) == 2
    assert rewards[0] > rewards[1]  # exact match outscores executable-but-wrong


# --- the SQL task satisfies the harness protocols -----------------------------------------------


def test_sql_pieces_satisfy_the_harness_protocols() -> None:
    assert isinstance(SQL_PARSER, AnswerParser)
    assert isinstance(SqlVerifier(), Verifier)
    assert isinstance(sql_task(), Task)


def test_task_registry_resolves_sql_and_rejects_unknown() -> None:
    assert isinstance(get_task("sql"), SqlTask)
    with pytest.raises(ValueError, match="unknown task"):
        get_task("does-not-exist")


# --- completion_text normalizes the TRL shapes --------------------------------------------------


def test_completion_text_handles_str_dict_and_message_list() -> None:
    assert completion_text("hello") == "hello"
    assert completion_text({"role": "assistant", "content": "hi"}) == "hi"
    assert completion_text([{"role": "assistant", "content": "answer"}]) == "answer"
