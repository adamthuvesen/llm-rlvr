"""Forced-revision GRPO rollout: attempt → execute → feedback → revise (Stage 3b)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from llm_rlvr_sql.data.paths import resolve_database
from llm_rlvr_sql.env.text2sql_env import FeedbackConfig
from llm_rlvr_sql.eval.parsing import extract_sql
from llm_rlvr_sql.revision.feedback import attempt_feedback, revision_user_message
from llm_rlvr_sql.rewards.multiturn import (
    MultiTurnRewardConfig,
    TrajectoryBreakdown,
    score_trajectory,
)

_SUFFIX_PROBE_CONVERSATION = (
    {"role": "user", "content": "d"},
    {"role": "assistant", "content": "d"},
)


def build_suffix_ids(apply_chat_template: Callable[..., list[int]], user_content: str) -> list[int]:
    """Token ids for the spliced feedback segment, via chat-template differencing."""
    probe_conversation = [dict(m) for m in _SUFFIX_PROBE_CONVERSATION]
    prefix = apply_chat_template(
        probe_conversation, add_generation_prompt=False, tokenize=True, return_dict=False
    )
    full = apply_chat_template(
        [*probe_conversation, {"role": "user", "content": user_content}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    if full[: len(prefix)] != prefix:
        raise ValueError(
            "chat-template differencing is unsound for this tokenizer: the bare conversation "
            "is not a prefix of the conversation with the feedback turn appended"
        )
    return full[len(prefix) :]


def splice_episode(
    attempt_ids: Sequence[int],
    attempt_logprobs: Sequence[float],
    suffix_ids: Sequence[int],
    revision_ids: Sequence[int],
    revision_logprobs: Sequence[float],
) -> tuple[list[int], list[int], list[float]]:
    """Assemble one episode's (completion_ids, env_mask, logprobs), all index-aligned."""
    if len(attempt_ids) != len(attempt_logprobs):
        raise ValueError(
            f"attempt ids/logprobs misaligned: {len(attempt_ids)} vs {len(attempt_logprobs)}"
        )
    if len(revision_ids) != len(revision_logprobs):
        raise ValueError(
            f"revision ids/logprobs misaligned: {len(revision_ids)} vs {len(revision_logprobs)}"
        )
    completion_ids = [*attempt_ids, *suffix_ids, *revision_ids]
    env_mask = [1] * len(attempt_ids) + [0] * len(suffix_ids) + [1] * len(revision_ids)
    logprobs = [*attempt_logprobs, *([0.0] * len(suffix_ids)), *revision_logprobs]
    return completion_ids, env_mask, logprobs


def _top1_logprobs(
    seqs: Sequence[Sequence[Sequence[float | None]]],
) -> tuple[list[list[float]], int]:
    n_none = 0
    flat: list[list[float]] = []
    for seq in seqs:
        row: list[float] = []
        for token_logprobs in seq:
            top1 = token_logprobs[0]
            if top1 is None:
                n_none += 1
                top1 = 0.0
            row.append(top1)
        flat.append(row)
    return flat, n_none


def prompt_lookup_key(prompt: Sequence[Mapping[str, object]]) -> str:
    return str(prompt[-1]["content"])


def build_db_id_lookup(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    return {prompt_lookup_key(row["prompt"]): str(row["db_id"]) for row in rows}


def make_forced_revision_rollout(
    db_root: str | Path,
    db_id_by_prompt: Mapping[str, str],
    *,
    feedback: FeedbackConfig | None = None,
    resolver: Callable[[str, str | Path], Path] = resolve_database,
) -> Callable[..., dict[str, object]]:
    """Hand-written rollout_func: attempt → execute → feedback → revise, every episode."""
    fb_config = feedback or FeedbackConfig()

    def rollout(prompts: list, trainer: object) -> dict[str, object]:
        tokenizer = trainer.processing_class
        prompt_ids = [
            list(
                tokenizer.apply_chat_template(
                    list(prompt), add_generation_prompt=True, tokenize=True, return_dict=False
                )
            )
            for prompt in prompts
        ]

        _, attempt_ids, attempt_lps, _ = trainer.vllm_generation.generate(
            prompts=prompt_ids, images=None, num_generations=1
        )
        if attempt_lps is None:
            raise RuntimeError(
                "rollout needs per-token sampling logprobs; "
                "vllm_importance_sampling_correction must stay enabled"
            )
        attempt_ids = [list(ids) for ids in attempt_ids]
        attempt_lps, n_none_attempt = _top1_logprobs(attempt_lps)
        attempt_texts = tokenizer.batch_decode(attempt_ids, skip_special_tokens=True)

        feedbacks: list[str] = []
        for prompt, text in zip(prompts, attempt_texts, strict=True):
            key = prompt_lookup_key(prompt)
            if key not in db_id_by_prompt:
                raise KeyError(
                    "no db_id recorded for a rollout prompt — the lookup must be built from "
                    "the same rows as the train dataset"
                )
            db_path = resolver(db_id_by_prompt[key], db_root)
            feedbacks.append(attempt_feedback(text, db_path, fb_config))

        suffix_ids = [
            build_suffix_ids(tokenizer.apply_chat_template, revision_user_message(feedback))
            for feedback in feedbacks
        ]
        revision_prompt_ids = [
            [*p, *a, *s] for p, a, s in zip(prompt_ids, attempt_ids, suffix_ids, strict=True)
        ]
        _, revision_ids, revision_lps, _ = trainer.vllm_generation.generate(
            prompts=revision_prompt_ids, images=None, num_generations=1
        )
        revision_ids = [list(ids) for ids in revision_ids]
        revision_lps, n_none_revision = _top1_logprobs(revision_lps)
        revision_texts = tokenizer.batch_decode(revision_ids, skip_special_tokens=True)

        completion_ids: list[list[int]] = []
        env_mask: list[list[int]] = []
        logprobs: list[list[float]] = []
        for a_ids, a_lps, s_ids, r_ids, r_lps in zip(
            attempt_ids, attempt_lps, suffix_ids, revision_ids, revision_lps, strict=True
        ):
            ids, mask, lps = splice_episode(a_ids, a_lps, s_ids, r_ids, r_lps)
            completion_ids.append(ids)
            env_mask.append(mask)
            logprobs.append(lps)

        n_none = n_none_attempt + n_none_revision
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "env_mask": env_mask,
            "fr_attempt_text": list(attempt_texts),
            "fr_feedback": feedbacks,
            "fr_revision_text": list(revision_texts),
            "fr_none_logprob_tokens": [n_none] + [0] * (len(prompts) - 1),
        }

    return rollout


def make_forced_revision_reward_fn(
    db_root: str | Path,
    *,
    config: MultiTurnRewardConfig | None = None,
    resolver: Callable[[str, str | Path], Path] = resolve_database,
    on_episode: Callable[[list[str | None], TrajectoryBreakdown], None] | None = None,
) -> Callable[..., list[float]]:
    cfg = config or MultiTurnRewardConfig()

    def reward(
        completions,
        gold_sql,
        db_id,
        fr_attempt_text=None,
        fr_revision_text=None,
        **_kwargs,
    ) -> list[float]:
        if fr_attempt_text is None or fr_revision_text is None:
            raise ValueError(
                "forced-revision reward needs fr_attempt_text/fr_revision_text; "
                "was the trainer built with make_forced_revision_rollout?"
            )
        rewards: list[float] = []
        for gold, dbid, attempt, revision in zip(
            gold_sql, db_id, fr_attempt_text, fr_revision_text, strict=True
        ):
            attempt_sql = extract_sql(attempt)
            queries = [attempt_sql] if attempt_sql is not None else []
            breakdown = score_trajectory(
                queries, revision, gold, resolver(dbid, db_root), config=cfg
            )
            if on_episode is not None:
                on_episode([attempt_sql, extract_sql(revision)], breakdown)
            rewards.append(breakdown.total)
        return rewards

    reward.__name__ = f"forced_revision_reward_{cfg.credit}"
    return reward
