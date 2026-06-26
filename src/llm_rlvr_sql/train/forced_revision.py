"""Forced-execution revision GRPO (Stage 3b, grpo-forced-revision task 2.1).

The archived grpo-multi-turn change died on tool *election*: the 3B policy never chooses to
call an offered tool (0/672 episodes, every checkpoint rung). This rollout removes election —
the harness executes every attempt unconditionally (SCoRe / ReEx-SQL shape):

    model writes SQL  →  sandbox executes it  →  deterministic gold-free feedback appended
    →  model revises  →  the final answer is scored.

Built on TRL v1.5.1's experimental `rollout_func` (the archived design's named D1 fallback),
verified against the tagged source:

  - `rollout_func(prompts, trainer)` receives the per-process batch (each prompt already
    duplicated `num_generations` times by RepeatSampler) and must return `prompt_ids`,
    `completion_ids`, and per-token sampling `logprobs` aligned with `completion_ids`.
  - An `env_mask` extra field (1 = model tokens, 0 = external tokens) is consumed by the
    trainer as `tool_mask`: the loss mask becomes `completion_mask * tool_mask` and the
    IS-ratio metrics are computed under the same combined mask — so the spliced feedback
    tokens carry no gradient and their placeholder logprobs never touch the #4543 check.
  - All other extra fields are forwarded per-row to the reward functions, which is how the
    attempt/revision texts reach the trajectory reward without re-parsing spliced text.

The feedback segment's token ids are produced by chat-template differencing over a probe
conversation — the same technique TRL's native tool loop uses (`_get_tool_suffix_ids`), with
the same disclosed artifact: the spliced sequence omits the single newline the template would
put between the assistant's EOS and the next user header. The smoke's mask + IS assertions
validate the splice on real rollouts before any paid run.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from llm_rlvr_sql._base import Frozen
from llm_rlvr_sql.data.models import DatabaseSchema, Example
from llm_rlvr_sql.data.spider import resolve_database
from llm_rlvr_sql.env.sandbox import SqliteSandbox
from llm_rlvr_sql.env.text2sql_env import (
    AGENT_ROW_CAP,
    AGENT_TIMEOUT_S,
    FeedbackConfig,
    format_feedback,
)
from llm_rlvr_sql.eval.completions import write_json
from llm_rlvr_sql.eval.parsing import extract_sql
from llm_rlvr_sql.rewards.multiturn import (
    MultiTurnRewardConfig,
    TrajectoryBreakdown,
    score_trajectory,
)
from llm_rlvr_sql.train.diagnostics import CollapseDiagnostics, compute_diagnostics
from llm_rlvr_sql.train.grpo import (
    GrpoHyperparams,
    GrpoProvenance,
    build_grpo_config_from_kwargs,
    build_grpo_provenance,
    build_grpo_rows,
    grpo_config_kwargs,
    split_train_dev,
)
from llm_rlvr_sql.train.multiturn import (
    IsRatioCheck,
    MaskCheck,
    check_is_ratio,
    check_tool_masking,
)

SMOKE_REPORT_FILENAME = "smoke-report.json"
DIAGNOSTICS_FILENAME = "d8-diagnostics.json"
PROVENANCE_FILENAME = "provenance.json"

# Windowed D8 diagnostics cadence: one CollapseDiagnostics snapshot per this many episodes.
DIAGNOSTICS_WINDOW = 128

# Gold-free feedback for an attempt with no extractable SQL — starts with the marker so the
# mask check treats it like every other observation (a unit test pins the prefix).
UNPARSEABLE_FEEDBACK = (
    "Execution feedback: error (parse): no SQL code block found in your answer. "
    "Write a single SQLite query in a ```sql ... ``` code block."
)

# The revision turn's instruction — the operative sentences of the probe's REVISE_INSTRUCTION,
# rephrased for the in-context continuation (the previous query is the assistant turn directly
# above, so it is not restated). Gold-free; same final-answer format as Stage 2.
REVISE_INSTRUCTION = (
    "Above is the execution feedback for your query. If the query errored or its result does "
    "not answer the question, revise it; if it is already correct, you may keep it. Reason "
    "step by step inside <think></think>, then give the final query in a ```sql ... ``` "
    "code block."
)


def revision_user_message(feedback: str) -> str:
    """The user turn appended after the attempt: the execution feedback + the revise
    instruction. Shared by the training rollout and the eval driver so they cannot drift."""
    return f"{feedback}\n\n{REVISE_INSTRUCTION}"


def attempt_feedback(
    attempt_text: str,
    db_path: str | Path,
    config: FeedbackConfig | None = None,
) -> str:
    """Execute the attempt's SQL in the read-only sandbox and render the model-facing feedback.

    Deterministic and gold-free; uses the agent-facing caps (the same ones the probe used), so
    training, eval, and the probe all show the model byte-identical observations.
    """
    sql = extract_sql(attempt_text)
    if sql is None:
        return UNPARSEABLE_FEEDBACK
    sandbox = SqliteSandbox(db_path, timeout_s=AGENT_TIMEOUT_S, row_cap=AGENT_ROW_CAP)
    return format_feedback(sandbox.execute(sql), config or FeedbackConfig())


# The probe conversation for suffix differencing: rendering it with and without the appended
# feedback turn isolates the token ids of "<assistant turn ends> user: <feedback> <generation
# prompt>". Any default system prompt the template injects cancels out in the difference.
_SUFFIX_PROBE_CONVERSATION = (
    {"role": "user", "content": "d"},
    {"role": "assistant", "content": "d"},
)


def build_suffix_ids(apply_chat_template: Callable[..., list[int]], user_content: str) -> list[int]:
    """Token ids for the spliced feedback segment, via chat-template differencing.

    `apply_chat_template` is the tokenizer's method (tests pass a fake). Raises if the
    no-feedback render is not a strict prefix of the with-feedback render — a template where
    differencing is unsound must fail loudly, not corrupt the mask.
    """
    probe_conversation = [dict(m) for m in _SUFFIX_PROBE_CONVERSATION]
    # return_dict=False: transformers v5 defaults apply_chat_template to a dict return; we
    # need the bare id list (TRL's own suffix builder passes the same flag).
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
    """Assemble one episode's (completion_ids, env_mask, logprobs), all index-aligned.

    Mirrors TRL's native tool loop: model segments keep their sampling logprobs and mask 1;
    the external feedback segment gets placeholder 0.0 logprobs and mask 0 (excluded from the
    loss and from the IS metrics by `completion_mask * tool_mask`).
    """
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
    """vLLM returns per-token top-k logprob lists (sampled token first); keep the top-1 floats.

    NaN logprobs arrive as None (TRL's extract_logprobs); they are replaced with 0.0 and
    counted — the smoke asserts the count is zero before any paid run trusts the IS ratios.
    """
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
    """The rollout's db_id lookup key: the target user message content (unique per example —
    it embeds the serialized schema and the question)."""
    return str(prompt[-1]["content"])


def build_db_id_lookup(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    """db_id keyed by prompt content, from `build_grpo_rows` output. The trainer hands the
    rollout only (prompt, trainer), so the database binding must travel by content."""
    return {prompt_lookup_key(row["prompt"]): str(row["db_id"]) for row in rows}


def make_forced_revision_rollout(
    db_root: str | Path,
    db_id_by_prompt: Mapping[str, str],
    *,
    feedback: FeedbackConfig | None = None,
    resolver: Callable[[str, str | Path], Path] = resolve_database,
) -> Callable[..., dict[str, object]]:
    """The hand-written rollout_func: attempt → execute → feedback → revise, every episode.

    Generates through `trainer.vllm_generation` (weights are pre-synced by the trainer for
    custom rollouts; colocate generates one completion per already-duplicated prompt entry).
    Returns the TRL contract dict plus `env_mask` and the `fr_*` extra fields the reward and
    diagnostics consume.
    """
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
            # Per-row so TRL's per-row extra-field merge keeps it aligned; the batch total is
            # what the smoke asserts on (sum over rows).
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
    """Score forced-revision episodes with the shipped trajectory reward.

    The episode's executed query is the attempt's extracted SQL (always executed by the
    harness); the final answer is the revision text. The `fr_*` kwargs arrive via the
    rollout's extra fields — their absence means the trainer was built without the
    forced-revision rollout, which is a wiring bug, not a soft default.
    """
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


def count_nan_log_values(log_history: Sequence[Mapping[str, object]]) -> int:
    """NaN losses/grad-norms in the trainer log — the smoke's no-NaN assertion."""
    count = 0
    for entry in log_history:
        for key in ("loss", "grad_norm"):
            value = entry.get(key)
            if isinstance(value, float) and math.isnan(value):
                count += 1
    return count


def fr_smoke_failures(
    *,
    n_episodes: int,
    mask: MaskCheck,
    is_ratio: IsRatioCheck,
    nan_log_values: int,
    none_logprob_tokens: int,
) -> tuple[str, ...]:
    """The de-risk verdict, pure: which assertions failed (empty = paid runs may proceed).

    Forced revision has no election step, so unlike the archived tool-loop smoke, EVERY
    rollout row must carry a masked feedback segment.
    """
    failures: list[str] = []
    if n_episodes == 0:
        failures.append("no episodes were scored — the rollout never reached the reward")
    if mask.rows_with_masked_tokens < mask.n_rows:
        failures.append(
            f"only {mask.rows_with_masked_tokens}/{mask.n_rows} rollout rows carry masked "
            "tokens — forced revision must splice feedback into every episode"
        )
    if mask.masked_tokens_total == 0:
        failures.append("no completion token was feedback-masked")
    if not mask.feedback_in_masked_text:
        failures.append("env feedback never appeared in the masked (loss-excluded) spans")
    if mask.feedback_in_unmasked_text:
        failures.append("env feedback leaked into gradient-carrying tokens")
    if not is_ratio.in_band:
        failures.append(
            f"importance-sampling ratio outside the Stage-2 band {is_ratio.band}: "
            f"means avg {is_ratio.mean_of_means}, n_logged {is_ratio.n_logged}"
        )
    if nan_log_values > 0:
        failures.append(f"{nan_log_values} NaN loss/grad_norm values in the trainer log")
    if none_logprob_tokens > 0:
        failures.append(
            f"{none_logprob_tokens} sampled tokens came back with NaN logprobs from vLLM"
        )
    return tuple(failures)


class ForcedRevisionProvenance(Frozen):
    """Everything needed to reproduce a forced-revision checkpoint: the Stage-2 GRPO record
    plus the pieces unique to this rollout (feedback config, trajectory reward, the revise
    instruction the policy was trained against)."""

    grpo: GrpoProvenance
    environment: FeedbackConfig
    multiturn_reward: MultiTurnRewardConfig
    rollout: str = "forced_revision"
    prompt_style: str = "cot"
    revise_instruction: str = REVISE_INSTRUCTION


class FrSmokeReport(Frozen):
    """The forced-revision smoke's full evidence: assertions, episode census, envelope."""

    passed: bool
    failures: tuple[str, ...]
    n_train: int
    n_steps: int
    wall_time_s: float
    mean_step_time_s: float | None
    n_episodes: int
    n_unparseable_attempts: int  # attempt had no extractable SQL (parse-feedback path)
    n_final_sql_missing: int  # revision had no extractable SQL
    mask: MaskCheck
    is_ratio: IsRatioCheck
    nan_log_values: int
    none_logprob_tokens: int
    diagnostics: CollapseDiagnostics | None
    mean_reward: float | None
    peak_torch_memory_gib: float | None
    device_memory_used_gib: float | None
    completions_mean_length: float | None
    provenance: ForcedRevisionProvenance


class DiagnosticsSeries(Frozen):
    """The D8 record persisted next to a trained adapter: windowed collapse diagnostics over
    the whole run (window = `DIAGNOSTICS_WINDOW` episodes, in order)."""

    window_episodes: int
    windows: tuple[CollapseDiagnostics, ...]
    overall: CollapseDiagnostics | None


def _logged_mean(log_history: Sequence[Mapping[str, object]], key: str) -> float | None:
    values = [float(entry[key]) for entry in log_history if key in entry]
    return (sum(values) / len(values)) if values else None


class _EpisodeLog:
    """Accumulates scored episodes; prints a D8 window every `DIAGNOSTICS_WINDOW` episodes so
    collapse is visible live in the Modal logs, not post-hoc."""

    def __init__(self) -> None:
        self.episodes: list[tuple[list[str | None], TrajectoryBreakdown]] = []
        self.windows: list[CollapseDiagnostics] = []

    def __call__(self, sqls: list[str | None], breakdown: TrajectoryBreakdown) -> None:
        self.episodes.append((sqls, breakdown))
        if len(self.episodes) % DIAGNOSTICS_WINDOW == 0:
            window = compute_diagnostics(self.episodes[-DIAGNOSTICS_WINDOW:])
            self.windows.append(window)
            print(f"[fr-d8] episodes={len(self.episodes)} {window.model_dump_json()}")

    def series(self) -> DiagnosticsSeries:
        return DiagnosticsSeries(
            window_episodes=DIAGNOSTICS_WINDOW,
            windows=tuple(self.windows),
            overall=compute_diagnostics(self.episodes) if self.episodes else None,
        )


def _build_trainer_pieces(
    examples: Sequence[Example],
    schemas: Mapping[str, DatabaseSchema],
    db_root: str | Path,
    *,
    hyperparams: GrpoHyperparams,
    environment: FeedbackConfig,
    reward_config: MultiTurnRewardConfig,
    on_episode: Callable[[list[str | None], TrajectoryBreakdown], None],
):
    """Shared smoke/train setup: split, dataset rows, rollout + reward fns. CPU-pure."""
    from datasets import Dataset  # lazy: heavy training stack, train extra only

    train_examples, dev_examples = split_train_dev(
        examples, dev_size=hyperparams.eval_slice_size, seed=hyperparams.seed
    )
    rows = build_grpo_rows(train_examples, schemas, style="cot")
    rollout = make_forced_revision_rollout(db_root, build_db_id_lookup(rows), feedback=environment)
    reward_fn = make_forced_revision_reward_fn(db_root, config=reward_config, on_episode=on_episode)
    return train_examples, dev_examples, Dataset.from_list(rows), rollout, reward_fn


def _load_policy(base_model: str, base_model_revision: str | None, warm_start_adapter: str | Path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype="bfloat16",
        **({"revision": base_model_revision} if base_model_revision else {}),
    )
    # Warm-start from the Stage-2 adapter as a *trainable* PEFT model; with beta != 0 TRL
    # snapshots it as the frozen "ref" adapter, so the KL anchors to the Stage-2 policy (D4).
    return PeftModel.from_pretrained(base, str(warm_start_adapter), is_trainable=True)


def run_forced_revision_smoke(
    examples: Sequence[Example],
    schemas: Mapping[str, DatabaseSchema],
    db_root: str | Path,
    output_dir: str | Path,
    *,
    base_model: str,
    warm_start_adapter: str | Path,
    dataset: str,
    dataset_revision: str,
    base_model_revision: str | None = None,
    hyperparams: GrpoHyperparams | None = None,
    environment: FeedbackConfig | None = None,
    multiturn_reward: MultiTurnRewardConfig | None = None,
    max_steps: int | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> Path:
    """The de-risk smoke: a few real forced-revision GRPO steps, then the mask + IS-ratio +
    no-NaN assertions. Raises AFTER persisting the report — the evidence survives the exit."""
    import torch
    from trl import GRPOTrainer

    hp = hyperparams or GrpoHyperparams()
    env_cfg = environment or FeedbackConfig()
    reward_cfg = multiturn_reward or MultiTurnRewardConfig()

    log = _EpisodeLog()
    train_examples, dev_examples, train_dataset, rollout, reward_fn = _build_trainer_pieces(
        examples,
        schemas,
        db_root,
        hyperparams=hp,
        environment=env_cfg,
        reward_config=reward_cfg,
        on_episode=log,
    )

    captured: list[dict[str, list[list[int]]]] = []
    none_logprob_total = 0

    def counting_rollout(prompts, trainer):
        nonlocal none_logprob_total
        output = rollout(prompts, trainer)
        none_logprob_total += sum(output["fr_none_logprob_tokens"])
        return output

    class _CapturingTrainer(GRPOTrainer):
        """Stock trainer + a tap on the scored rollout batches (the mask evidence)."""

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            if "tool_mask" in output:
                captured.append(
                    {
                        key: output[key].detach().cpu().tolist()
                        for key in ("completion_ids", "completion_mask", "tool_mask")
                    }
                )
            return output

    desired = grpo_config_kwargs(hp, output_dir)
    if max_steps is not None:
        desired["max_steps"] = max_steps

    policy = _load_policy(base_model, base_model_revision, warm_start_adapter)
    torch.cuda.reset_peak_memory_stats()
    trainer = _CapturingTrainer(
        model=policy,
        reward_funcs=[reward_fn],
        args=build_grpo_config_from_kwargs(desired),
        train_dataset=train_dataset,
        rollout_func=counting_rollout,
    )
    started = time.perf_counter()
    trainer.train()
    wall_time_s = time.perf_counter() - started

    tokenizer = trainer.processing_class

    def decode(ids: Sequence[int]) -> str:
        return tokenizer.decode(list(ids), skip_special_tokens=True)

    mask = check_tool_masking(
        [row for batch in captured for row in batch["completion_ids"]],
        [row for batch in captured for row in batch["completion_mask"]],
        [row for batch in captured for row in batch["tool_mask"]],
        decode,
    )
    log_history = trainer.state.log_history
    is_ratio = check_is_ratio(log_history)
    nan_log_values = count_nan_log_values(log_history)

    failures = fr_smoke_failures(
        n_episodes=len(log.episodes),
        mask=mask,
        is_ratio=is_ratio,
        nan_log_values=nan_log_values,
        none_logprob_tokens=none_logprob_total,
    )

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    report = FrSmokeReport(
        passed=not failures,
        failures=failures,
        n_train=len(train_examples),
        n_steps=int(trainer.state.global_step),
        wall_time_s=wall_time_s,
        mean_step_time_s=_logged_mean(log_history, "step_time"),
        n_episodes=len(log.episodes),
        n_unparseable_attempts=sum(1 for sqls, _ in log.episodes if sqls[0] is None),
        n_final_sql_missing=sum(1 for sqls, _ in log.episodes if sqls[-1] is None),
        mask=mask,
        is_ratio=is_ratio,
        nan_log_values=nan_log_values,
        none_logprob_tokens=none_logprob_total,
        diagnostics=compute_diagnostics(log.episodes) if log.episodes else None,
        mean_reward=_logged_mean(log_history, "reward"),
        peak_torch_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
        device_memory_used_gib=(total_bytes - free_bytes) / 2**30,
        completions_mean_length=_logged_mean(log_history, "completions/mean_length"),
        provenance=build_forced_revision_provenance(
            base_model=base_model,
            warm_start_adapter=warm_start_adapter,
            hyperparams=hp,
            environment=env_cfg,
            multiturn_reward=reward_cfg,
            dataset=dataset,
            dataset_revision=dataset_revision,
            n_train=len(train_examples),
            n_dev=len(dev_examples),
            base_model_revision=base_model_revision,
            commit=commit,
            dirty=dirty,
        ),
    )
    out = write_json(report, Path(output_dir) / SMOKE_REPORT_FILENAME)
    if failures:
        raise RuntimeError(
            "forced-revision smoke FAILED — paid runs are blocked: "
            + "; ".join(failures)
            + f" (report: {out})"
        )
    return out


def build_forced_revision_provenance(
    *,
    base_model: str,
    warm_start_adapter: str | Path,
    hyperparams: GrpoHyperparams,
    environment: FeedbackConfig,
    multiturn_reward: MultiTurnRewardConfig,
    dataset: str,
    dataset_revision: str,
    n_train: int,
    n_dev: int,
    base_model_revision: str | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> ForcedRevisionProvenance:
    return ForcedRevisionProvenance(
        grpo=build_grpo_provenance(
            base_model=base_model,
            warm_start_adapter=warm_start_adapter,
            hyperparams=hyperparams,
            reward=multiturn_reward.base,
            dataset=dataset,
            dataset_revision=dataset_revision,
            n_train=n_train,
            n_dev=n_dev,
            base_model_revision=base_model_revision,
            commit=commit,
            dirty=dirty,
        ),
        environment=environment,
        multiturn_reward=multiturn_reward,
    )


def run_forced_revision_train(
    examples: Sequence[Example],
    schemas: Mapping[str, DatabaseSchema],
    db_root: str | Path,
    output_dir: str | Path,
    *,
    base_model: str,
    warm_start_adapter: str | Path,
    dataset: str,
    dataset_revision: str,
    base_model_revision: str | None = None,
    hyperparams: GrpoHyperparams | None = None,
    environment: FeedbackConfig | None = None,
    multiturn_reward: MultiTurnRewardConfig | None = None,
    max_steps: int | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> Path:
    """Train the forced-revision adapter (pilot or full run, governed by `max_steps`).

    Persists the adapter, provenance, and the full D8 diagnostics series (windowed collapse
    metrics) under `output_dir`.
    """
    import torch
    from trl import GRPOTrainer

    hp = hyperparams or GrpoHyperparams()
    env_cfg = environment or FeedbackConfig()
    reward_cfg = multiturn_reward or MultiTurnRewardConfig()

    log = _EpisodeLog()
    train_examples, dev_examples, train_dataset, rollout, reward_fn = _build_trainer_pieces(
        examples,
        schemas,
        db_root,
        hyperparams=hp,
        environment=env_cfg,
        reward_config=reward_cfg,
        on_episode=log,
    )

    desired = grpo_config_kwargs(hp, output_dir)
    if max_steps is not None:
        desired["max_steps"] = max_steps

    policy = _load_policy(base_model, base_model_revision, warm_start_adapter)
    torch.cuda.reset_peak_memory_stats()
    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=[reward_fn],
        args=build_grpo_config_from_kwargs(desired),
        train_dataset=train_dataset,
        rollout_func=rollout,
    )
    trainer.train()
    trainer.save_model(str(output_dir))

    write_json(log.series(), Path(output_dir) / DIAGNOSTICS_FILENAME)
    provenance = build_forced_revision_provenance(
        base_model=base_model,
        warm_start_adapter=warm_start_adapter,
        hyperparams=hp,
        environment=env_cfg,
        multiturn_reward=reward_cfg,
        dataset=dataset,
        dataset_revision=dataset_revision,
        n_train=len(train_examples),
        n_dev=len(dev_examples),
        base_model_revision=base_model_revision,
        commit=commit,
        dirty=dirty,
    )
    write_json(provenance, Path(output_dir) / PROVENANCE_FILENAME)
    return Path(output_dir)
