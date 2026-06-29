"""Forced-revision evaluation: the two-pass harness at inference time (no RL).

Mirrors the training rollout exactly — greedy attempt under the Stage-2 `cot` prompt, the
attempt's SQL executed unconditionally in the read-only sandbox, the deterministic gold-free
feedback + revise instruction appended as a user turn, one greedy revision — except that the
revision prompt is rendered canonically by the chat template (training splices token ids and
omits one inter-turn newline; a disclosed, sub-token-level delta).

The revision is the final answer: it goes into a normal `CompletionSet` (existing scorer and
`make eval` work unchanged; `backend="vllm-forced-revision"` disclosed in provenance), and the
full episodes are persisted as a `ForcedRevisionTranscriptSet` so fix/harm analysis re-derives
on CPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from llm_rlvr_sql._base import Frozen
from llm_rlvr_sql.data.models import DatabaseSchema, Example
from llm_rlvr_sql.data.paths import resolve_database
from llm_rlvr_sql.env.text2sql_env import FeedbackConfig
from llm_rlvr_sql.eval.completions import (
    CompletionSet,
    GenerationSample,
    ProblemCompletions,
)
from llm_rlvr_sql.eval.generate import build_prompts
from llm_rlvr_sql.eval.provenance import DecodingConfig
from llm_rlvr_sql.eval.runner import finish_eval_run
from llm_rlvr_sql.revision.episode import ForcedRevisionEpisode, ForcedRevisionTranscriptSet
from llm_rlvr_sql.revision.feedback import (
    attempt_feedback,
    is_error_feedback,
    revision_user_message,
)

ForcedRevisionPolicy = Literal["always", "error_only"]


class ErrorOnlyRevisionReport(Frozen):
    """CPU replay summary for the cheap error-only forced-revision policy."""

    n: int
    n_skipped_gold_error: int
    attempt_ex: float
    always_revise_ex: float
    error_only_ex: float
    n_revised: int
    revised_rate: float
    n_fixed: int
    n_harmed: int


def validate_forced_revision_policy(policy: str) -> ForcedRevisionPolicy:
    if policy not in {"always", "error_only"}:
        raise ValueError(f"policy must be 'always' or 'error_only', got {policy!r}")
    return policy  # type: ignore[return-value]


def should_request_revision(policy: ForcedRevisionPolicy, feedback: str) -> bool:
    """Apply the inference policy to one feedback turn."""
    if policy == "always":
        return True
    return is_error_feedback(feedback)


def forced_revision_artifact_stem(
    *,
    dataset: str,
    adapter: str | Path | None,
    limit: int | None,
    temperature: float,
    sampling_seed: int,
    policy: ForcedRevisionPolicy,
) -> str:
    """Collision-proof artifact stem for forced-revision eval outputs."""
    adapter_tag = f"-{Path(adapter).name}" if adapter else ""
    seed_tag = f"-t{temperature}-seed{sampling_seed}" if temperature > 0 else ""
    policy_tag = "fr" if policy == "always" else "fr-error-only"
    return f"{dataset}-dev-{policy_tag}{adapter_tag}{seed_tag}" + (f"-n{limit}" if limit else "")


def replay_error_only_revision(
    completion_set: CompletionSet,
    transcripts: ForcedRevisionTranscriptSet,
    db_root: str | Path,
) -> tuple[CompletionSet, ErrorOnlyRevisionReport]:
    """Replay committed always-revise transcripts under the cheaper inference policy.

    The policy is: use the first attempt unless the gold-free execution feedback was an
    error; only error feedback consumes the already-recorded revision. This is model-free and
    deterministic, so it can turn the n=200 always-revise artifacts into a committed
    CompletionSet for `make eval`.
    """
    from llm_rlvr_sql.eval.parsing import extract_sql
    from llm_rlvr_sql.eval.verifier import GoldExecutionError, execution_match

    if len(completion_set.items) != len(transcripts.items):
        raise ValueError(
            f"{len(completion_set.items)} completions but {len(transcripts.items)} transcripts"
        )

    selected_items: list[ProblemCompletions] = []
    n_revised = 0
    n_skipped = 0
    attempt_correct = always_correct = error_only_correct = 0
    n_fixed = n_harmed = 0
    n_scored = 0
    for item, episode in zip(completion_set.items, transcripts.items, strict=True):
        example = item.example
        if example.db_id != episode.db_id or example.question != episode.question:
            raise ValueError("completion set and transcripts are not aligned")

        use_revision = is_error_feedback(episode.feedback)
        selected_text = episode.final_text if use_revision else episode.attempt_text
        if use_revision:
            n_revised += 1
        selected_sql = extract_sql(selected_text)
        selected_items.append(
            ProblemCompletions(
                example=example,
                samples=(GenerationSample(text=selected_text, sql=selected_sql),),
            )
        )

        db_path = resolve_database(example.db_id, db_root)
        attempt_sql = extract_sql(episode.attempt_text)
        always_sql = item.samples[0].sql

        def verdict(
            sql: str | None,
            gold_sql: str = example.gold_sql,
            database_path: Path = db_path,
        ) -> bool:
            if sql is None:
                return False
            return execution_match(sql, gold_sql, database_path).matched

        try:
            attempt_ok = verdict(attempt_sql)
            always_ok = verdict(always_sql)
            error_only_ok = verdict(selected_sql)
        except GoldExecutionError:
            n_skipped += 1
            continue

        n_scored += 1
        attempt_correct += attempt_ok
        always_correct += always_ok
        error_only_correct += error_only_ok
        if not attempt_ok and error_only_ok:
            n_fixed += 1
        if attempt_ok and not error_only_ok:
            n_harmed += 1

    replay = CompletionSet(
        provenance=completion_set.provenance.model_copy(
            update={"backend": "forced-revision-error-only-replay"}
        ),
        items=tuple(selected_items),
    )
    report = ErrorOnlyRevisionReport(
        n=n_scored,
        n_skipped_gold_error=n_skipped,
        attempt_ex=attempt_correct / n_scored if n_scored else 0.0,
        always_revise_ex=always_correct / n_scored if n_scored else 0.0,
        error_only_ex=error_only_correct / n_scored if n_scored else 0.0,
        n_revised=n_revised,
        revised_rate=n_revised / len(completion_set.items) if completion_set.items else 0.0,
        n_fixed=n_fixed,
        n_harmed=n_harmed,
    )
    return replay, report


def build_revision_messages(
    prompt: Sequence[Mapping[str, object]], attempt_text: str, feedback: str
) -> list[dict[str, object]]:
    """The revision turn's chat messages: the original prompt + the attempt as an assistant
    turn + the feedback/revise user turn. Pure; shared shape with the training rollout via
    `revision_user_message`."""
    return [
        *[dict(message) for message in prompt],
        {"role": "assistant", "content": attempt_text},
        {"role": "user", "content": revision_user_message(feedback)},
    ]


def generate_forced_revision(
    examples: Sequence[Example],
    schemas: Mapping[str, DatabaseSchema],
    db_root: str | Path,
    *,
    model: str,
    decoding: DecodingConfig,
    dataset: str,
    dataset_revision: str,
    split: str,
    adapter_path: str | Path | None = None,
    model_revision: str | None = None,
    db_archive_sha256: str | None = None,
    environment: FeedbackConfig | None = None,
    policy: ForcedRevisionPolicy = "always",
    commit: str | None = None,
    dirty: bool | None = None,
) -> tuple[CompletionSet, ForcedRevisionTranscriptSet]:
    """Run the batched two-pass forced-revision eval on the GPU."""
    from llm_rlvr_sql.eval.vllm_backend import make_vllm_engine, vllm_sampling_params

    policy = validate_forced_revision_policy(policy)
    env_cfg = environment or FeedbackConfig()
    llm, chat_kwargs = make_vllm_engine(model, adapter_path=adapter_path)
    params = vllm_sampling_params(decoding)

    prompts = build_prompts(examples, schemas, style="cot")
    attempt_outputs = llm.chat([list(p) for p in prompts], params, **chat_kwargs)
    attempt_texts = [output.outputs[0].text for output in attempt_outputs]

    feedbacks = [
        attempt_feedback(text, resolve_database(example.db_id, db_root), env_cfg)
        for example, text in zip(examples, attempt_texts, strict=True)
    ]
    revision_indices = [
        i for i, feedback in enumerate(feedbacks) if should_request_revision(policy, feedback)
    ]
    revision_prompts = [
        build_revision_messages(prompts[i], attempt_texts[i], feedbacks[i])
        for i in revision_indices
    ]
    revision_outputs = llm.chat(revision_prompts, params, **chat_kwargs) if revision_prompts else []
    if len(revision_outputs) != len(revision_indices):
        raise ValueError(
            f"{len(revision_indices)} revision prompts but {len(revision_outputs)} outputs"
        )
    revision_text_by_index = {
        i: output.outputs[0].text
        for i, output in zip(revision_indices, revision_outputs, strict=True)
    }
    final_texts = [
        revision_text_by_index.get(i, attempt) for i, attempt in enumerate(attempt_texts)
    ]

    completion_set = finish_eval_run(
        examples,
        final_texts,
        model=model,
        backend="vllm-forced-revision" if policy == "always" else "vllm-forced-revision-error-only",
        dataset=dataset,
        dataset_revision=dataset_revision,
        split=split,
        decoding=decoding,
        model_revision=model_revision,
        db_archive_sha256=db_archive_sha256,
        adapter_path=adapter_path,
        prompt_style="cot",
        commit=commit,
        dirty=dirty,
    )
    transcripts = ForcedRevisionTranscriptSet(
        provenance=completion_set.provenance,
        items=tuple(
            ForcedRevisionEpisode(
                db_id=example.db_id,
                question=example.question,
                attempt_text=attempt,
                feedback=feedback,
                final_text=final,
                revision_requested=i in revision_text_by_index,
                revision_text=revision_text_by_index.get(i),
            )
            for i, (example, attempt, feedback, final) in enumerate(
                zip(examples, attempt_texts, feedbacks, final_texts, strict=True)
            )
        ),
    )
    return completion_set, transcripts
