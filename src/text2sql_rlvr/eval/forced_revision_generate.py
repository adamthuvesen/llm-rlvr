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

from text2sql_rlvr._base import Frozen
from text2sql_rlvr.data.models import DatabaseSchema, Example
from text2sql_rlvr.data.spider import resolve_database
from text2sql_rlvr.env.text2sql_env import FeedbackConfig
from text2sql_rlvr.eval.completions import CompletionSet, assemble_completion_set
from text2sql_rlvr.eval.generate import build_prompts
from text2sql_rlvr.eval.provenance import (
    DecodingConfig,
    GenerationProvenance,
    build_generation_provenance,
)
from text2sql_rlvr.train.forced_revision import attempt_feedback, revision_user_message


class ForcedRevisionEpisode(Frozen):
    """One evaluated episode: attempt, the feedback it produced, and the final (revision)."""

    db_id: str
    question: str
    attempt_text: str
    feedback: str
    final_text: str


class ForcedRevisionTranscriptSet(Frozen):
    """All episodes of one forced-revision eval, sharing its CompletionSet's provenance."""

    provenance: GenerationProvenance
    items: tuple[ForcedRevisionEpisode, ...]


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
    commit: str | None = None,
    dirty: bool | None = None,
) -> tuple[CompletionSet, ForcedRevisionTranscriptSet]:
    """Run the batched two-pass forced-revision eval on the GPU."""
    from vllm import LLM, SamplingParams  # lazy import: GPU backend only

    env_cfg = environment or FeedbackConfig()
    chat_kwargs: dict[str, object] = {}
    if adapter_path is not None:
        from vllm.lora.request import LoRARequest

        llm = LLM(model=model, enable_lora=True, max_lora_rank=64)
        chat_kwargs["lora_request"] = LoRARequest("adapter", 1, str(adapter_path))
    else:
        llm = LLM(model=model)
    params = SamplingParams(
        temperature=decoding.temperature,
        top_p=decoding.top_p,
        max_tokens=decoding.max_new_tokens,
        seed=decoding.seed,
    )

    prompts = build_prompts(examples, schemas, style="cot")
    attempt_outputs = llm.chat([list(p) for p in prompts], params, **chat_kwargs)
    attempt_texts = [output.outputs[0].text for output in attempt_outputs]

    feedbacks = [
        attempt_feedback(text, resolve_database(example.db_id, db_root), env_cfg)
        for example, text in zip(examples, attempt_texts, strict=True)
    ]
    revision_prompts = [
        build_revision_messages(prompt, attempt, feedback)
        for prompt, attempt, feedback in zip(prompts, attempt_texts, feedbacks, strict=True)
    ]
    revision_outputs = llm.chat(revision_prompts, params, **chat_kwargs)
    final_texts = [output.outputs[0].text for output in revision_outputs]

    provenance = build_generation_provenance(
        model=model,
        backend="vllm-forced-revision",
        dataset=dataset,
        dataset_revision=dataset_revision,
        split=split,
        n_examples=len(examples),
        decoding=decoding,
        model_revision=model_revision,
        db_archive_sha256=db_archive_sha256,
        adapter=str(adapter_path) if adapter_path is not None else None,
        prompt_style="cot",
        commit=commit,
        dirty=dirty,
    )
    completion_set = assemble_completion_set(examples, final_texts, provenance)
    transcripts = ForcedRevisionTranscriptSet(
        provenance=provenance,
        items=tuple(
            ForcedRevisionEpisode(
                db_id=example.db_id,
                question=example.question,
                attempt_text=attempt,
                feedback=feedback,
                final_text=final,
            )
            for example, attempt, feedback, final in zip(
                examples, attempt_texts, feedbacks, final_texts, strict=True
            )
        ),
    )
    return completion_set, transcripts
