"""Small helpers for finishing eval generation runs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from llm_rlvr.data.models import Example
from llm_rlvr.eval.completions import CompletionSet, assemble_completion_set
from llm_rlvr.eval.provenance import DecodingConfig, build_generation_provenance


def finish_eval_run(
    examples: Sequence[Example],
    final_texts: Sequence[str],
    *,
    model: str,
    backend: str,
    dataset: str,
    dataset_revision: str,
    split: str,
    decoding: DecodingConfig,
    model_revision: str | None = None,
    db_archive_sha256: str | None = None,
    adapter_path: str | Path | None = None,
    prompt_style: str = "cot",
    few_shot_k: int = 0,
    few_shot_seed: int | None = None,
    few_shot_exemplars: Sequence[tuple[str, str]] = (),
    commit: str | None = None,
    dirty: bool | None = None,
) -> CompletionSet:
    provenance = build_generation_provenance(
        model=model,
        backend=backend,
        dataset=dataset,
        dataset_revision=dataset_revision,
        split=split,
        n_examples=len(examples),
        decoding=decoding,
        model_revision=model_revision,
        db_archive_sha256=db_archive_sha256,
        adapter=str(adapter_path) if adapter_path is not None else None,
        prompt_style=prompt_style,
        few_shot_k=few_shot_k,
        few_shot_seed=few_shot_seed,
        few_shot_exemplars=few_shot_exemplars,
        commit=commit,
        dirty=dirty,
    )
    return assemble_completion_set(examples, final_texts, provenance)
