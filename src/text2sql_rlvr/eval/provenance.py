"""Provenance: everything needed to reproduce a baseline number.

Captures the dataset revision + DB-archive checksum, the model + revision, the decoding config,
the resolved few-shot exemplars, the git commit/dirty flag, and library versions. Mirrors
`grpo-gain-decomposition`'s RunProvenance.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from importlib import metadata

from pydantic import Field

from text2sql_rlvr._base import Frozen

_TRACKED_LIBRARIES = ("text2sql-rlvr", "transformers", "torch", "vllm", "datasets")


class DecodingConfig(Frozen):
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 1024
    seed: int = 0


class GenerationProvenance(Frozen):
    model: str
    backend: str
    dataset: str
    dataset_revision: str
    split: str
    n_examples: int
    decoding: DecodingConfig = Field(default_factory=DecodingConfig)
    model_revision: str | None = None
    db_archive_sha256: str | None = None
    # LoRA adapter loaded over the base (the SFT checkpoint); None = the plain base model.
    adapter: str | None = None
    prompt_style: str = "cot"
    few_shot_k: int = 0
    few_shot_seed: int | None = None
    # The resolved exemplars actually injected, as (db_id, question) pairs — so a reviewer can read
    # the demonstrations off a CompletionSet without re-deriving them from the seed.
    few_shot_exemplars: tuple[tuple[str, str], ...] = ()
    git_commit: str = "unknown"
    git_dirty: bool = False
    library_versions: dict[str, str] = Field(default_factory=dict)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def git_is_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return bool(out.stdout.strip())


def library_versions(packages: tuple[str, ...] = _TRACKED_LIBRARIES) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def build_generation_provenance(
    *,
    model: str,
    backend: str,
    dataset: str,
    dataset_revision: str,
    split: str,
    n_examples: int,
    decoding: DecodingConfig,
    model_revision: str | None = None,
    db_archive_sha256: str | None = None,
    adapter: str | None = None,
    prompt_style: str = "cot",
    few_shot_k: int = 0,
    few_shot_seed: int | None = None,
    few_shot_exemplars: Sequence[tuple[str, str]] = (),
    commit: str | None = None,
    dirty: bool | None = None,
) -> GenerationProvenance:
    """Assemble provenance, capturing git + library state from the current environment.

    `commit`/`dirty` override the local git lookup — needed on Modal, where the image strips `.git`
    and the entrypoint passes the real values in.
    """
    return GenerationProvenance(
        model=model,
        backend=backend,
        dataset=dataset,
        dataset_revision=dataset_revision,
        split=split,
        n_examples=n_examples,
        decoding=decoding,
        model_revision=model_revision,
        db_archive_sha256=db_archive_sha256,
        adapter=adapter,
        prompt_style=prompt_style,
        few_shot_k=few_shot_k,
        few_shot_seed=few_shot_seed,
        few_shot_exemplars=tuple(few_shot_exemplars),
        git_commit=commit if commit is not None else git_commit(),
        git_dirty=dirty if dirty is not None else git_is_dirty(),
        library_versions=library_versions(),
    )
