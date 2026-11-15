"""Provenance: everything needed to reproduce a baseline number.

Captures the dataset revision + DB-archive checksum, the model + revision, the decoding config,
the resolved few-shot exemplars, the git commit/dirty flag, and library versions. Mirrors
`grpo-decomp`'s RunProvenance.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self

from pydantic import Field, field_serializer, field_validator, model_validator

from llm_rlvr._base import Frozen

_TRACKED_LIBRARIES = ("llm-rlvr", "transformers", "torch", "vllm", "datasets")


class DecodingConfig(Frozen):
    temperature: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    max_new_tokens: int = Field(default=1024, gt=0)
    seed: int = 0


class GenerationProvenance(Frozen):
    model: str
    backend: str
    dataset: str
    dataset_revision: str
    split: str
    n_examples: int = Field(ge=0)
    decoding: DecodingConfig = Field(default_factory=DecodingConfig)
    model_revision: str | None = None
    db_archive_sha256: str | None = None
    # LoRA adapter loaded over the base (the SFT checkpoint); None = the plain base model.
    adapter: str | None = None
    adapter_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    adapter_provenance_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    prompt_style: str = "cot"
    selection_policy: Literal["all", "prefix", "seeded"] | None = None
    selection_seed: int | None = None
    few_shot_k: int = Field(default=0, ge=0)
    few_shot_seed: int | None = None
    # The resolved exemplars actually injected, as (db_id, question) pairs — so a reviewer can read
    # the demonstrations off a CompletionSet without re-deriving them from the seed.
    few_shot_exemplars: tuple[tuple[str, str], ...] = ()
    git_commit: str = "unknown"
    git_dirty: bool = False
    library_versions: Mapping[str, str] = Field(default_factory=lambda: MappingProxyType({}))

    @field_validator("library_versions")
    @classmethod
    def immutable_versions(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return MappingProxyType(dict(value))

    @field_serializer("library_versions")
    def serialize_versions(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    @model_validator(mode="after")
    def exemplars_match_count(self) -> Self:
        # Older artifacts recorded the recipe without the resolved exemplars.
        if self.few_shot_exemplars and len(self.few_shot_exemplars) != self.few_shot_k:
            raise ValueError("few_shot_exemplars must match few_shot_k when recorded")
        if (self.selection_policy == "seeded") != (self.selection_seed is not None):
            raise ValueError("selection_seed is required only for seeded selection")
        return self


class ScoredDatabase(Frozen):
    db_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScoringProvenance(Frozen):
    """Scoring inputs and code state, separate from the generation environment."""

    schema_version: Literal[1] = 1
    completion_schema_version: Literal[1] = 1
    # SHA256 of canonical sorted compact model JSON, not the original file bytes.
    completion_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scorer_policy: str
    scorer_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    databases: tuple[ScoredDatabase, ...]
    git_commit: str
    git_dirty: bool

    @model_validator(mode="after")
    def database_ids_are_unique(self) -> Self:
        ids = [database.db_id for database in self.databases]
        if len(ids) != len(set(ids)):
            raise ValueError("scored database ids must be unique")
        return self


def _source_checkout() -> Path | None:
    package = Path(__file__).resolve().parent
    while package.name != "llm_rlvr":
        if package == package.parent:
            return None
        package = package.parent
    if package.parent.name != "src":
        return None
    root = package.parent.parent
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return root if Path(out.stdout.strip()).resolve() == root else None
    except subprocess.SubprocessError, OSError:
        return None


def git_commit() -> str:
    root = _source_checkout()
    if root is None:
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except subprocess.SubprocessError, OSError:
        return "unknown"


def git_is_dirty() -> bool:
    root = _source_checkout()
    if root is None:
        return False
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.SubprocessError, OSError:
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
    adapter_sha256: str | None = None,
    adapter_provenance_sha256: str | None = None,
    prompt_style: str = "cot",
    selection_policy: Literal["all", "prefix", "seeded"] | None = None,
    selection_seed: int | None = None,
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
        adapter_sha256=adapter_sha256,
        adapter_provenance_sha256=adapter_provenance_sha256,
        prompt_style=prompt_style,
        selection_policy=selection_policy,
        selection_seed=selection_seed,
        few_shot_k=few_shot_k,
        few_shot_seed=few_shot_seed,
        few_shot_exemplars=tuple(few_shot_exemplars),
        git_commit=commit if commit is not None else git_commit(),
        git_dirty=dirty if dirty is not None else git_is_dirty(),
        library_versions=library_versions(),
    )
