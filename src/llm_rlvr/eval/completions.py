"""The `CompletionSet` artifact: model outputs persisted so scoring is a separate, model-free step.

`generate` writes a `CompletionSet`; `score` reads it on a CPU box with no generation backend.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from llm_rlvr._base import Frozen
from llm_rlvr.artifacts.io import atomic_write_json
from llm_rlvr.artifacts.provenance import GenerationProvenance
from llm_rlvr.artifacts.runs import require_complete
from llm_rlvr.data.models import Example
from llm_rlvr.tasks.sql.parsing import extract_sql


class GenerationSample(Frozen):
    """One model output for one example: the raw text and the extracted SQL (None if absent)."""

    text: str
    sql: str | None = None

    @model_validator(mode="before")
    @classmethod
    def derive_or_check_sql(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        text = data.get("text")
        if not isinstance(text, str):
            return data

        parsed_sql = extract_sql(text)
        supplied_sql = data.get("sql")
        if supplied_sql is None:
            return {**data, "sql": parsed_sql}
        if not isinstance(supplied_sql, str):
            return data

        if parsed_sql is not None:
            if parsed_sql != supplied_sql:
                raise ValueError("GenerationSample.sql does not match SQL extracted from text")
        else:
            raise ValueError("GenerationSample.sql is set but text contains no extracted SQL")
        return data


class ProblemCompletions(Frozen):
    example: Example
    samples: tuple[GenerationSample, ...] = Field(min_length=1)


class CompletionSet(Frozen):
    schema_version: Literal[1] = 1
    provenance: GenerationProvenance
    items: tuple[ProblemCompletions, ...]

    @property
    def canonical_sha256(self) -> str:
        """SHA256 of the fixed v1 identity projection, including v1 defaults."""
        # Keep this projection fixed. New identity fields require a new schema version.
        payload = self.model_dump(
            mode="json",
            include={
                "schema_version": True,
                "provenance": {
                    "model": True,
                    "backend": True,
                    "dataset": True,
                    "dataset_revision": True,
                    "split": True,
                    "n_examples": True,
                    "model_revision": True,
                    "db_archive_sha256": True,
                    "adapter": True,
                    "adapter_sha256": True,
                    "adapter_provenance_sha256": True,
                    "prompt_style": True,
                    "selection_policy": True,
                    "selection_seed": True,
                    "few_shot_k": True,
                    "few_shot_seed": True,
                    "few_shot_exemplars": True,
                    "git_commit": True,
                    "git_dirty": True,
                    "library_versions": True,
                    "decoding": {"temperature", "top_p", "max_new_tokens", "seed"},
                },
                "items": {
                    "__all__": {
                        "example": {
                            "db_id",
                            "question",
                            "gold_sql",
                            "split",
                            "dataset",
                            "evidence",
                        },
                        "samples": {"__all__": {"text", "sql"}},
                    }
                },
            },
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def item_count_matches_provenance(self) -> Self:
        if len(self.items) != self.provenance.n_examples:
            raise ValueError(
                "CompletionSet item count does not match provenance.n_examples: "
                f"{len(self.items)} != {self.provenance.n_examples}"
            )
        for item in self.items:
            if item.example.dataset != self.provenance.dataset:
                raise ValueError("CompletionSet example dataset does not match provenance.dataset")
            if item.example.split != self.provenance.split:
                raise ValueError("CompletionSet example split does not match provenance.split")
        return self


def assemble_completion_set(
    examples: Sequence[Example],
    raw_texts: Sequence[str],
    provenance: GenerationProvenance,
) -> CompletionSet:
    """Pair each example with its raw completion (one sample), extracting the SQL. Pure/testable."""
    if len(examples) != len(raw_texts):
        raise ValueError(f"{len(examples)} examples but {len(raw_texts)} completions")
    items = tuple(
        ProblemCompletions(
            example=example,
            samples=(GenerationSample(text=text, sql=extract_sql(text)),),
        )
        for example, text in zip(examples, raw_texts, strict=True)
    )
    return CompletionSet(provenance=provenance, items=items)


def write_completion_set(
    completion_set: CompletionSet, path: str | Path, *, overwrite: bool = False
) -> Path:
    return atomic_write_json(completion_set, path, overwrite=overwrite)


def read_completion_set(path: str | Path) -> CompletionSet:
    require_complete(path)
    return CompletionSet.model_validate_json(Path(path).read_text(encoding="utf-8"))
