"""The `CompletionSet` artifact: model outputs persisted so scoring is a separate, model-free step.

`generate` writes a `CompletionSet`; `score` reads it on a CPU box with no generation backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator

from llm_rlvr_sql._base import Frozen
from llm_rlvr_sql.data.models import Example
from llm_rlvr_sql.eval.parsing import extract_sql
from llm_rlvr_sql.eval.provenance import GenerationProvenance


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
        elif text.strip() != supplied_sql.strip():
            raise ValueError("GenerationSample.sql is set but text contains no matching SQL")
        return data


class ProblemCompletions(Frozen):
    example: Example
    samples: tuple[GenerationSample, ...] = Field(min_length=1)


class CompletionSet(Frozen):
    provenance: GenerationProvenance
    items: tuple[ProblemCompletions, ...]

    @model_validator(mode="after")
    def item_count_matches_provenance(self) -> Self:
        if len(self.items) != self.provenance.n_examples:
            raise ValueError(
                "CompletionSet item count does not match provenance.n_examples: "
                f"{len(self.items)} != {self.provenance.n_examples}"
            )
        return self


def write_json(model: BaseModel, path: str | Path) -> Path:
    """Write any Pydantic model to a JSON file, creating parent directories. Returns the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    return out


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


def write_completion_set(completion_set: CompletionSet, path: str | Path) -> Path:
    return write_json(completion_set, path)


def read_completion_set(path: str | Path) -> CompletionSet:
    return CompletionSet.model_validate_json(Path(path).read_text(encoding="utf-8"))
