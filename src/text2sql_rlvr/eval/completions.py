"""The `CompletionSet` artifact: model outputs persisted so scoring is a separate, model-free step.

`generate` writes a `CompletionSet`; `score` reads it on a CPU box with no generation backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from text2sql_rlvr._base import Frozen
from text2sql_rlvr.data.models import Example
from text2sql_rlvr.eval.parsing import extract_sql
from text2sql_rlvr.eval.provenance import GenerationProvenance


class GenerationSample(Frozen):
    """One model output for one example: the raw text and the extracted SQL (None if absent)."""

    text: str
    sql: str | None = None


class ProblemCompletions(Frozen):
    example: Example
    samples: tuple[GenerationSample, ...]


class CompletionSet(Frozen):
    provenance: GenerationProvenance
    items: tuple[ProblemCompletions, ...]


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
