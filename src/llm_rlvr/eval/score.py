"""Score a `CompletionSet` into execution accuracy — deterministic and backend-free.

Takes the first sample per example (pass@1 baseline), runs the execution-match verifier against
the gold database, and aggregates. No model is loaded, so this runs on a CPU box and reproduces
the same number every time.

`execution_accuracy` is `n_correct / n` over all items: completions with no SQL and items where
the gold query itself errors both count as failures in the denominator. `n_gold_errors` is
reported separately and should be 0 on a healthy run (a non-zero count signals a broken
example/database pairing, not a model failure). `well_formed_rate` is the fraction of raw
completions in the trained `<think>` + ```sql``` shape — the cold-start's format metric,
independent of execution accuracy.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, Self

from pydantic import Field, field_serializer, field_validator, model_validator

from llm_rlvr._base import Frozen
from llm_rlvr.artifacts.provenance import (
    GenerationProvenance,
    ScoredDatabase,
    ScoringProvenance,
    git_commit,
    git_is_dirty,
)
from llm_rlvr.data.paths import resolve_database
from llm_rlvr.env import sandbox
from llm_rlvr.eval import completions
from llm_rlvr.eval.completions import CompletionSet, ProblemCompletions
from llm_rlvr.tasks.sql import parsing, verifier
from llm_rlvr.tasks.sql.parsing import well_formed
from llm_rlvr.tasks.sql.verifier import (
    SCORER_POLICY_VERSION,
    GoldExecutionError,
    MatchReason,
    VerifierConfig,
    execution_match,
)
from llm_rlvr.third_party.test_suite_sql_eval import exec_eval, parse

ScoreReason = MatchReason | Literal["no_sql", "gold_error"]


class ScoreOutcome(Frozen):
    index: int = Field(ge=0, strict=True)
    db_id: str
    reason: ScoreReason
    matched: bool
    well_formed: bool
    detail: str | None = None

    @model_validator(mode="after")
    def match_reason_is_consistent(self) -> Self:
        if self.matched != (self.reason == "match"):
            raise ValueError("ScoreOutcome matched must agree with reason")
        return self


class ScoreReport(Frozen):
    execution_accuracy: float = Field(ge=0, le=1, allow_inf_nan=False)
    well_formed_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    n: int = Field(ge=0, strict=True)
    n_correct: int = Field(ge=0, strict=True)
    n_gold_errors: int = Field(ge=0, strict=True)
    reasons: Mapping[ScoreReason, Annotated[int, Field(ge=0, strict=True)]]
    verifier: VerifierConfig
    provenance: GenerationProvenance
    scoring: ScoringProvenance | None = None
    outcomes: tuple[ScoreOutcome, ...] | None = None

    @field_validator("reasons")
    @classmethod
    def immutable_reasons(cls, value: Mapping[ScoreReason, int]) -> Mapping[ScoreReason, int]:
        return MappingProxyType(dict(value))

    @field_serializer("reasons")
    def serialize_reasons(self, value: Mapping[ScoreReason, int]) -> dict[ScoreReason, int]:
        return dict(value)

    @model_validator(mode="after")
    def counts_match_rates(self) -> Self:
        if self.n != self.provenance.n_examples:
            raise ValueError("ScoreReport n must match provenance.n_examples")
        if sum(self.reasons.values()) != self.n:
            raise ValueError("ScoreReport reasons must sum to n")
        if self.n_correct != self.reasons.get("match", 0):
            raise ValueError("ScoreReport n_correct must match reasons.match")
        if self.n_gold_errors != self.reasons.get("gold_error", 0):
            raise ValueError("ScoreReport n_gold_errors must match reasons.gold_error")
        expected = self.n_correct / self.n if self.n else 0.0
        if not math.isclose(self.execution_accuracy, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("ScoreReport execution_accuracy must equal n_correct / n")
        if not self.n and self.well_formed_rate != 0:
            raise ValueError("ScoreReport empty population must have zero well_formed_rate")
        formed = self.well_formed_rate * self.n
        if not math.isclose(formed, round(formed), rel_tol=0, abs_tol=1e-9):
            raise ValueError("ScoreReport well_formed_rate must describe an integer count")
        if self.outcomes is not None:
            if tuple(outcome.index for outcome in self.outcomes) != tuple(range(self.n)):
                raise ValueError("ScoreReport outcomes must cover each item in order")
            if Counter(outcome.reason for outcome in self.outcomes) != Counter(self.reasons):
                raise ValueError("ScoreReport reasons must match outcomes")
            if sum(outcome.well_formed for outcome in self.outcomes) != round(formed):
                raise ValueError("ScoreReport well_formed_rate must match outcomes")
        return self


def _score_item(
    index: int, item: ProblemCompletions, db_root: str | Path, cfg: VerifierConfig
) -> ScoreOutcome:
    """One item's verdict: (well_formed, reason, matched). Runs on any thread — the sandbox
    opens a fresh read-only connection per execution, so items are independent."""
    sample = item.samples[0]
    formed = well_formed(sample.text)
    sql = sample.sql
    if sql is None:
        return ScoreOutcome(
            index=index,
            db_id=item.example.db_id,
            well_formed=formed,
            reason="no_sql",
            matched=False,
            detail="first sample contains no extracted SQL",
        )
    db_path = resolve_database(item.example.db_id, db_root)
    try:
        result = execution_match(sql, item.example.gold_sql, db_path, config=cfg)
    except GoldExecutionError as exc:
        return ScoreOutcome(
            index=index,
            db_id=item.example.db_id,
            well_formed=formed,
            reason="gold_error",
            matched=False,
            detail=str(exc),
        )
    return ScoreOutcome(
        index=index,
        db_id=item.example.db_id,
        well_formed=formed,
        reason=result.reason,
        matched=result.matched,
        detail=None if result.matched else f"database={db_path}",
    )


def score_completion_set(
    completion_set: CompletionSet,
    db_root: str | Path,
    *,
    config: VerifierConfig | None = None,
    workers: int = 1,
) -> ScoreReport:
    """Aggregate pass@1 single-DB execution accuracy over a completion set.

    `workers` > 1 scores items on a thread pool (SQLite runs outside the GIL; BIRD's large
    databases dominate wall-clock). Verdicts are aggregated in item order, so the report is
    byte-identical at any worker count.
    """
    cfg = config or VerifierConfig()
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    database_ids = sorted(
        {item.example.db_id for item in completion_set.items if item.samples[0].sql is not None}
    )
    databases: list[ScoredDatabase] = []
    for db_id in database_ids:
        path = resolve_database(db_id, db_root)
        with path.open("rb") as database:
            digest = hashlib.file_digest(database, "sha256").hexdigest()
        databases.append(ScoredDatabase(db_id=db_id, sha256=digest))
    sources = [
        Path(__file__),
        *(
            Path(inspect.getfile(module))
            for module in (completions, verifier, parsing, sandbox, exec_eval, parse)
        ),
    ]
    source_hash = hashlib.sha256()
    package_root = Path(__file__).resolve().parents[1]
    for source in sorted(sources):
        source_hash.update(str(source.resolve().relative_to(package_root)).encode("utf-8"))
        source_hash.update(b"\0")
        source_hash.update(source.read_bytes())
    scoring = ScoringProvenance(
        completion_set_sha256=completion_set.canonical_sha256,
        scorer_policy=SCORER_POLICY_VERSION,
        scorer_source_sha256=source_hash.hexdigest(),
        databases=tuple(databases),
        git_commit=git_commit(),
        git_dirty=git_is_dirty(),
    )

    if workers == 1:
        verdicts = [
            _score_item(index, item, db_root, cfg)
            for index, item in enumerate(completion_set.items)
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            verdicts = list(
                pool.map(
                    lambda pair: _score_item(pair[0], pair[1], db_root, cfg),
                    enumerate(completion_set.items),
                )
            )

    reasons: Counter[ScoreReason] = Counter()
    correct = 0
    gold_errors = 0
    n_well_formed = 0
    for outcome in verdicts:
        if outcome.well_formed:
            n_well_formed += 1
        reasons[outcome.reason] += 1
        if outcome.reason == "gold_error":
            gold_errors += 1
        if outcome.matched:
            correct += 1

    n = len(completion_set.items)
    return ScoreReport(
        execution_accuracy=correct / n if n else 0.0,
        well_formed_rate=n_well_formed / n if n else 0.0,
        n=n,
        n_correct=correct,
        n_gold_errors=gold_errors,
        reasons=dict(reasons),
        verifier=cfg,
        provenance=completion_set.provenance,
        scoring=scoring,
        outcomes=tuple(verdicts),
    )
