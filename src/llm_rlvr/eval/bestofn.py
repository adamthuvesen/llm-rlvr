"""Best-of-N selection — the D6 matched-compute control (grpo-forced-revision task 2.2).

Forced revision spends exactly two generations per question, so the control is best-of-2:
N independently sampled CompletionSets (same checkpoint, same questions, no feedback), one
candidate picked per question by a **gold-free** heuristic, the selection scored normally.

The disclosed heuristic, in order: first candidate whose SQL executes without error AND
returns at least one row; else first that executes without error; else the first candidate.
It never sees gold SQL, gold rows, or a verifier verdict — the same information the
forced-revision policy's feedback channel exposes, minus the revision turn.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from llm_rlvr._base import Frozen
from llm_rlvr.data.paths import resolve_database
from llm_rlvr.env.sandbox import ExecutionResult, SqliteSandbox
from llm_rlvr.env.text2sql_env import AGENT_ROW_CAP, AGENT_TIMEOUT_S
from llm_rlvr.eval.completions import CompletionSet, ProblemCompletions
from llm_rlvr.eval.provenance import GenerationProvenance


class BestOfNReport(Frozen):
    """How the gold-free selection decided, per rule — committed with the control."""

    n_items: int
    n_candidates_per_item: int
    n_selected_nonempty: int  # executed without error, non-empty result
    n_selected_executes: int  # executed without error, empty result (no non-empty candidate)
    n_fallback_first: int  # no candidate executed; the first one stands
    n_items_with_disagreeing_candidates: int  # candidates' SQLs differ (selection mattered)


def _candidate_rank(sql: str | None, db_path: str | Path) -> int:
    """0 = executes + non-empty, 1 = executes + empty, 2 = error/unparseable. Gold-free."""
    if sql is None:
        return 2
    outcome = SqliteSandbox(db_path, timeout_s=AGENT_TIMEOUT_S, row_cap=AGENT_ROW_CAP).execute(sql)
    if isinstance(outcome, ExecutionResult):
        return 0 if outcome.row_count > 0 else 1
    return 2


def select_best_of_n(
    completion_sets: Sequence[CompletionSet],
    db_root: str | Path,
) -> tuple[CompletionSet, BestOfNReport]:
    """Pick one sample per question across N aligned CompletionSets. Pure CPU + sandbox.

    The sets must cover the same examples in the same order (asserted). Ties at the same
    rank go to the earliest set — deterministic.
    """
    if len(completion_sets) < 2:
        raise ValueError("best-of-N needs at least 2 completion sets")
    first = completion_sets[0]
    for other in completion_sets[1:]:
        if len(other.items) != len(first.items):
            raise ValueError("completion sets cover different numbers of examples")

    n_nonempty = n_executes = n_fallback = n_disagree = 0
    selected: list[ProblemCompletions] = []
    for row_items in zip(*(cs.items for cs in completion_sets), strict=True):
        example = row_items[0].example
        for item in row_items[1:]:
            if item.example.question != example.question or item.example.db_id != example.db_id:
                raise ValueError(
                    "completion sets are not aligned: differing examples at the same index"
                )
        candidates = [item.samples[0] for item in row_items]
        db_path = resolve_database(example.db_id, db_root)
        ranks = [_candidate_rank(candidate.sql, db_path) for candidate in candidates]
        best = min(range(len(candidates)), key=lambda i: (ranks[i], i))
        if ranks[best] == 0:
            n_nonempty += 1
        elif ranks[best] == 1:
            n_executes += 1
        else:
            n_fallback += 1
        if len({candidate.sql for candidate in candidates}) > 1:
            n_disagree += 1
        selected.append(ProblemCompletions(example=example, samples=(candidates[best],)))

    provenance = GenerationProvenance.model_validate(
        {**first.provenance.model_dump(), "backend": f"bestof{len(completion_sets)}-select"}
    )
    report = BestOfNReport(
        n_items=len(selected),
        n_candidates_per_item=len(completion_sets),
        n_selected_nonempty=n_nonempty,
        n_selected_executes=n_executes,
        n_fallback_first=n_fallback,
        n_items_with_disagreeing_candidates=n_disagree,
    )
    return CompletionSet(provenance=provenance, items=tuple(selected)), report
