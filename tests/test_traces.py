"""coldstart-trace-dataset: execution filter, selection policy, guards, manifest (no GPU)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_rlvr_sql.data.models import Example
from llm_rlvr_sql.train.traces import (
    CandidateSet,
    TraceCandidates,
    TraceSamplingConfig,
    build_trace_manifest,
    filter_traces,
    read_candidate_set,
    read_trace_dataset,
    sample_candidates,
    write_trace_dataset,
)

PROMPT = (
    {"role": "system", "content": "instruction"},
    {"role": "user", "content": "Database schema:\n...\n\nQuestion: how many?"},
)

WF_CORRECT = "<think>count rows</think>\n```sql\nSELECT a FROM t\n```"
WF_CORRECT_LONGER = (
    "<think>let me reason about this at some length</think>\n"
    "```sql\nSELECT a FROM t ORDER BY a\n```"
)
WF_WRONG = "<think>r</think>\n```sql\nSELECT a FROM t WHERE a = 99\n```"
MALFORMED_CORRECT = "```sql\nSELECT a FROM t\n```"  # right SQL, no <think> block


def _candidates(*texts: str, gold_sql: str = "SELECT a FROM t") -> list[TraceCandidates]:
    example = Example(db_id="concert", question="how many?", gold_sql=gold_sql, split="train")
    return [TraceCandidates(example=example, prompt=PROMPT, texts=tuple(texts))]


def test_execution_incorrect_candidates_are_dropped(scoring_root: Path) -> None:
    kept, counts = filter_traces(_candidates(WF_WRONG), scoring_root)
    assert kept == ()
    assert counts.n_candidates == 1
    assert counts.n_eligible == 0


def test_malformed_candidates_are_dropped_even_if_correct(scoring_root: Path) -> None:
    kept, counts = filter_traces(_candidates(MALFORMED_CORRECT), scoring_root)
    assert kept == ()
    assert counts.n_eligible == 0


def test_correct_well_formed_trace_is_kept_with_prompt(scoring_root: Path) -> None:
    kept, counts = filter_traces(_candidates(WF_CORRECT, WF_WRONG), scoring_root)
    assert counts.n_kept == 1
    assert kept[0].completion == WF_CORRECT
    assert kept[0].prompt == PROMPT
    assert kept[0].db_id == "concert"


def test_selection_picks_shortest_and_dedups_exact(scoring_root: Path) -> None:
    # Two exact duplicates of the longer trace + one shorter trace: dedup leaves two eligible,
    # selection keeps the shortest.
    kept, counts = filter_traces(
        _candidates(WF_CORRECT_LONGER, WF_CORRECT_LONGER, WF_CORRECT), scoring_root
    )
    assert counts.n_candidates == 3
    assert counts.n_eligible == 2
    assert counts.n_kept == 1
    assert kept[0].completion == WF_CORRECT


def test_trivial_empty_match_does_not_qualify_by_default(scoring_root: Path) -> None:
    # Gold is genuinely empty; an empty-result prediction "matches" under official semantics
    # but must not become training data (the trivial-empty guard is ON for traces).
    empty_pred = "<think>r</think>\n```sql\nSELECT a FROM t WHERE a = 88\n```"
    kept, counts = filter_traces(
        _candidates(empty_pred, gold_sql="SELECT a FROM t WHERE a = 99"), scoring_root
    )
    assert kept == ()
    assert counts.n_kept == 0


def test_gold_error_drops_question_and_is_counted(scoring_root: Path) -> None:
    kept, counts = filter_traces(
        _candidates(WF_CORRECT, gold_sql="SELECT bad FROM nope"), scoring_root
    )
    assert kept == ()
    assert counts.n_gold_errors == 1


def test_train_only_guard_raises_on_dev_pool() -> None:
    dev = Example(db_id="concert", question="q", gold_sql="SELECT a FROM t", split="dev")
    with pytest.raises(ValueError, match="train-split examples only"):
        sample_candidates([dev], {}, model="m", sampling=TraceSamplingConfig())


def _candidate_set(*texts: str, gold_sql: str = "SELECT a FROM t") -> CandidateSet:
    return CandidateSet(
        base_model="Qwen/Qwen2.5-Coder-3B-Instruct",
        base_model_revision="abc123",
        sampling=TraceSamplingConfig(),
        dataset="spider",
        dataset_revision="rev",
        db_archive_sha256="d" * 64,
        items=tuple(_candidates(*texts, gold_sql=gold_sql)),
    )


def test_manifest_carries_reproducibility_fields(scoring_root: Path) -> None:
    candidate_set = _candidate_set(WF_CORRECT)
    _, counts = filter_traces(candidate_set.items, scoring_root)
    manifest = build_trace_manifest(candidate_set, counts, commit="deadbeef", dirty=False)
    assert manifest.base_model_revision == "abc123"
    assert manifest.sampling.n == 8
    assert manifest.sampling.temperature == 0.8
    assert manifest.db_archive_sha256 == "d" * 64
    assert manifest.selection == "shortest"
    assert manifest.counts.n_kept == 1
    assert manifest.git_commit == "deadbeef"


def test_candidate_set_file_round_trip(tmp_path: Path) -> None:
    from llm_rlvr_sql.eval.completions import write_json

    candidate_set = _candidate_set(WF_CORRECT, WF_WRONG)
    path = write_json(candidate_set, tmp_path / "candidates.json")
    assert read_candidate_set(path) == candidate_set


def test_build_traces_cli_runs_the_whole_cpu_funnel(scoring_root: Path, tmp_path: Path) -> None:
    import json

    from llm_rlvr_sql.eval.cli import main
    from llm_rlvr_sql.eval.completions import write_json

    candidates_path = write_json(
        _candidate_set(WF_CORRECT, WF_WRONG, MALFORMED_CORRECT), tmp_path / "candidates.json"
    )
    out = tmp_path / "traces.jsonl"
    manifest_path = tmp_path / "manifest.json"
    code = main(
        [
            "build-traces",
            "--candidates", str(candidates_path),
            "--db-root", str(scoring_root),
            "--out", str(out),
            "--manifest", str(manifest_path),
        ]
    )  # fmt: skip
    assert code == 0
    assert read_trace_dataset(out)[0].completion == WF_CORRECT
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"] == {
        "n_questions": 1,
        "n_candidates": 3,
        "n_eligible": 1,
        "n_kept": 1,
        "n_gold_errors": 0,
    }


def test_trace_dataset_jsonl_round_trip(scoring_root: Path, tmp_path: Path) -> None:
    kept, _ = filter_traces(_candidates(WF_CORRECT), scoring_root)
    path = write_trace_dataset(kept, tmp_path / "traces.jsonl")
    assert read_trace_dataset(path) == kept
    # The on-disk shape is TRL conversational prompt-completion.
    import json

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["prompt"][0]["role"] == "system"
    assert record["completion"] == [{"role": "assistant", "content": WF_CORRECT}]
