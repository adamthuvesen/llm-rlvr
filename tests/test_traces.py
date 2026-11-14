"""coldstart-trace-dataset: execution filter, selection policy, guards, manifest (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from llm_rlvr.data.models import DatabaseSchema, Example
from llm_rlvr.harness.messages import ChatMessage
from llm_rlvr.train.traces import (
    CandidateSet,
    TraceCandidates,
    TraceSamplingConfig,
    build_trace_manifest,
    filter_traces,
    read_candidate_set,
    read_trace_rows,
    sample_candidates,
    write_trace_dataset,
)

PROMPT = (
    ChatMessage(role="system", content="instruction"),
    ChatMessage(role="user", content="Database schema:\n...\n\nQuestion: how many?"),
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


def test_sampling_passes_model_revision_to_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakeLLM:
        def __init__(self, **kwargs) -> None:
            seen.update(kwargs)

        def chat(self, prompts, params, **_kwargs):
            return [
                SimpleNamespace(
                    outputs=[SimpleNamespace(text=f"<think>{i}</think>\n```sql\nSELECT {i}\n```")]
                )
                for _prompt in prompts
                for i in range(params["n"])
            ]

    class GroupingLLM(FakeLLM):
        def chat(self, prompts, params, **_kwargs):
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(text=f"<think>{i}</think>\n```sql\nSELECT {i}\n```")
                        for i in range(params["n"])
                    ]
                )
                for _prompt in prompts
            ]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=GroupingLLM, SamplingParams=lambda **kwargs: kwargs),
    )
    example = Example(db_id="concert", question="q", gold_sql="SELECT 1", split="train")
    schema = {"concert": DatabaseSchema(db_id="concert", tables=())}
    candidates = sample_candidates(
        [example],
        schema,
        model="fake",
        model_revision="abc123",
        sampling=TraceSamplingConfig(n=2),
    )
    assert seen["revision"] == "abc123"
    assert seen["tokenizer_revision"] == "abc123"
    assert len(candidates[0].texts) == 2


def _candidate_set(*texts: str, gold_sql: str = "SELECT a FROM t") -> CandidateSet:
    return CandidateSet(
        base_model="Qwen/Qwen2.5-Coder-3B-Instruct",
        base_model_revision="abc123",
        sampling=TraceSamplingConfig(n=len(texts)),
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
    assert manifest.sampling.n == 1
    assert manifest.sampling.temperature == 0.8
    assert manifest.db_archive_sha256 == "d" * 64
    assert manifest.selection == "shortest"
    assert manifest.counts.n_kept == 1
    assert manifest.git_commit == "deadbeef"


def test_candidate_set_file_round_trip(tmp_path: Path) -> None:
    from llm_rlvr.artifacts.io import atomic_write_json

    candidate_set = _candidate_set(WF_CORRECT, WF_WRONG)
    path = atomic_write_json(candidate_set, tmp_path / "candidates.json")
    assert read_candidate_set(path) == candidate_set


def test_build_traces_cli_runs_the_whole_cpu_funnel(scoring_root: Path, tmp_path: Path) -> None:
    import json

    from llm_rlvr.artifacts.io import atomic_write_json
    from llm_rlvr.cli import main

    candidates_path = atomic_write_json(
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
    assert read_trace_rows(out)[0].completion[0].content == WF_CORRECT
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"] == {
        "n_questions": 1,
        "n_candidates": 3,
        "n_eligible": 1,
        "n_kept": 1,
        "n_gold_errors": 0,
        "gold_error_examples": [],
    }


def test_trace_dataset_jsonl_round_trip(scoring_root: Path, tmp_path: Path) -> None:
    kept, _ = filter_traces(_candidates(WF_CORRECT), scoring_root)
    path = write_trace_dataset(kept, tmp_path / "traces.jsonl")
    rows = read_trace_rows(path)
    assert [
        (row.db_id, row.question, row.prompt, row.completion[0].content, row.source_example)
        for row in rows
    ] == [
        (trace.db_id, trace.question, trace.prompt, trace.completion, trace.source_example)
        for trace in kept
    ]
    # The on-disk shape is TRL conversational prompt-completion.
    import json

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["prompt"][0]["role"] == "system"
    assert record["completion"] == [{"role": "assistant", "content": WF_CORRECT}]


def test_trace_dataset_rejects_empty_completion(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"prompt": [], "completion": [], "db_id": "concert", "question": "q"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="completion"):
        read_trace_rows(path)


def test_dev_candidates_rejected_at_ingestion() -> None:
    dev = Example(db_id="concert", question="q", gold_sql="SELECT 1", split="dev")
    with pytest.raises(ValueError, match="train-split examples only"):
        TraceCandidates(example=dev, prompt=PROMPT, texts=(WF_CORRECT,))


def test_filter_rejects_dev_even_for_constructed_candidates(scoring_root: Path) -> None:
    candidate = _candidates(WF_CORRECT)[0]
    leaked = candidate.model_copy(
        update={"example": candidate.example.model_copy(update={"split": "dev"})}
    )
    with pytest.raises(ValueError, match="train-split examples only"):
        filter_traces([leaked], scoring_root)


def test_trace_filter_cannot_disable_empty_guard(scoring_root: Path) -> None:
    from llm_rlvr.tasks.sql.verifier import VerifierConfig

    with pytest.raises(ValueError, match="guard_trivial_empty=True"):
        filter_traces(_candidates(WF_CORRECT), scoring_root, config=VerifierConfig())


@pytest.mark.parametrize(
    "updates",
    [
        {"sampling": TraceSamplingConfig(n=2)},
        {"dataset": "bird"},
        {"n_questions": 2},
        {"n_candidates": 2},
    ],
)
def test_candidates_reject_inconsistent_metadata(updates: dict[str, object]) -> None:
    record = _candidate_set(WF_CORRECT).model_dump() | updates
    with pytest.raises(ValueError):
        CandidateSet.model_validate(record)


def test_manifest_retains_all_sources_and_gold_failures(scoring_root: Path) -> None:
    candidates = _candidate_set(WF_CORRECT, gold_sql="SELECT missing FROM absent")
    _, counts = filter_traces(candidates.items, scoring_root)
    manifest = build_trace_manifest(candidates, counts, commit="x", dirty=False)
    assert manifest.source_examples == (candidates.items[0].example,)
    assert counts.gold_error_examples == manifest.source_examples


def test_manifest_rejects_unrelated_filter_counts(scoring_root: Path) -> None:
    candidates = _candidate_set(WF_CORRECT)
    _, counts = filter_traces(candidates.items, scoring_root)
    wrong = counts.model_copy(update={"n_questions": 2})
    with pytest.raises(ValueError, match="filter counts differ"):
        build_trace_manifest(candidates, wrong)


@pytest.mark.parametrize(
    "field,role", [("completion", "user"), ("prompt", "assistant"), ("prompt", "tool")]
)
def test_trace_rows_reject_invalid_role_semantics(tmp_path: Path, field: str, role: str) -> None:
    import json

    messages = {
        "prompt": [dict(message) for message in PROMPT],
        "completion": [{"role": "assistant", "content": "answer"}],
    }
    messages[field][-1]["role"] = role
    row = {**messages, "db_id": "concert", "question": "q"}
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        read_trace_rows(path)


def test_legacy_trace_lineage_remains_unknown(tmp_path: Path) -> None:
    import json

    row = {
        "prompt": [dict(message) for message in PROMPT],
        "completion": [{"role": "assistant", "content": "answer"}],
        "db_id": "concert",
        "question": "q",
    }
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert read_trace_rows(path)[0].source_example is None
    with pytest.raises(ValueError, match="requires source_example"):
        read_trace_rows(path, require_train_lineage=True)


def test_new_trace_rows_preserve_train_source(scoring_root: Path, tmp_path: Path) -> None:

    candidates = _candidates(WF_CORRECT)
    traces, _ = filter_traces(candidates, scoring_root)
    path = write_trace_dataset(traces, tmp_path / "traces.jsonl")
    assert (
        read_trace_rows(path, require_train_lineage=True)[0].source_example == candidates[0].example
    )


def test_sql_grpo_rows_reject_non_train_examples() -> None:
    from llm_rlvr.tasks.sql import build_grpo_rows

    dev = Example(db_id="concert", question="q", gold_sql="SELECT 1", split="dev")
    with pytest.raises(ValueError, match="train-split examples only"):
        build_grpo_rows([dev], {})
