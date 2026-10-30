"""CLI smoke: score subcommand on a tiny in-memory completion set."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_rlvr.cli import main


def test_score_cli_writes_report(tmp_path: Path, scoring_root: Path, provenance) -> None:
    from llm_rlvr.data.models import Example
    from llm_rlvr.eval.completions import (
        CompletionSet,
        GenerationSample,
        ProblemCompletions,
        write_completion_set,
    )

    example = Example(
        db_id="concert",
        question="List ids where b is x",
        gold_sql="SELECT a FROM t WHERE b = 'x'",
        split="dev",
    )
    completion_set = CompletionSet(
        provenance=provenance.model_copy(update={"n_examples": 1}),
        items=(
            ProblemCompletions(
                example=example,
                samples=(GenerationSample(text="```sql\nSELECT a FROM t WHERE b = 'x'\n```"),),
            ),
        ),
    )
    completions = tmp_path / "completions.json"
    out = tmp_path / "score.json"
    write_completion_set(completion_set, completions)

    code = main(
        [
            "score",
            "--completions",
            str(completions),
            "--db-root",
            str(scoring_root),
            "--out",
            str(out),
        ]
    )

    assert code == 0
    assert out.is_file()
    assert '"execution_accuracy"' in out.read_text(encoding="utf-8")


def _write_completion(tmp_path: Path, provenance, *, gold_sql: str = "SELECT 1") -> Path:
    from llm_rlvr.data.models import Example
    from llm_rlvr.eval.completions import (
        CompletionSet,
        GenerationSample,
        ProblemCompletions,
        write_completion_set,
    )

    record = CompletionSet(
        provenance=provenance.model_copy(update={"n_examples": 1}),
        items=(
            ProblemCompletions(
                example=Example(db_id="concert", question="q", gold_sql=gold_sql, split="dev"),
                samples=(GenerationSample(text="```sql\nSELECT 1\n```"),),
            ),
        ),
    )
    return write_completion_set(record, tmp_path / "completions.json")


def test_cli_gold_error_is_unhealthy_without_changing_denominator(
    tmp_path: Path, scoring_root: Path, provenance, capsys
) -> None:
    import json

    completions = _write_completion(tmp_path, provenance, gold_sql="SELECT missing FROM absent")
    out = tmp_path / "score.json"
    assert (
        main(
            [
                "score",
                "--completions",
                str(completions),
                "--db-root",
                str(scoring_root),
                "--out",
                str(out),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "denominator remains 1" in captured.err
    report = json.loads(out.read_text())
    assert report["n"] == 1
    assert report["n_gold_errors"] == 1


def test_cli_does_not_overwrite_existing_score_output(tmp_path: Path, monkeypatch, capsys) -> None:
    out = tmp_path / "existing.json"
    out.write_text("existing")
    monkeypatch.setattr("llm_rlvr.cli.read_completion_set", lambda *_: pytest.fail("input read"))
    assert main(["score", "--completions", "unused", "--db-root", "unused", "--out", str(out)]) == 1
    assert out.read_text() == "existing"
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "arguments",
    [["--limit", "0"], ["--max-new-tokens", "0"], ["--temperature", "-1"], ["--few-shot-k", "-1"]],
)
def test_cli_generation_preflight_precedes_data_loading(
    tmp_path: Path, monkeypatch, capsys, arguments: list[str]
) -> None:
    monkeypatch.setattr(
        "llm_rlvr.data.spider.load_examples", lambda *_: pytest.fail("data fetched")
    )
    assert (
        main(
            [
                "generate",
                "--model",
                "org/model",
                "--tables-json",
                "unused",
                "--out",
                str(tmp_path / "out.json"),
                *arguments,
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error:")


def test_cli_generation_records_selection_and_unknown_database_identity(
    tmp_path: Path, monkeypatch, provenance
) -> None:
    from llm_rlvr.data.models import Example
    from llm_rlvr.eval.completions import assemble_completion_set

    examples = [Example(db_id="concert", question="q", gold_sql="SELECT 1", split="dev")]
    seen = {}
    monkeypatch.setattr("llm_rlvr.data.spider.load_examples", lambda *_: examples)
    monkeypatch.setattr("llm_rlvr.data.spider.load_schemas", lambda *_: {})
    monkeypatch.setattr(
        "llm_rlvr.train.configuration.resolve_model_revision", lambda *_: "resolved"
    )

    def generate(rows, _schemas, **kwargs):
        seen.update(kwargs)
        return assemble_completion_set(
            rows, ["SELECT 1"], provenance.model_copy(update={"n_examples": 1})
        )

    monkeypatch.setattr("llm_rlvr.eval.generate.generate", generate)
    assert (
        main(
            [
                "generate",
                "--model",
                "org/model",
                "--tables-json",
                "unused",
                "--out",
                str(tmp_path / "out.json"),
                "--limit",
                "1",
                "--slice-seed",
                "7",
            ]
        )
        == 0
    )
    assert seen["db_archive_sha256"] is None
    assert seen["selection_policy"] == "seeded"
    assert seen["selection_seed"] == 7
    assert seen["model_revision"] == "resolved"


def test_cli_missing_optional_backend_has_actionable_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    def missing(_args):
        raise ModuleNotFoundError("No module named vllm", name="vllm")

    monkeypatch.setattr("llm_rlvr.cli._cmd_generate", missing)
    assert (
        main(
            [
                "generate",
                "--model",
                "org/model",
                "--tables-json",
                "unused",
                "--out",
                str(tmp_path / "out"),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "uv sync --locked --extra dev --extra gpu" in captured.err
    assert "Traceback" not in captured.err


def test_cli_build_traces_rejects_invalid_gold_without_outputs(
    tmp_path: Path, scoring_root: Path, capsys
) -> None:
    from llm_rlvr.data.models import Example
    from llm_rlvr.eval.completions import write_pydantic_json
    from llm_rlvr.train.traces import CandidateSet, TraceCandidates, TraceSamplingConfig

    candidates = CandidateSet(
        base_model="m",
        sampling=TraceSamplingConfig(n=1),
        dataset="spider",
        dataset_revision="rev",
        items=(
            TraceCandidates(
                example=Example(
                    db_id="concert",
                    question="q",
                    gold_sql="SELECT missing FROM absent",
                    split="train",
                ),
                prompt=({"role": "user", "content": "q"},),
                texts=("<think>r</think>\n```sql\nSELECT 1\n```",),
            ),
        ),
    )
    source = write_pydantic_json(candidates, tmp_path / "candidates.json")
    out, manifest = tmp_path / "traces.jsonl", tmp_path / "manifest.json"
    assert (
        main(
            [
                "build-traces",
                "--candidates",
                str(source),
                "--db-root",
                str(scoring_root),
                "--out",
                str(out),
                "--manifest",
                str(manifest),
            ]
        )
        == 1
    )
    assert not out.exists() and not manifest.exists()
    assert capsys.readouterr().out == ""


def test_cli_public_reproduction_rejects_unused_options(capsys) -> None:
    assert main(["reproduce", "--mode", "public", "--workers", "2"]) == 1
    assert "applies only" in capsys.readouterr().err


@pytest.mark.parametrize("requested_revision", [None, "matching-tag"])
def test_cli_adapter_generation_uses_pinned_base_and_records_actual_identity(
    tmp_path: Path, monkeypatch, requested_revision: str | None
) -> None:
    import json

    from llm_rlvr.artifacts.io import sha256_of
    from llm_rlvr.data.models import DatabaseSchema, Example
    from llm_rlvr.eval.completions import read_completion_set
    from llm_rlvr.train.configuration import inspect_adapter

    pinned = "a" * 40
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {"base_model_name_or_path": "org/model", "r": 8, "lora_alpha": 16, "lora_dropout": 0}
        )
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
    (adapter / "provenance.json").write_text(
        json.dumps({"base_model": "org/model", "base_model_revision": pinned})
    )
    identity = inspect_adapter(adapter, base_model="org/model", base_model_revision=None)
    seen = {}
    calls = []

    def resolve(_model, revision):
        calls.append(revision)
        return pinned if revision in {pinned, "matching-tag"} else "b" * 40

    def backend(_prompts, **kwargs):
        seen.update(kwargs)
        return ["```sql\nSELECT 1\n```"]

    monkeypatch.setattr("llm_rlvr.train.configuration.resolve_model_revision", resolve)
    monkeypatch.setattr(
        "llm_rlvr.data.spider.load_examples",
        lambda *_: (Example(db_id="concert", question="q", gold_sql="SELECT 1", split="dev"),),
    )
    monkeypatch.setattr(
        "llm_rlvr.data.spider.load_schemas",
        lambda *_: {"concert": DatabaseSchema(db_id="concert", tables=())},
    )
    monkeypatch.setattr("llm_rlvr.eval.generate._run_backend", backend)
    out = tmp_path / "generated.json"
    arguments = [
        "generate",
        "--model",
        "org/model",
        "--adapter",
        str(adapter),
        "--tables-json",
        "unused",
        "--out",
        str(out),
    ]
    if requested_revision is not None:
        arguments.extend(["--model-revision", requested_revision])
    assert main(arguments) == 0
    assert None not in calls
    assert seen["model_revision"] == pinned
    provenance = read_completion_set(out).provenance
    assert provenance.model_revision == pinned
    assert provenance.adapter_sha256 == identity.sha256
    assert provenance.adapter_provenance_sha256 == sha256_of(adapter / "provenance.json")


def test_cli_adapter_base_mismatch_rejected_before_generation(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import json

    from llm_rlvr.data.models import Example

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {"base_model_name_or_path": "org/model", "r": 8, "lora_alpha": 16, "lora_dropout": 0}
        )
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
    (adapter / "provenance.json").write_text(
        json.dumps({"base_model": "org/model", "base_model_revision": "a" * 40})
    )
    monkeypatch.setattr(
        "llm_rlvr.train.configuration.resolve_model_revision",
        lambda _model, revision: "b" * 40 if revision == "different" else "a" * 40,
    )
    monkeypatch.setattr(
        "llm_rlvr.data.spider.load_examples",
        lambda *_: (Example(db_id="concert", question="q", gold_sql="SELECT 1", split="dev"),),
    )
    monkeypatch.setattr("llm_rlvr.data.spider.load_schemas", lambda *_: {})
    monkeypatch.setattr(
        "llm_rlvr.eval.generate._run_backend", lambda *_args, **_kwargs: pytest.fail("model loaded")
    )
    out = tmp_path / "generated.json"
    assert (
        main(
            [
                "generate",
                "--model",
                "org/model",
                "--model-revision",
                "different",
                "--adapter",
                str(adapter),
                "--tables-json",
                "unused",
                "--out",
                str(out),
            ]
        )
        == 1
    )
    assert not out.exists()
    assert capsys.readouterr().out == ""
