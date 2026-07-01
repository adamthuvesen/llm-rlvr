"""CLI smoke: score subcommand on a tiny in-memory completion set."""

from __future__ import annotations

from pathlib import Path

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
