"""Tiny local verifier demo.

Builds a disposable SQLite database, scores two fixed completions with the same
execution-match path used by `make eval`, and prints the aggregate result.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

from text2sql_rlvr.data.models import Example
from text2sql_rlvr.eval.completions import (
    CompletionSet,
    GenerationSample,
    ProblemCompletions,
)
from text2sql_rlvr.eval.provenance import DecodingConfig, GenerationProvenance
from text2sql_rlvr.eval.score import score_completion_set


def _build_demo_db(db_root: Path) -> None:
    db_path = db_root / "concert" / "concert.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE shows (id INTEGER PRIMARY KEY, artist TEXT, tickets_sold INTEGER);
        INSERT INTO shows (artist, tickets_sold) VALUES
          ('Ada SQL', 120),
          ('Grace Joins', 80);
        """
    )
    conn.commit()
    conn.close()


def _completion_set() -> CompletionSet:
    examples = (
        Example(
            db_id="concert",
            question="Which artist sold 120 tickets?",
            gold_sql="SELECT artist FROM shows WHERE tickets_sold = 120",
            split="demo",
        ),
        Example(
            db_id="concert",
            question="How many shows sold at least 80 tickets?",
            gold_sql="SELECT COUNT(*) FROM shows WHERE tickets_sold >= 80",
            split="demo",
        ),
    )
    sqls = (
        "SELECT artist FROM shows WHERE tickets_sold = 120",
        "SELECT COUNT(*) FROM shows WHERE tickets_sold > 80",
    )
    items = tuple(
        ProblemCompletions(
            example=example,
            samples=(GenerationSample(text=sql, sql=sql),),
        )
        for example, sql in zip(examples, sqls, strict=True)
    )
    provenance = GenerationProvenance(
        model="fixed-demo-completions",
        backend="none",
        dataset="local-demo",
        dataset_revision="none",
        split="demo",
        n_examples=len(items),
        decoding=DecodingConfig(),
    )
    return CompletionSet(provenance=provenance, items=items)


def main() -> int:
    with TemporaryDirectory(prefix="rlvr-sql-demo-") as tmp:
        db_root = Path(tmp) / "database"
        _build_demo_db(db_root)
        report = score_completion_set(_completion_set(), db_root)

    print(
        f"execution_accuracy={report.execution_accuracy:.4f} "
        f"well_formed_rate={report.well_formed_rate:.4f} "
        f"({report.n_correct}/{report.n}; gold_errors={report.n_gold_errors})"
    )
    print(f"reasons={report.reasons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
