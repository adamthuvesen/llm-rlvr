"""Write CPU fixtures for the real CLI: a SQLite root, a CompletionSet, and a CandidateSet.

Run by `verify` as `python fixtures.py <dir>`. The CompletionSet and its database are the
`make demo` ones, so the drive and the demo cannot drift apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

from llm_rlvr.artifacts.io import atomic_write_json
from llm_rlvr.data.models import Example
from llm_rlvr.train.traces import CandidateSet, TraceCandidates, TraceSamplingConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))
from no_gpu_demo import _build_demo_db, _completion_set

PROMPT = (
    {"role": "system", "content": "instruction"},
    {"role": "user", "content": "Database schema:\n...\n\nQuestion: who sold 120 tickets?"},
)
GOLD = "SELECT artist FROM shows WHERE tickets_sold = 120"
CANDIDATES = (
    f"<think>filter on tickets</think>\n```sql\n{GOLD}\n```",
    "<think>guess</think>\n```sql\nSELECT artist FROM shows WHERE tickets_sold = 80\n```",
    f"```sql\n{GOLD}\n```",
)


def main(out: Path) -> None:
    _build_demo_db(out / "database")
    atomic_write_json(_completion_set(), out / "completions.json")
    example = Example(
        db_id="concert", question="Who sold 120 tickets?", gold_sql=GOLD, split="train"
    )
    candidates = CandidateSet(
        base_model="Qwen/Qwen2.5-Coder-3B-Instruct",
        base_model_revision="fixture",
        sampling=TraceSamplingConfig(n=len(CANDIDATES)),
        dataset="spider",
        dataset_revision="fixture",
        db_archive_sha256="0" * 64,
        items=(TraceCandidates(example=example, prompt=PROMPT, texts=CANDIDATES),),
    )
    atomic_write_json(candidates, out / "candidates.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
