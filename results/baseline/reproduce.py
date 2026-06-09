"""Re-derive the committed base-model baseline numbers from the CompletionSets (CPU, no model).

Run via `make eval` (needs the databases: `make data` for Spider; BIRD is fetched here). Every
number is a deterministic re-score of a committed `CompletionSet`, so a clean checkout reproduces
the table exactly.
"""

from __future__ import annotations

from pathlib import Path

from text2sql_rlvr.data import ensure_bird_dev, ensure_spider_databases
from text2sql_rlvr.eval import read_completion_set, score_completion_set

HERE = Path(__file__).parent


def main() -> None:
    spider_db = ensure_spider_databases().database_root
    bird_db = ensure_bird_dev().database_root
    rows = [
        ("Spider-dev", "cot zero-shot", "spider-dev-zs.json", spider_db),
        ("Spider-dev", "cot few-shot k=3", "spider-dev-fs3.json", spider_db),
        ("Spider-dev", "direct zero-shot", "spider-dev-zs-direct.json", spider_db),
        ("BIRD-dev", "cot zero-shot", "bird-dev-zs.json", bird_db),
        ("BIRD-dev", "direct zero-shot", "bird-dev-zs-direct.json", bird_db),
    ]
    print(f"{'benchmark':<12} {'setting':<18} {'EX':>7}  correct/n")
    for benchmark, setting, filename, db_root in rows:
        report = score_completion_set(read_completion_set(HERE / filename), db_root)
        print(
            f"{benchmark:<12} {setting:<18} {report.execution_accuracy:>7.4f}  "
            f"{report.n_correct}/{report.n}"
        )


if __name__ == "__main__":
    main()
