"""Re-derive the committed baseline + SFT numbers from the CompletionSets (CPU, no model).

Run via `make eval` (needs the databases: `make data` for Spider; BIRD is fetched here). Every
number is a deterministic re-score of a committed `CompletionSet`, so a clean checkout reproduces
the table exactly. The `+ sft-r32` rows are the Stage-1 LoRA adapter (see `results/sft/`); the
success bar is a near-100% well-formed rate with no EX regression below the base cot anchor.
"""

from __future__ import annotations

from pathlib import Path

from text2sql_rlvr.data import ensure_bird_dev, ensure_spider_databases
from text2sql_rlvr.eval import read_completion_set, score_completion_set

HERE = Path(__file__).parent
SFT = HERE.parent / "sft"


def main() -> None:
    spider_db = ensure_spider_databases().database_root
    bird_db = ensure_bird_dev().database_root
    rows = [
        ("Spider-dev", "cot zero-shot", HERE / "spider-dev-zs.json", spider_db),
        ("Spider-dev", "cot few-shot k=3", HERE / "spider-dev-fs3.json", spider_db),
        ("Spider-dev", "direct zero-shot", HERE / "spider-dev-zs-direct.json", spider_db),
        ("Spider-dev", "cot zs + sft-r32", SFT / "spider-dev-zs-sft-r32.json", spider_db),
        ("Spider-dev", "direct zs + sft-r32", SFT / "spider-dev-zs-direct-sft-r32.json", spider_db),
        ("BIRD-dev", "cot zero-shot", HERE / "bird-dev-zs.json", bird_db),
        ("BIRD-dev", "direct zero-shot", HERE / "bird-dev-zs-direct.json", bird_db),
        ("BIRD-dev", "cot zs + sft-r32", SFT / "bird-dev-zs-sft-r32.json", bird_db),
    ]
    print(f"{'benchmark':<12} {'setting':<20} {'EX':>7} {'well-formed':>12}  correct/n")
    for benchmark, setting, path, db_root in rows:
        report = score_completion_set(read_completion_set(path), db_root)
        print(
            f"{benchmark:<12} {setting:<20} {report.execution_accuracy:>7.4f} "
            f"{report.well_formed_rate:>12.4f}  {report.n_correct}/{report.n}"
        )


if __name__ == "__main__":
    main()
