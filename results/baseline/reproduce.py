"""Re-derive every committed number — baseline, SFT, RL headline, ablation — from the
CompletionSets (CPU, no model).

Run via `make eval` (needs the databases: `make data` for Spider; BIRD is fetched here). Every
number is a deterministic re-score of a committed `CompletionSet`, so a clean checkout reproduces
the flat table AND the base → SFT → RL progression tables (seed-level CIs, gain bar, single-DB
caveat) exactly. Scoring is threaded (`workers`): identical results at any worker count.
"""

from __future__ import annotations

from pathlib import Path

from text2sql_rlvr.data import ensure_bird_dev, ensure_spider_databases
from text2sql_rlvr.eval import read_completion_set, score_completion_set
from text2sql_rlvr.report import build_rl_row, render_progression

HERE = Path(__file__).parent
SFT = HERE.parent / "sft"
GRPO = HERE.parent / "grpo"
WORKERS = 8
GRPO_SEEDS = (0, 1, 2)


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
    for seed in GRPO_SEEDS:
        shaped = f"grpo-s{seed} (shaped)"
        rows.append(("Spider-dev", shaped, GRPO / f"spider-dev-zs-grpo-s{seed}.json", spider_db))
        rows.append(("BIRD-dev", shaped, GRPO / f"bird-dev-zs-grpo-s{seed}.json", bird_db))
    nb = "grpo-nb-s0 (near-bin)"
    rows.append(("Spider-dev", nb, GRPO / "spider-dev-zs-grpo-nb-s0.json", spider_db))
    rows.append(("BIRD-dev", nb, GRPO / "bird-dev-zs-grpo-nb-s0.json", bird_db))

    ex: dict[tuple[str, str], float] = {}
    print(f"{'benchmark':<12} {'setting':<22} {'EX':>7} {'well-formed':>12}  correct/n")
    for benchmark, setting, path, db_root in rows:
        report = score_completion_set(read_completion_set(path), db_root, workers=WORKERS)
        ex[(benchmark, setting)] = report.execution_accuracy
        print(
            f"{benchmark:<12} {setting:<22} {report.execution_accuracy:>7.4f} "
            f"{report.well_formed_rate:>12.4f}  {report.n_correct}/{report.n}"
        )

    # The base -> SFT -> RL progression (seed-level CIs over the headline seeds, the near-binary
    # ablation as a single-seed point estimate, the single-DB caveat disclosed).
    for benchmark in ("Spider-dev", "BIRD-dev"):
        base = ex[(benchmark, "cot zero-shot")]
        sft = ex[(benchmark, "cot zs + sft-r32")]
        headline = [ex[(benchmark, f"grpo-s{seed} (shaped)")] for seed in GRPO_SEEDS]
        ablation = [ex[(benchmark, "grpo-nb-s0 (near-bin)")]]
        print()
        print(
            render_progression(
                benchmark,
                base_ex=base,
                sft_ex=sft,
                rl_rows=[
                    build_rl_row(
                        "RL 5e-5 shaped (3 seeds)", benchmark, sft_anchor=sft, per_seed_ex=headline
                    ),
                    build_rl_row(
                        "RL 5e-5 near-binary (1)", benchmark, sft_anchor=sft, per_seed_ex=ablation
                    ),
                ],
                single_db=True,
            )
        )


if __name__ == "__main__":
    main()
