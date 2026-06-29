"""Re-derive every committed number — baseline, SFT, RL headline, ablation — from the
CompletionSets (CPU, no model).

Run via `make eval` (needs the databases: `make data` for Spider; BIRD is fetched here). Every
number is a deterministic re-score of a committed `CompletionSet`, so a clean checkout reproduces
the flat table AND the base → SFT → RL progression tables (seed-level CIs, gain bar, single-DB
caveat) exactly. Scoring is threaded (`workers`): identical results at any worker count.
"""

from __future__ import annotations

import sys
from pathlib import Path

from llm_rlvr_sql.data import ensure_bird_dev, ensure_spider_databases
from llm_rlvr_sql.eval import read_completion_set, score_completion_set
from llm_rlvr_sql.report import build_rl_row, render_progression

HERE = Path(__file__).parent
RESULTS = HERE.parent
GRPO = RESULTS / "grpo"
WORKERS = 8

_RESULTS_ROOT = RESULTS
if str(_RESULTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESULTS_ROOT))
from manifest import DbKind, GRPO_SEEDS, repro_manifest  # noqa: E402, I001


def main() -> None:
    spider_db = ensure_spider_databases().database_root
    bird_db = ensure_bird_dev().database_root
    db_roots = {DbKind.SPIDER: spider_db, DbKind.BIRD: bird_db}
    rows = [
        (row.benchmark, row.setting, row.completion_path, db_roots[row.db])
        for row in repro_manifest()
    ]

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

    # The cross-family control: Llama-3.2-3B under the identical recipe (1 seed, no CI claim).
    for benchmark in ("Spider-dev", "BIRD-dev"):
        llama_sft = ex[(benchmark, "llama + sft-llama-r32")]
        print()
        print(
            render_progression(
                f"{benchmark} — Llama-3.2-3B cross-family control",
                base_ex=ex[(benchmark, "llama base cot zs")],
                sft_ex=llama_sft,
                rl_rows=[
                    build_rl_row(
                        "RL 5e-5 shaped (1 seed)",
                        benchmark,
                        sft_anchor=llama_sft,
                        per_seed_ex=[ex[(benchmark, "llama + grpo-llama-s0")]],
                    )
                ],
                single_db=True,
            )
        )

    _stage3b_forced_revision(spider_db, bird_db)


def _stage3b_forced_revision(spider_db, bird_db) -> None:
    """Stage 3b (grpo-forced-revision, ABORTED at the pre-registered pilot gate): re-derive
    the n=200 pilot-checkpoint documentation table — FR harness vs Stage-2 anchor vs the
    gold-free best-of-2 control — plus the fix/harm decomposition. Point estimates on a
    200-item slice; differences of 1-3 items are inside noise and no claim rests on them."""
    from llm_rlvr_sql.eval import CompletionSet
    from llm_rlvr_sql.eval.bestofn import select_best_of_n
    from llm_rlvr_sql.eval.forced_revision_generate import (
        ForcedRevisionTranscriptSet,
        replay_error_only_revision,
    )
    from llm_rlvr_sql.eval.fr_metrics import (
        compute_forced_revision_eval_report,
        compute_fr_metrics,
    )

    fr = HERE.parent / "grpo-fr"
    print()
    print("Stage 3b — forced revision (pilot checkpoint, n=200 Spider-dev; ABORTED at gate)")

    s2 = read_completion_set(GRPO / "spider-dev-zs-grpo-s0.json")
    s2_slice = CompletionSet(provenance=s2.provenance, items=s2.items[:200])
    rows = [
        ("Stage-2 grpo-s0 single-turn", s2_slice),
        (
            "FR-pilot single-turn greedy",
            read_completion_set(fr / "spider-dev-zs-grpo-fr-pilot-n200.json"),
        ),
        (
            "FR-pilot forced-revision (2 gen)",
            read_completion_set(fr / "spider-dev-fr-grpo-fr-pilot-n200.json"),
        ),
    ]
    transcripts = ForcedRevisionTranscriptSet.model_validate_json(
        (fr / "spider-dev-fr-grpo-fr-pilot-n200.transcripts.json").read_text(encoding="utf-8")
    )
    error_only, error_only_report = replay_error_only_revision(
        read_completion_set(fr / "spider-dev-fr-grpo-fr-pilot-n200.json"), transcripts, spider_db
    )
    rows.append(("FR-pilot error-only revision", error_only))
    sampled = [
        read_completion_set(fr / f"spider-dev-zs-grpo-fr-pilot-t1.0-seed{seed}-n200.json")
        for seed in (0, 1)
    ]
    selection, bo2 = select_best_of_n(sampled, spider_db)
    rows.append(("FR-pilot best-of-2 (T=1.0, gold-free)", selection))
    for setting, completion_set in rows:
        report = score_completion_set(completion_set, spider_db, workers=WORKERS)
        print(f"  {setting:<38} {report.execution_accuracy:>7.4f}  {report.n_correct}/{report.n}")
    print(
        f"  best-of-2 selection (gold-free): nonempty {bo2.n_selected_nonempty}, "
        f"executes {bo2.n_selected_executes}, fallback {bo2.n_fallback_first}"
    )
    metrics = compute_fr_metrics(
        read_completion_set(fr / "spider-dev-fr-grpo-fr-pilot-n200.json"), transcripts, spider_db
    )
    print(
        f"  revision pass: attempt {metrics.attempt_ex:.4f} -> final {metrics.final_ex:.4f} "
        f"(net {metrics.net_delta:+.4f}); fixed {metrics.n_fixed}/{metrics.n_attempt_wrong} "
        f"({metrics.fix_rate:.1%}), harmed {metrics.n_harmed}/{metrics.n_attempt_right} "
        f"({metrics.harm_rate:.1%}), revised {metrics.n_revised}/{metrics.n}"
    )
    print(
        f"  error-only replay: revised {error_only_report.n_revised}/{error_only_report.n} "
        f"({error_only_report.revised_rate:.1%}), fixed {error_only_report.n_fixed}, "
        f"harmed {error_only_report.n_harmed}"
    )

    fresh_error_only = read_completion_set(fr / "spider-dev-fr-error-only-grpo-fr-pilot.json")
    fresh_transcripts = ForcedRevisionTranscriptSet.model_validate_json(
        (fr / "spider-dev-fr-error-only-grpo-fr-pilot.transcripts.json").read_text(encoding="utf-8")
    )
    fresh_score = score_completion_set(fresh_error_only, spider_db, workers=WORKERS)
    fresh_report = compute_forced_revision_eval_report(
        fresh_error_only, fresh_transcripts, spider_db, policy="error_only"
    )
    print(
        f"  fresh full Spider error-only: attempt {fresh_report.attempt_ex:.4f} -> "
        f"final {fresh_score.execution_accuracy:.4f} ({fresh_score.n_correct}/{fresh_score.n}); "
        f"second generations {fresh_report.n_second_generation}/{len(fresh_transcripts.items)} "
        f"({fresh_report.second_generation_rate:.1%}), fixed {fresh_report.n_fixed}, "
        f"harmed {fresh_report.n_harmed}"
    )

    bird_error_only = read_completion_set(fr / "bird-dev-fr-error-only-grpo-fr-pilot.json")
    bird_transcripts = ForcedRevisionTranscriptSet.model_validate_json(
        (fr / "bird-dev-fr-error-only-grpo-fr-pilot.transcripts.json").read_text(encoding="utf-8")
    )
    bird_score = score_completion_set(bird_error_only, bird_db, workers=WORKERS)
    bird_report = compute_forced_revision_eval_report(
        bird_error_only, bird_transcripts, bird_db, policy="error_only"
    )
    gold_note = (
        f"; local gold-query timeouts {bird_score.n_gold_errors}"
        if bird_score.n_gold_errors
        else ""
    )
    print(
        f"  fresh full BIRD error-only: attempt {bird_report.attempt_ex:.4f} -> "
        f"final {bird_score.execution_accuracy:.4f} ({bird_score.n_correct}/{bird_score.n}"
        f"{gold_note}); second generations {bird_report.n_second_generation}/"
        f"{len(bird_transcripts.items)} ({bird_report.second_generation_rate:.1%}), "
        f"fixed {bird_report.n_fixed}, harmed {bird_report.n_harmed}"
    )


if __name__ == "__main__":
    main()
