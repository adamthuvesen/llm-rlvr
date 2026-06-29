"""`llm-rlvr` CLI: `generate` (the only model-loading step), `score`, and `build-traces`
(both model-free).

    llm-rlvr generate --model Qwen/Qwen2.5-Coder-3B-Instruct --split dev \
        --tables-json data/spider/tables.json --backend transformers --out runs/baseline
    llm-rlvr score --completions runs/baseline --db-root data/spider/database \
        --out results/baseline.json
    llm-rlvr build-traces --candidates runs/sft/candidates.json \
        --db-root data/spider/spider_data/database --out data/sft/traces.jsonl \
        --manifest results/sft/traces.manifest.json
"""

from __future__ import annotations

import argparse
import random
import sys

from llm_rlvr.eval.completions import read_completion_set, write_completion_set, write_json
from llm_rlvr.eval.score import score_completion_set
from llm_rlvr.eval.verifier import VerifierConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm-rlvr",
        description="Text-to-SQL eval and Stage-3 support utilities.",
        epilog=(
            "Core commands (Stages 0-2): generate, score, build-traces\n"
            "Stage-3 support: build-tooluse, select-bestofn, fr-metrics, fr-error-only-replay"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="sample the model and write a CompletionSet")
    gen.add_argument("--model", required=True)
    gen.add_argument("--model-revision", default=None)
    # eval-only splits: never `train`, so the few-shot pool can't overlap the eval set (R2).
    gen.add_argument("--split", choices=["dev", "validation"], default="dev")
    gen.add_argument("--tables-json", required=True, help="Spider tables.json for schemas")
    gen.add_argument(
        "--db-archive-sha256",
        default=None,
        help="checksum of the Spider DB archive, recorded in provenance (5.4)",
    )
    gen.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    gen.add_argument("--adapter", default=None, help="LoRA adapter dir to load over the base")
    gen.add_argument("--out", required=True, help="output CompletionSet path")
    gen.add_argument("--few-shot-k", type=int, default=0)
    gen.add_argument("--few-shot-seed", type=int, default=0)
    gen.add_argument("--limit", type=int, default=None, help="deterministic dev slice size")
    gen.add_argument("--slice-seed", type=int, default=0)
    gen.add_argument("--max-new-tokens", type=int, default=1024)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--prompt-style", choices=["cot", "direct"], default="cot")
    gen.set_defaults(func=_cmd_generate)

    score = sub.add_parser("score", help="score a CompletionSet into execution accuracy")
    score.add_argument("--completions", required=True, help="CompletionSet json path")
    score.add_argument("--db-root", required=True, help="extracted Spider database/ directory")
    score.add_argument("--out", default=None, help="optional ScoreReport json path")
    score.add_argument("--no-keep-distinct", action="store_true")
    score.add_argument(
        "--workers", type=int, default=1, help="scoring threads (results identical at any count)"
    )
    score.set_defaults(func=_cmd_score)

    traces = sub.add_parser(
        "build-traces", help="filter sampled candidates into the SFT trace dataset (CPU)"
    )
    traces.add_argument("--candidates", required=True, help="CandidateSet json from the GPU run")
    traces.add_argument("--db-root", required=True, help="extracted Spider database/ directory")
    traces.add_argument("--out", required=True, help="output trace-dataset JSONL path")
    traces.add_argument("--manifest", required=True, help="output manifest json path")
    traces.set_defaults(func=_cmd_build_traces)

    tooluse = sub.add_parser(
        "build-tooluse",
        help="[stage-3] build the tool-use SFT dataset (CPU; executes in the sandbox)",
    )
    tooluse.add_argument("--tables-json", required=True, help="Spider tables.json for schemas")
    tooluse.add_argument("--db-root", required=True, help="extracted Spider database/ directory")
    tooluse.add_argument("--traces", required=True, help="Stage-1 trace JSONL (direct-row source)")
    tooluse.add_argument("--out", required=True, help="output tool-use dataset JSONL path")
    tooluse.add_argument("--manifest", required=True, help="output manifest json path")
    tooluse.add_argument("--n-revise", type=int, default=600)
    tooluse.add_argument("--n-check", type=int, default=300)
    tooluse.add_argument("--n-direct", type=int, default=200)
    tooluse.add_argument("--seed", type=int, default=0)
    tooluse.set_defaults(func=_cmd_build_tooluse)

    bestofn = sub.add_parser(
        "select-bestofn",
        help="[stage-3] gold-free best-of-N selection over sampled CompletionSets (CPU)",
    )
    bestofn.add_argument(
        "--completions", required=True, nargs="+", help="N aligned sampled CompletionSet paths"
    )
    bestofn.add_argument("--db-root", required=True, help="extracted database/ directory")
    bestofn.add_argument("--out", required=True, help="output selection CompletionSet path")
    bestofn.add_argument("--report", required=True, help="output selection-report json path")
    bestofn.set_defaults(func=_cmd_select_bestofn)

    fr_metrics = sub.add_parser(
        "fr-metrics",
        help="[stage-3] fix/harm decomposition of a forced-revision eval (CPU)",
    )
    fr_metrics.add_argument("--completions", required=True, help="FR CompletionSet json path")
    fr_metrics.add_argument("--transcripts", required=True, help="FR transcripts json path")
    fr_metrics.add_argument("--db-root", required=True, help="extracted database/ directory")
    fr_metrics.add_argument("--out", default=None, help="optional FrMetricsReport json path")
    fr_metrics.set_defaults(func=_cmd_fr_metrics)

    fr_error_only = sub.add_parser(
        "fr-error-only-replay",
        help="[stage-3] replay forced-revision eval with second generation on errors only",
    )
    fr_error_only.add_argument("--completions", required=True, help="FR CompletionSet json path")
    fr_error_only.add_argument("--transcripts", required=True, help="FR transcripts json path")
    fr_error_only.add_argument("--db-root", required=True, help="extracted database/ directory")
    fr_error_only.add_argument("--out", required=True, help="output replay CompletionSet path")
    fr_error_only.add_argument("--report", required=True, help="output replay report json path")
    fr_error_only.set_defaults(func=_cmd_fr_error_only_replay)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _cmd_score(args: argparse.Namespace) -> int:
    completion_set = read_completion_set(args.completions)
    config = VerifierConfig(keep_distinct=not args.no_keep_distinct)
    report = score_completion_set(completion_set, args.db_root, config=config, workers=args.workers)
    if args.out:
        write_json(report, args.out)
    print(
        f"execution_accuracy={report.execution_accuracy:.4f} "
        f"well_formed_rate={report.well_formed_rate:.4f} "
        f"({report.n_correct}/{report.n}; gold_errors={report.n_gold_errors})"
    )
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    # Imported here so `score` never pays for (or requires) the data/generation imports.
    from llm_rlvr.data.artifacts import SPIDER_DB_ARCHIVE
    from llm_rlvr.data.spider import SPIDER_HF_REVISION, load_examples, load_schemas
    from llm_rlvr.eval.generate import generate
    from llm_rlvr.eval.provenance import DecodingConfig

    db_archive_sha256 = args.db_archive_sha256 or SPIDER_DB_ARCHIVE.sha256
    schemas = load_schemas(args.tables_json)
    examples = list(load_examples(args.split))
    if args.limit is not None:
        examples = _deterministic_slice(examples, args.limit, args.slice_seed)
    train_pool = list(load_examples("train")) if args.few_shot_k else []

    completion_set = generate(
        examples,
        schemas,
        model=args.model,
        backend=args.backend,
        decoding=DecodingConfig(temperature=args.temperature, max_new_tokens=args.max_new_tokens),
        dataset="spider",
        dataset_revision=SPIDER_HF_REVISION,
        split=args.split,
        model_revision=args.model_revision,
        db_archive_sha256=db_archive_sha256,
        train_pool=train_pool,
        few_shot_k=args.few_shot_k,
        few_shot_seed=args.few_shot_seed,
        prompt_style=args.prompt_style,
        adapter_path=args.adapter,
    )
    path = write_completion_set(completion_set, args.out)
    print(f"wrote {len(completion_set.items)} completions -> {path}")
    return 0


def _cmd_build_traces(args: argparse.Namespace) -> int:
    # Imported here so `generate`/`score` never pay for the train-layer imports.
    from llm_rlvr.train.traces import (
        build_trace_manifest,
        filter_traces,
        read_candidate_set,
        write_trace_dataset,
    )

    candidate_set = read_candidate_set(args.candidates)
    kept, counts = filter_traces(candidate_set.items, args.db_root)
    path = write_trace_dataset(kept, args.out)
    write_json(build_trace_manifest(candidate_set, counts), args.manifest)
    print(
        f"kept {counts.n_kept}/{counts.n_questions} questions "
        f"({counts.n_candidates} candidates, {counts.n_eligible} eligible, "
        f"{counts.n_gold_errors} gold errors) -> {path}"
    )
    return 0


def _cmd_build_tooluse(args: argparse.Namespace) -> int:
    # Imported here so `generate`/`score` never pay for the train-layer imports.
    from llm_rlvr.data.spider import SPIDER_HF_REVISION, load_examples, load_schemas
    from llm_rlvr.env.text2sql_env import FeedbackConfig
    from llm_rlvr.train.tooluse import (
        ToolUseMix,
        build_tooluse_manifest,
        build_tooluse_rows,
        write_tooluse_dataset,
    )
    from llm_rlvr.train.traces import read_trace_dataset

    mix = ToolUseMix(
        n_revise=args.n_revise, n_check=args.n_check, n_direct=args.n_direct, seed=args.seed
    )
    feedback = FeedbackConfig()
    schemas = load_schemas(args.tables_json)
    examples = list(load_examples("train"))
    traces = read_trace_dataset(args.traces)
    rows, counts = build_tooluse_rows(
        examples, schemas, args.db_root, traces, mix=mix, feedback=feedback
    )
    path = write_tooluse_dataset(rows, args.out)
    manifest = build_tooluse_manifest(
        dataset="spider",
        dataset_revision=SPIDER_HF_REVISION,
        mix=mix,
        feedback=feedback,
        trace_dataset_path=args.traces,
        counts=counts,
    )
    write_json(manifest, args.manifest)
    print(
        f"built {counts.n_rows} rows ({counts.n_revise} revise / {counts.n_check} check / "
        f"{counts.n_direct} direct; {counts.n_gold_errors} gold errors, "
        f"{counts.n_corruption_fallbacks} corruption fallbacks) -> {path}"
    )
    return 0


def _cmd_select_bestofn(args: argparse.Namespace) -> int:
    from llm_rlvr.eval.bestofn import select_best_of_n

    sets = [read_completion_set(path) for path in args.completions]
    selection, report = select_best_of_n(sets, args.db_root)
    path = write_completion_set(selection, args.out)
    write_json(report, args.report)
    print(
        f"selected {report.n_items} of {report.n_candidates_per_item} candidates each "
        f"(non-empty {report.n_selected_nonempty}, executes {report.n_selected_executes}, "
        f"fallback {report.n_fallback_first}, "
        f"disagreeing {report.n_items_with_disagreeing_candidates}) -> {path}"
    )
    return 0


def _cmd_fr_metrics(args: argparse.Namespace) -> int:
    from pathlib import Path

    from llm_rlvr.eval.forced_revision_generate import ForcedRevisionTranscriptSet
    from llm_rlvr.eval.fr_metrics import compute_fr_metrics

    completion_set = read_completion_set(args.completions)
    transcripts = ForcedRevisionTranscriptSet.model_validate_json(
        Path(args.transcripts).read_text(encoding="utf-8")
    )
    report = compute_fr_metrics(completion_set, transcripts, args.db_root)
    if args.out:
        write_json(report, args.out)
    print(
        f"attempt_ex={report.attempt_ex:.4f} final_ex={report.final_ex:.4f} "
        f"net={report.net_delta:+.4f} | fixed {report.n_fixed}/{report.n_attempt_wrong} "
        f"({report.fix_rate:.1%}; err-fb {report.n_fixed_from_error_feedback}, "
        f"result-fb {report.n_fixed_from_result_feedback}) | "
        f"harmed {report.n_harmed}/{report.n_attempt_right} ({report.harm_rate:.1%}) | "
        f"revised {report.n_revised}/{report.n}"
    )
    return 0


def _cmd_fr_error_only_replay(args: argparse.Namespace) -> int:
    from pathlib import Path

    from llm_rlvr.eval.forced_revision_generate import (
        ForcedRevisionTranscriptSet,
        replay_error_only_revision,
    )

    completion_set = read_completion_set(args.completions)
    transcripts = ForcedRevisionTranscriptSet.model_validate_json(
        Path(args.transcripts).read_text(encoding="utf-8")
    )
    replay, report = replay_error_only_revision(completion_set, transcripts, args.db_root)
    path = write_completion_set(replay, args.out)
    write_json(report, args.report)
    print(
        f"attempt_ex={report.attempt_ex:.4f} always_revise_ex={report.always_revise_ex:.4f} "
        f"error_only_ex={report.error_only_ex:.4f} | revised {report.n_revised}/{report.n} "
        f"({report.revised_rate:.1%}) | fixed {report.n_fixed}, harmed {report.n_harmed} "
        f"-> {path}"
    )
    return 0


def _deterministic_slice(examples: list, limit: int, seed: int) -> list:
    if limit < 1:
        raise ValueError(f"--limit must be >= 1, got {limit}")
    if limit >= len(examples):
        return examples
    return random.Random(seed).sample(examples, limit)


if __name__ == "__main__":
    raise SystemExit(main())
