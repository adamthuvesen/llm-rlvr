"""`text2sql-rlvr` CLI: `generate` (the only model-loading step), `score`, and `build-traces`
(both model-free).

    text2sql-rlvr generate --model Qwen/Qwen2.5-Coder-3B-Instruct --split dev \
        --tables-json data/spider/tables.json --backend transformers --out runs/baseline
    text2sql-rlvr score --completions runs/baseline --db-root data/spider/database \
        --out results/baseline.json
    text2sql-rlvr build-traces --candidates runs/sft/candidates.json \
        --db-root data/spider/spider_data/database --out data/sft/traces.jsonl \
        --manifest results/sft/traces.manifest.json
"""

from __future__ import annotations

import argparse
import random
import sys

from text2sql_rlvr.eval.completions import read_completion_set, write_completion_set, write_json
from text2sql_rlvr.eval.score import score_completion_set
from text2sql_rlvr.eval.verifier import VerifierConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="text2sql-rlvr")
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
    score.set_defaults(func=_cmd_score)

    traces = sub.add_parser(
        "build-traces", help="filter sampled candidates into the SFT trace dataset (CPU)"
    )
    traces.add_argument("--candidates", required=True, help="CandidateSet json from the GPU run")
    traces.add_argument("--db-root", required=True, help="extracted Spider database/ directory")
    traces.add_argument("--out", required=True, help="output trace-dataset JSONL path")
    traces.add_argument("--manifest", required=True, help="output manifest json path")
    traces.set_defaults(func=_cmd_build_traces)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _cmd_score(args: argparse.Namespace) -> int:
    completion_set = read_completion_set(args.completions)
    config = VerifierConfig(keep_distinct=not args.no_keep_distinct)
    report = score_completion_set(completion_set, args.db_root, config=config)
    if args.out:
        write_json(report, args.out)
    print(
        f"execution_accuracy={report.execution_accuracy:.4f} "
        f"({report.n_correct}/{report.n}; gold_errors={report.n_gold_errors})"
    )
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    # Imported here so `score` never pays for (or requires) the data/generation imports.
    from text2sql_rlvr.data.artifacts import SPIDER_DB_ARCHIVE
    from text2sql_rlvr.data.spider import SPIDER_HF_REVISION, load_examples, load_schemas
    from text2sql_rlvr.eval.generate import generate
    from text2sql_rlvr.eval.provenance import DecodingConfig

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
    )
    path = write_completion_set(completion_set, args.out)
    print(f"wrote {len(completion_set.items)} completions -> {path}")
    return 0


def _cmd_build_traces(args: argparse.Namespace) -> int:
    # Imported here so `generate`/`score` never pay for the train-layer imports.
    from text2sql_rlvr.train.traces import (
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


def _deterministic_slice(examples: list, limit: int, seed: int) -> list:
    if limit < 1:
        raise ValueError(f"--limit must be >= 1, got {limit}")
    if limit >= len(examples):
        return examples
    return random.Random(seed).sample(examples, limit)


if __name__ == "__main__":
    raise SystemExit(main())
