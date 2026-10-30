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
import tempfile
from pathlib import Path

from pydantic import ValidationError

from llm_rlvr.artifacts.io import atomic_write_text, ensure_new_output
from llm_rlvr.data.paths import sha256_of
from llm_rlvr.eval.completions import (
    read_completion_set,
    write_completion_set,
    write_pydantic_json,
)
from llm_rlvr.eval.score import score_completion_set
from llm_rlvr.tasks.sql.verifier import VerifierConfig


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ModuleNotFoundError as exc:
        module = (exc.name or "").split(".")[0]
        extras = {
            "torch": "generate",
            "transformers": "generate",
            "vllm": "gpu",
            "peft": "train",
            "trl": "train",
            "accelerate": "train",
        }
        if module not in extras:
            raise
        print(
            f"error: {args.command} requires {module}. "
            f"Run: uv sync --locked --extra dev --extra {extras[module]}",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError, ValidationError, KeyError) as exc:
        if isinstance(exc, ValidationError):
            first = exc.errors(include_url=False)[0]
            message = f"{'.'.join(str(part) for part in first['loc'])}: {first['msg']}"
        else:
            message = str(exc)
        print(f"error: {message}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-rlvr",
        description="Text-to-SQL generation, scoring, and training-data utilities.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_generate_parser(sub)
    _add_score_parser(sub)
    _add_build_traces_parser(sub)
    reproduce = sub.add_parser(
        "reproduce", help="validate public evidence or re-score restored raw artifacts"
    )
    reproduce.add_argument("--mode", choices=["public", "full"], default="public")
    reproduce.add_argument("--root", type=Path, default=None)
    reproduce.add_argument("--spider-db-root", type=Path, default=None)
    reproduce.add_argument("--bird-db-root", type=Path, default=None)
    reproduce.add_argument("--workers", type=int, default=1)
    reproduce.set_defaults(func=_cmd_reproduce)
    return parser


def _add_generate_parser(sub: argparse._SubParsersAction) -> None:
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


def _add_score_parser(sub: argparse._SubParsersAction) -> None:
    score = sub.add_parser("score", help="score a CompletionSet into execution accuracy")
    score.add_argument("--completions", required=True, help="CompletionSet json path")
    score.add_argument("--db-root", required=True, help="extracted Spider database/ directory")
    score.add_argument("--out", default=None, help="optional ScoreReport json path")
    score.add_argument("--no-keep-distinct", action="store_true")
    score.add_argument(
        "--workers", type=int, default=1, help="scoring threads (results identical at any count)"
    )
    score.set_defaults(func=_cmd_score)


def _add_build_traces_parser(sub: argparse._SubParsersAction) -> None:
    traces = sub.add_parser(
        "build-traces", help="filter sampled candidates into the SFT trace dataset (CPU)"
    )
    traces.add_argument("--candidates", required=True, help="CandidateSet json from the GPU run")
    traces.add_argument("--db-root", required=True, help="extracted Spider database/ directory")
    traces.add_argument("--out", required=True, help="output trace-dataset JSONL path")
    traces.add_argument("--manifest", required=True, help="output manifest json path")
    traces.set_defaults(func=_cmd_build_traces)


def _cmd_score(args: argparse.Namespace) -> int:
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.out:
        ensure_new_output(args.out)
    completion_set = read_completion_set(args.completions)
    config = VerifierConfig(keep_distinct=not args.no_keep_distinct)
    report = score_completion_set(completion_set, args.db_root, config=config, workers=args.workers)
    if args.out:
        write_pydantic_json(report, args.out)
    if report.n_gold_errors:
        print(
            f"error: score report is unhealthy: {report.n_gold_errors} reference SQL errors. "
            f"The denominator remains {report.n}.",
            file=sys.stderr,
        )
        return 2
    print(
        f"execution_accuracy={report.execution_accuracy:.4f} "
        f"well_formed_rate={report.well_formed_rate:.4f} "
        f"({report.n_correct}/{report.n}; gold_errors={report.n_gold_errors})"
    )
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    # Imported here so `score` never pays for (or requires) the data/generation imports.
    from llm_rlvr.artifacts.provenance import DecodingConfig
    from llm_rlvr.data.spider import SPIDER_HF_REVISION, load_examples, load_schemas
    from llm_rlvr.eval.generate import generate
    from llm_rlvr.modal_helpers import GenerationRequest
    from llm_rlvr.train.configuration import resolve_adapter_base, resolve_model_revision

    GenerationRequest(
        base_model=args.model,
        model_revision=args.model_revision,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        few_shot_k=args.few_shot_k,
        few_shot_seed=args.few_shot_seed,
        prompt_style=args.prompt_style,
    )
    ensure_new_output(args.out)
    if args.adapter:
        from llm_rlvr.artifacts.runs import require_complete

        require_complete(args.adapter)
    db_archive_sha256 = args.db_archive_sha256
    schemas = load_schemas(args.tables_json)
    examples = list(load_examples(args.split))
    if args.limit is not None:
        examples = _deterministic_slice(examples, args.limit, args.slice_seed)
    train_pool = list(load_examples("train")) if args.few_shot_k else []

    adapter_identity = None
    if args.adapter:
        resolved_revision, adapter_identity = resolve_adapter_base(
            args.adapter, base_model=args.model, requested_revision=args.model_revision
        )
    else:
        resolved_revision = resolve_model_revision(args.model, args.model_revision)
    completion_set = generate(
        examples,
        schemas,
        model=args.model,
        backend=args.backend,
        decoding=DecodingConfig(temperature=args.temperature, max_new_tokens=args.max_new_tokens),
        dataset="spider",
        dataset_revision=SPIDER_HF_REVISION,
        split=args.split,
        model_revision=resolved_revision,
        db_archive_sha256=db_archive_sha256,
        train_pool=train_pool,
        few_shot_k=args.few_shot_k,
        few_shot_seed=args.few_shot_seed,
        prompt_style=args.prompt_style,
        adapter_path=args.adapter,
        adapter_sha256=adapter_identity.sha256 if adapter_identity else None,
        adapter_provenance_sha256=adapter_identity.provenance_sha256 if adapter_identity else None,
        selection_policy="seeded" if args.limit is not None else "all",
        selection_seed=args.slice_seed if args.limit is not None else None,
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

    ensure_new_output(args.out)
    ensure_new_output(args.manifest)
    if Path(args.out).resolve() == Path(args.manifest).resolve():
        raise ValueError("--out and --manifest must be different paths")
    candidate_set = read_candidate_set(args.candidates)
    kept, counts = filter_traces(candidate_set.items, args.db_root)
    if counts.n_gold_errors:
        raise ValueError(
            f"trace dataset has {counts.n_gold_errors} invalid reference queries. "
            "No training artifacts were published."
        )
    if not kept:
        raise ValueError("trace filtering kept no eligible training examples")
    with tempfile.TemporaryDirectory(prefix="llm-rlvr-traces-") as temporary:
        staged = write_trace_dataset(kept, Path(temporary) / "traces.jsonl")
        manifest = build_trace_manifest(candidate_set, counts).model_copy(
            update={"trace_dataset_sha256": sha256_of(staged)}
        )
        path = atomic_write_text(args.out, staged.read_text(encoding="utf-8"))
        write_pydantic_json(manifest, args.manifest)
    print(
        f"kept {counts.n_kept}/{counts.n_questions} questions "
        f"({counts.n_candidates} candidates, {counts.n_eligible} eligible, "
        f"{counts.n_gold_errors} gold errors) -> {path}"
    )
    return 0


def _cmd_reproduce(args: argparse.Namespace) -> int:
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.mode == "public":
        from llm_rlvr.report.public import main as reproduce_public

        if args.workers != 1:
            raise ValueError("--workers applies only to --mode full")
        if args.spider_db_root is not None or args.bird_db_root is not None:
            raise ValueError("database roots apply only to --mode full")
        return reproduce_public(root=args.root)
    from llm_rlvr.report.full import main as reproduce_full

    return reproduce_full(
        root=args.root,
        spider_db_root=args.spider_db_root,
        bird_db_root=args.bird_db_root,
        workers=args.workers,
    )


def _deterministic_slice(examples: list, limit: int, seed: int) -> list:
    if limit < 1:
        raise ValueError(f"--limit must be >= 1, got {limit}")
    if limit >= len(examples):
        return examples
    return random.Random(seed).sample(examples, limit)


if __name__ == "__main__":
    raise SystemExit(main())
