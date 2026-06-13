"""Modal app: GPU steps — baseline generation, trace sampling, and LoRA SFT (single A100).

    modal run modal_app.py::main --dataset spider --limit 8        # day-1 smoke
    modal run --detach modal_app.py::main --dataset spider --spawn # full Spider-dev baseline
    modal run modal_app.py::traces --limit 8                       # trace-sampling smoke
    modal run --detach modal_app.py::traces --spawn                # Spider-train candidates
    modal run --detach modal_app.py::sft --spawn                   # LoRA SFT on the traces

Generation/sampling write artifacts to a Modal Volume; pull with `modal volume get text2sql-runs
<name> ./runs/<name>` and filter/score on CPU (`text2sql-rlvr build-traces` / `score` / `make
eval`). Run with the `modal` CLI (`uv tool install modal`), not as a project dependency. Public HF
datasets/model, so no HF token is needed.

CUDA / torch / vLLM versions must agree — expect to re-verify them against the box on the first
image build (the "re-verify at build").
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

APP_NAME = "text2sql-rlvr-baseline"
CUDA_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
REMOTE_ROOT = "/root/text2sql-rlvr"
RUNS_DIR = "/runs"
MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"


def _model_tag(model_id: str) -> str:
    """A short, filesystem-safe family tag from a HF model id (the repo name, lowercased).

    Used to namespace per-family intermediate artifacts (e.g. SFT candidate sets) so the Qwen and
    Llama runs of the same recipe never collide on the Volume.
    """
    return model_id.rsplit("/", 1)[-1].lower()


app = modal.App(APP_NAME)

# Transient/local-only caches: never part of the image (they also mutate during a build and trip
# Modal's "modified during build" guard). .git is stripped — the entrypoint passes git state in.
# NOTE: the cache dirs are root-anchored (`runs/**`, not `**/runs`) — a deep glob like `**/data`
# would also strip the `src/text2sql_rlvr/data/` *package* from the image.
_IGNORE = [
    "**/.git",
    "**/.venv",
    "**/__pycache__",
    "**/*.pyc",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/.DS_Store",
    "runs/**",
    "data/**",
    "results/**",
]

# Two layers: the heavy GPU install (layer 1) is keyed only on pyproject.toml + uv.lock, so editing
# source doesn't re-download the ~GBs of CUDA/torch/vLLM — only the fast editable relink re-runs.
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .apt_install("git")
    .pip_install("uv")
    .add_local_file(
        Path(__file__).parent / "pyproject.toml", f"{REMOTE_ROOT}/pyproject.toml", copy=True
    )
    .add_local_file(Path(__file__).parent / "uv.lock", f"{REMOTE_ROOT}/uv.lock", copy=True)
    .workdir(REMOTE_ROOT)
    .run_commands(
        "uv export --frozen --extra gpu --extra train --no-emit-project -o /tmp/reqs.txt",
        "uv pip install --system -r /tmp/reqs.txt",
    )
    .add_local_dir(Path(__file__).parent, remote_path=REMOTE_ROOT, copy=True, ignore=_IGNORE)
    .run_commands("uv pip install --system --no-deps -e .")
)

runs = modal.Volume.from_name("text2sql-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("text2sql-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RUNS_DIR: runs, "/root/.cache/huggingface": hf_cache},
    timeout=6 * 60 * 60,
)
def baseline(
    dataset: str = "spider",
    split: str = "dev",
    few_shot_k: int = 0,
    few_shot_seed: int = 0,
    prompt_style: str = "cot",
    adapter: str | None = None,
    base_model: str = MODEL,
    limit: int | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    sampling_seed: int = 0,
    commit: str | None = None,
    dirty: bool | None = None,
) -> str:
    """Generate base or base+adapter completions for one dataset on the GPU; persist to Volume.

    `adapter` is a LoRA adapter dir relative to the runs Volume (e.g. `sft/sft-r32`) — that is
    how the SFT/GRPO checkpoint is evaluated with the identical pipeline. `base_model` defaults to
    Qwen; set it to the Llama base for the cross-family control (the adapter's base must match).
    """
    from pathlib import Path

    from text2sql_rlvr.data.artifacts import BIRD_ARCHIVE, SPIDER_DB_ARCHIVE, SPIDER_HF_REVISION
    from text2sql_rlvr.data.bird import ensure_bird_dev, load_bird_examples
    from text2sql_rlvr.data.spider import ensure_spider_databases, load_examples, load_schemas
    from text2sql_rlvr.eval.completions import write_completion_set
    from text2sql_rlvr.eval.generate import generate
    from text2sql_rlvr.eval.provenance import DecodingConfig

    data_root = Path(RUNS_DIR) / "data"
    if dataset == "spider":
        dbs = ensure_spider_databases(data_root / "spider")
        schemas = load_schemas(dbs.tables_json)
        examples = list(load_examples(split))
        train_pool = list(load_examples("train")) if few_shot_k else []
        dataset_revision = SPIDER_HF_REVISION
        db_sha = SPIDER_DB_ARCHIVE.sha256
    elif dataset == "bird":
        bird = ensure_bird_dev(data_root / "bird")
        schemas = load_schemas(bird.tables_json)
        examples = list(load_bird_examples(bird.dev_json))
        # BIRD few-shot needs BIRD-train (not self-hosted yet); BIRD baseline is zero-shot for now.
        train_pool = []
        few_shot_k = 0
        dataset_revision = BIRD_ARCHIVE.revision
        db_sha = BIRD_ARCHIVE.sha256
    else:
        raise ValueError(f"dataset must be 'spider' or 'bird', got {dataset!r}")

    if limit is not None:
        examples = examples[:limit]

    completion_set = generate(
        examples,
        schemas,
        model=base_model,
        backend="vllm",
        decoding=DecodingConfig(
            temperature=temperature, max_new_tokens=max_new_tokens, seed=sampling_seed
        ),
        dataset=dataset,
        dataset_revision=dataset_revision,
        split=split,
        db_archive_sha256=db_sha,
        train_pool=train_pool,
        few_shot_k=few_shot_k,
        few_shot_seed=few_shot_seed,
        prompt_style=prompt_style,
        adapter_path=(Path(RUNS_DIR) / adapter) if adapter else None,
        commit=commit,
        dirty=dirty,
    )
    shot = f"fs{few_shot_k}" if few_shot_k else "zs"
    style_tag = "" if prompt_style == "cot" else f"-{prompt_style}"
    adapter_tag = f"-{Path(adapter).name}" if adapter else ""
    # Non-default bases get a family tag so a Llama eval can't overwrite the Qwen artifact.
    base_tag = "" if base_model == MODEL else f"-{_model_tag(base_model)}"
    # Sampled (temperature > 0) runs get a sampling tag so they can't overwrite greedy artifacts.
    sample_tag = f"-t{temperature}-seed{sampling_seed}" if temperature > 0 else ""
    name = f"{dataset}-{split}-{shot}{style_tag}{base_tag}{adapter_tag}{sample_tag}" + (
        f"-n{limit}" if limit else ""
    )
    out = write_completion_set(completion_set, Path(RUNS_DIR) / f"{name}.json")
    runs.commit()
    print(f"wrote {len(completion_set.items)} completions -> {out}")
    return str(out)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RUNS_DIR: runs, "/root/.cache/huggingface": hf_cache},
    timeout=6 * 60 * 60,
)
def sample_traces(
    n: int = 8,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
    max_new_tokens: int = 1024,
    limit: int | None = None,
    base_model: str = MODEL,
) -> str:
    """Sample N `<think>` completions per Spider-train question; write a CandidateSet (D2/D4).

    `base_model` defaults to Qwen; set the Llama base for the cross-family control so the SFT
    cold-start traces are distilled from the same family that GRPO will then train (matched recipe).
    """
    from pathlib import Path

    from text2sql_rlvr.data.artifacts import SPIDER_DB_ARCHIVE, SPIDER_HF_REVISION
    from text2sql_rlvr.data.spider import ensure_spider_databases, load_examples, load_schemas
    from text2sql_rlvr.eval.completions import write_json
    from text2sql_rlvr.train.traces import CandidateSet, TraceSamplingConfig, sample_candidates

    dbs = ensure_spider_databases(Path(RUNS_DIR) / "data" / "spider")
    schemas = load_schemas(dbs.tables_json)
    examples = list(load_examples("train"))
    if limit is not None:
        examples = examples[:limit]

    sampling = TraceSamplingConfig(
        n=n, temperature=temperature, top_p=top_p, seed=seed, max_new_tokens=max_new_tokens
    )
    candidate_set = CandidateSet(
        base_model=base_model,
        sampling=sampling,
        dataset="spider",
        dataset_revision=SPIDER_HF_REVISION,
        db_archive_sha256=SPIDER_DB_ARCHIVE.sha256,
        items=sample_candidates(examples, schemas, model=base_model, sampling=sampling),
    )
    tag = _model_tag(base_model)
    name = f"candidates-{tag}-n{n}-seed{seed}" + (f"-limit{limit}" if limit else "") + ".json"
    out = write_json(candidate_set, Path(RUNS_DIR) / "sft" / name)
    runs.commit()
    print(f"wrote {len(candidate_set.items)} questions x {n} samples -> {out}")
    return str(out)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RUNS_DIR: runs, "/root/.cache/huggingface": hf_cache},
    timeout=6 * 60 * 60,
)
def sft_train(
    traces: str = "sft/traces.jsonl",
    run_id: str = "sft-r32",
    base_model: str = MODEL,
    model_revision: str | None = None,
    warm_start: str | None = None,
    render_tool_prompts: bool = False,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    seed: int = 0,
    commit: str | None = None,
    dirty: bool | None = None,
) -> str:
    """LoRA SFT on the trace dataset (D3); adapter + provenance land under `sft/<run-id>/`.

    `base_model` defaults to Qwen; set the Llama base (with its own trace dataset) for the
    cross-family control so the warm-start adapter's base matches the GRPO base. `warm_start`
    (an adapter dir relative to the Volume, e.g. `grpo/grpo-s0`) continues that adapter instead
    of training a fresh one — the Stage-3 tool-use cold start.
    """
    from pathlib import Path

    from text2sql_rlvr.train.sft import SftHyperparams, run_sft

    out = run_sft(
        Path(RUNS_DIR) / traces,
        Path(RUNS_DIR) / "sft" / run_id,
        base_model=base_model,
        base_model_revision=model_revision,
        warm_start_adapter=(Path(RUNS_DIR) / warm_start) if warm_start else None,
        render_tool_prompts=render_tool_prompts,
        hyperparams=SftHyperparams(epochs=epochs, learning_rate=learning_rate, seed=seed),
        commit=commit,
        dirty=dirty,
    )
    runs.commit()
    print(f"adapter + provenance -> {out}")
    return str(out)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RUNS_DIR: runs, "/root/.cache/huggingface": hf_cache},
    timeout=8 * 60 * 60,
)
def grpo_train(
    run_id: str = "grpo-s0",
    warm_start: str = "sft/sft-r32",
    base_model: str = MODEL,
    base_model_revision: str | None = None,
    reward_arm: str = "shaped",
    learning_rate: float = 1e-5,
    seed: int = 0,
    dev_size: int | None = None,
    limit: int | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> str:
    """Single-turn GRPO LoRA on Spider-train, warm-started from the SFT adapter (Stage 2).

    `warm_start` is a LoRA adapter dir relative to the runs Volume (e.g. `sft/sft-r32`). The
    `base_model` defaults to Qwen; set the Llama base + a Llama warm-start adapter for the control.
    `reward_arm` is `shaped` (headline) or `near_binary` (the ablation). Adapter + provenance land
    under `grpo/<run-id>/`.
    """
    from pathlib import Path

    from text2sql_rlvr.data.artifacts import SPIDER_HF_REVISION
    from text2sql_rlvr.data.spider import ensure_spider_databases, load_examples, load_schemas
    from text2sql_rlvr.rewards import RewardConfig
    from text2sql_rlvr.train.grpo import GrpoHyperparams, run_grpo

    dbs = ensure_spider_databases(Path(RUNS_DIR) / "data" / "spider")
    schemas = load_schemas(dbs.tables_json)
    examples = list(load_examples("train"))
    if limit is not None:
        examples = examples[:limit]

    hp_kwargs: dict[str, object] = {"learning_rate": learning_rate, "seed": seed}
    if dev_size is not None:  # smoke runs need a tiny dev slice so the train split isn't starved
        hp_kwargs["eval_slice_size"] = dev_size

    out = run_grpo(
        examples,
        schemas,
        dbs.database_root,
        Path(RUNS_DIR) / "grpo" / run_id,
        base_model=base_model,
        base_model_revision=base_model_revision,
        warm_start_adapter=(Path(RUNS_DIR) / warm_start) if warm_start else None,
        dataset="spider",
        dataset_revision=SPIDER_HF_REVISION,
        hyperparams=GrpoHyperparams(**hp_kwargs),
        reward=RewardConfig(arm=reward_arm),
        commit=commit,
        dirty=dirty,
    )
    runs.commit()
    print(f"GRPO adapter + provenance -> {out}")
    return str(out)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RUNS_DIR: runs, "/root/.cache/huggingface": hf_cache},
    timeout=2 * 60 * 60,
)
def multiturn_smoke(
    run_id: str = "grpo-mt-smoke",
    warm_start: str | None = "grpo/grpo-s0",
    base_model: str = MODEL,
    base_model_revision: str | None = None,
    prompt_style: str = "mt_demo",
    limit: int = 32,
    dev_size: int = 8,
    max_steps: int | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> str:
    """The Stage-3 de-risk smoke (tasks 2.1/2.2): a few real multi-turn GRPO steps from the
    Stage-2 adapter, asserting tool-output loss masking + the TRL #4543 IS-ratio band, and
    measuring the multi-turn memory/step-time envelope. Report JSON lands under
    `grpo-mt/<run-id>/`; a failed assertion exits non-zero AFTER persisting the report.
    """
    from pathlib import Path

    from text2sql_rlvr.data.artifacts import SPIDER_HF_REVISION
    from text2sql_rlvr.data.spider import ensure_spider_databases, load_examples, load_schemas
    from text2sql_rlvr.train.multiturn import MultiTurnHyperparams, run_multiturn_smoke

    dbs = ensure_spider_databases(Path(RUNS_DIR) / "data" / "spider")
    schemas = load_schemas(dbs.tables_json)
    examples = list(load_examples("train"))[:limit]

    out = run_multiturn_smoke(
        examples,
        schemas,
        dbs.database_root,
        Path(RUNS_DIR) / "grpo-mt" / run_id,
        base_model=base_model,
        base_model_revision=base_model_revision,
        warm_start_adapter=(Path(RUNS_DIR) / warm_start) if warm_start else None,
        dataset="spider",
        dataset_revision=SPIDER_HF_REVISION,
        # The Stage-2 smoke shape: a tiny dev slice so the train split isn't starved.
        hyperparams=MultiTurnHyperparams(eval_slice_size=dev_size),
        prompt_style=prompt_style,
        max_steps=max_steps,
        commit=commit,
        dirty=dirty,
    )
    runs.commit()
    print(f"multi-turn smoke report -> {out}")
    return str(out)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RUNS_DIR: runs, "/root/.cache/huggingface": hf_cache},
    timeout=4 * 60 * 60,
)
def multiturn_eval(
    dataset: str = "spider",
    adapter: str | None = None,
    base_model: str = MODEL,
    limit: int | None = None,
    max_new_tokens: int = 1024,
    commit: str | None = None,
    dirty: bool | None = None,
) -> str:
    """Multi-turn greedy eval (no RL): the tool loop at inference time. Writes a normal
    CompletionSet plus the full `MultiTurnTranscriptSet` under `grpo-mt/eval/` on the Volume.
    """
    from pathlib import Path

    from text2sql_rlvr.data.artifacts import BIRD_ARCHIVE, SPIDER_DB_ARCHIVE, SPIDER_HF_REVISION
    from text2sql_rlvr.data.bird import ensure_bird_dev, load_bird_examples
    from text2sql_rlvr.data.spider import ensure_spider_databases, load_examples, load_schemas
    from text2sql_rlvr.eval.completions import write_completion_set, write_json
    from text2sql_rlvr.eval.multiturn_generate import generate_multiturn
    from text2sql_rlvr.eval.provenance import DecodingConfig

    data_root = Path(RUNS_DIR) / "data"
    if dataset == "spider":
        dbs = ensure_spider_databases(data_root / "spider")
        schemas = load_schemas(dbs.tables_json)
        examples = list(load_examples("dev"))
        db_root = dbs.database_root
        dataset_revision, db_sha = SPIDER_HF_REVISION, SPIDER_DB_ARCHIVE.sha256
    elif dataset == "bird":
        bird = ensure_bird_dev(data_root / "bird")
        schemas = load_schemas(bird.tables_json)
        examples = list(load_bird_examples(bird.dev_json))
        db_root = bird.database_root
        dataset_revision, db_sha = BIRD_ARCHIVE.revision, BIRD_ARCHIVE.sha256
    else:
        raise ValueError(f"dataset must be 'spider' or 'bird', got {dataset!r}")
    if limit is not None:
        examples = examples[:limit]

    completion_set, transcripts = generate_multiturn(
        examples,
        schemas,
        db_root,
        model=base_model,
        decoding=DecodingConfig(temperature=0.0, max_new_tokens=max_new_tokens),
        dataset=dataset,
        dataset_revision=dataset_revision,
        split="dev",
        adapter_path=(Path(RUNS_DIR) / adapter) if adapter else None,
        db_archive_sha256=db_sha,
        commit=commit,
        dirty=dirty,
    )
    adapter_tag = f"-{Path(adapter).name}" if adapter else ""
    name = f"{dataset}-dev-mt{adapter_tag}" + (f"-n{limit}" if limit else "")
    out = write_completion_set(completion_set, Path(RUNS_DIR) / "grpo-mt" / "eval" / f"{name}.json")
    write_json(transcripts, Path(RUNS_DIR) / "grpo-mt" / "eval" / f"{name}.transcripts.json")
    runs.commit()
    print(f"wrote {len(completion_set.items)} multi-turn completions -> {out}")
    return str(out)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RUNS_DIR: runs, "/root/.cache/huggingface": hf_cache},
    timeout=2 * 60 * 60,
)
def forced_revision_smoke(
    run_id: str = "grpo-fr-smoke",
    warm_start: str = "grpo/grpo-s0",
    base_model: str = MODEL,
    base_model_revision: str | None = None,
    limit: int = 32,
    dev_size: int = 8,
    max_steps: int | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> str:
    """The forced-revision de-risk smoke (grpo-forced-revision task 2.1): a few real
    attempt→execute→feedback→revise GRPO steps from the Stage-2 adapter, asserting feedback
    masking, the TRL #4543 IS-ratio band, and no NaNs. Report lands under `grpo-fr/<run-id>/`;
    a failed assertion exits non-zero AFTER persisting the report.
    """
    from pathlib import Path

    from text2sql_rlvr.data.artifacts import SPIDER_HF_REVISION
    from text2sql_rlvr.data.spider import ensure_spider_databases, load_examples, load_schemas
    from text2sql_rlvr.train.forced_revision import run_forced_revision_smoke
    from text2sql_rlvr.train.grpo import GrpoHyperparams

    dbs = ensure_spider_databases(Path(RUNS_DIR) / "data" / "spider")
    schemas = load_schemas(dbs.tables_json)
    examples = list(load_examples("train"))[:limit]

    out = run_forced_revision_smoke(
        examples,
        schemas,
        dbs.database_root,
        Path(RUNS_DIR) / "grpo-fr" / run_id,
        base_model=base_model,
        base_model_revision=base_model_revision,
        warm_start_adapter=Path(RUNS_DIR) / warm_start,
        dataset="spider",
        dataset_revision=SPIDER_HF_REVISION,
        hyperparams=GrpoHyperparams(eval_slice_size=dev_size),
        max_steps=max_steps,
        commit=commit,
        dirty=dirty,
    )
    runs.commit()
    print(f"forced-revision smoke report -> {out}")
    return str(out)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RUNS_DIR: runs, "/root/.cache/huggingface": hf_cache},
    timeout=12 * 60 * 60,
)
def forced_revision_train(
    run_id: str = "grpo-fr-s0",
    warm_start: str = "grpo/grpo-s0",
    base_model: str = MODEL,
    base_model_revision: str | None = None,
    learning_rate: float = 5e-5,
    seed: int = 0,
    dev_size: int | None = None,
    limit: int | None = None,
    max_steps: int | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> str:
    """Forced-revision GRPO on Spider-train, warm-started from the Stage-2 adapter (Stage 3b).

    `max_steps` bounds the pilot; the full run leaves it unset. Adapter + provenance + the D8
    diagnostics series land under `grpo-fr/<run-id>/`.
    """
    from pathlib import Path

    from text2sql_rlvr.data.artifacts import SPIDER_HF_REVISION
    from text2sql_rlvr.data.spider import ensure_spider_databases, load_examples, load_schemas
    from text2sql_rlvr.train.forced_revision import run_forced_revision_train
    from text2sql_rlvr.train.grpo import GrpoHyperparams

    dbs = ensure_spider_databases(Path(RUNS_DIR) / "data" / "spider")
    schemas = load_schemas(dbs.tables_json)
    examples = list(load_examples("train"))
    if limit is not None:
        examples = examples[:limit]

    hp_kwargs: dict[str, object] = {"learning_rate": learning_rate, "seed": seed}
    if dev_size is not None:
        hp_kwargs["eval_slice_size"] = dev_size

    out = run_forced_revision_train(
        examples,
        schemas,
        dbs.database_root,
        Path(RUNS_DIR) / "grpo-fr" / run_id,
        base_model=base_model,
        base_model_revision=base_model_revision,
        warm_start_adapter=Path(RUNS_DIR) / warm_start,
        dataset="spider",
        dataset_revision=SPIDER_HF_REVISION,
        hyperparams=GrpoHyperparams(**hp_kwargs),
        max_steps=max_steps,
        commit=commit,
        dirty=dirty,
    )
    runs.commit()
    print(f"forced-revision adapter + provenance + D8 series -> {out}")
    return str(out)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RUNS_DIR: runs, "/root/.cache/huggingface": hf_cache},
    timeout=4 * 60 * 60,
)
def forced_revision_eval(
    dataset: str = "spider",
    adapter: str | None = None,
    base_model: str = MODEL,
    limit: int | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    sampling_seed: int = 0,
    policy: str = "always",
    commit: str | None = None,
    dirty: bool | None = None,
) -> str:
    """Forced-revision eval: greedy attempt, execution feedback, optional greedy revision.
    Writes a normal CompletionSet plus the full transcripts under `grpo-fr/eval/`."""
    from pathlib import Path

    from text2sql_rlvr.data.artifacts import BIRD_ARCHIVE, SPIDER_DB_ARCHIVE, SPIDER_HF_REVISION
    from text2sql_rlvr.data.bird import ensure_bird_dev, load_bird_examples
    from text2sql_rlvr.data.spider import ensure_spider_databases, load_examples, load_schemas
    from text2sql_rlvr.eval.completions import write_completion_set, write_json
    from text2sql_rlvr.eval.forced_revision_generate import (
        forced_revision_artifact_stem,
        generate_forced_revision,
        validate_forced_revision_policy,
    )
    from text2sql_rlvr.eval.fr_metrics import compute_forced_revision_eval_report
    from text2sql_rlvr.eval.provenance import DecodingConfig

    policy = validate_forced_revision_policy(policy)
    data_root = Path(RUNS_DIR) / "data"
    if dataset == "spider":
        dbs = ensure_spider_databases(data_root / "spider")
        schemas = load_schemas(dbs.tables_json)
        examples = list(load_examples("dev"))
        db_root = dbs.database_root
        dataset_revision, db_sha = SPIDER_HF_REVISION, SPIDER_DB_ARCHIVE.sha256
    elif dataset == "bird":
        bird = ensure_bird_dev(data_root / "bird")
        schemas = load_schemas(bird.tables_json)
        examples = list(load_bird_examples(bird.dev_json))
        db_root = bird.database_root
        dataset_revision, db_sha = BIRD_ARCHIVE.revision, BIRD_ARCHIVE.sha256
    else:
        raise ValueError(f"dataset must be 'spider' or 'bird', got {dataset!r}")
    if limit is not None:
        examples = examples[:limit]

    completion_set, transcripts = generate_forced_revision(
        examples,
        schemas,
        db_root,
        model=base_model,
        decoding=DecodingConfig(
            temperature=temperature, max_new_tokens=max_new_tokens, seed=sampling_seed
        ),
        dataset=dataset,
        dataset_revision=dataset_revision,
        split="dev",
        adapter_path=(Path(RUNS_DIR) / adapter) if adapter else None,
        db_archive_sha256=db_sha,
        policy=policy,
        commit=commit,
        dirty=dirty,
    )
    report = compute_forced_revision_eval_report(
        completion_set, transcripts, db_root, policy=policy
    )
    name = forced_revision_artifact_stem(
        dataset=dataset,
        adapter=adapter,
        limit=limit,
        temperature=temperature,
        sampling_seed=sampling_seed,
        policy=policy,
    )
    out = write_completion_set(completion_set, Path(RUNS_DIR) / "grpo-fr" / "eval" / f"{name}.json")
    write_json(transcripts, Path(RUNS_DIR) / "grpo-fr" / "eval" / f"{name}.transcripts.json")
    write_json(report, Path(RUNS_DIR) / "grpo-fr" / "eval" / f"{name}.report.json")
    runs.commit()
    print(
        f"wrote {len(completion_set.items)} forced-revision completions "
        f"(policy={policy}, second_generation={report.n_second_generation}) -> {out}"
    )
    return str(out)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RUNS_DIR: runs, "/root/.cache/huggingface": hf_cache},
    timeout=2 * 60 * 60,
)
def raw_generate(
    prompts: str,
    out: str,
    adapter: str | None = None,
    base_model: str = MODEL,
    max_new_tokens: int = 1024,
) -> str:
    """Generic prompts-in/completions-out greedy pass: reads a JSONL whose rows carry a
    `prompt` (chat messages) column, writes an aligned JSONL of `{"completion": text}`.
    Used by the grpo-forced-revision probe; the caller's report records the source artifact
    + commit, so this fn carries no provenance of its own."""
    import json
    from pathlib import Path

    from text2sql_rlvr.eval.generate import _run_vllm
    from text2sql_rlvr.eval.provenance import DecodingConfig

    rows = [
        json.loads(line)
        for line in (Path(RUNS_DIR) / prompts).read_text(encoding="utf-8").splitlines()
    ]
    texts = _run_vllm(
        [row["prompt"] for row in rows],
        model=base_model,
        decoding=DecodingConfig(temperature=0.0, max_new_tokens=max_new_tokens),
        adapter_path=(Path(RUNS_DIR) / adapter) if adapter else None,
    )
    out_path = Path(RUNS_DIR) / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps({"completion": t}, ensure_ascii=False) for t in texts) + "\n",
        encoding="utf-8",
    )
    runs.commit()
    print(f"wrote {len(texts)} completions -> {out_path}")
    return str(out_path)


@app.local_entrypoint()
def generate_raw(
    prompts: str,
    out: str,
    adapter: str | None = None,
    base_model: str = MODEL,
    max_new_tokens: int = 1024,
    spawn: bool = False,
) -> None:
    """Launch the generic greedy pass over a prompts JSONL on the Volume."""
    kwargs = {
        "prompts": prompts,
        "out": out,
        "adapter": adapter,
        "base_model": base_model,
        "max_new_tokens": max_new_tokens,
    }
    if spawn:
        call = raw_generate.spawn(**kwargs)
        print(f"raw generation dispatched (spawn): function call {call.object_id}")
    else:
        print(f"raw generation complete: {raw_generate.remote(**kwargs)}")


@app.local_entrypoint()
def fr_smoke(
    run_id: str = "grpo-fr-smoke",
    warm_start: str = "grpo/grpo-s0",
    base_model: str = MODEL,
    base_model_revision: str | None = None,
    limit: int = 32,
    dev_size: int = 8,
    max_steps: int | None = None,
    spawn: bool = False,
) -> None:
    """Launch the forced-revision de-risk smoke (~$1, blocks paid runs on failure)."""
    kwargs = {
        "run_id": run_id,
        "warm_start": warm_start,
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "limit": limit,
        "dev_size": dev_size,
        "max_steps": max_steps,
        # The image strips .git, so compute git state locally and pass it into provenance.
        "commit": _local_git_commit(),
        "dirty": _local_git_dirty(),
    }
    if spawn:
        call = forced_revision_smoke.spawn(**kwargs)
        print(f"forced-revision smoke dispatched (spawn): function call {call.object_id}")
    else:
        print(f"forced-revision smoke complete: {forced_revision_smoke.remote(**kwargs)}")


@app.local_entrypoint()
def fr_train(
    run_id: str = "grpo-fr-s0",
    warm_start: str = "grpo/grpo-s0",
    base_model: str = MODEL,
    base_model_revision: str | None = None,
    learning_rate: float = 5e-5,
    seed: int = 0,
    dev_size: int | None = None,
    limit: int | None = None,
    max_steps: int | None = None,
    spawn: bool = False,
) -> None:
    """Launch forced-revision GRPO. Use --detach --spawn for the pilot and the full run."""
    kwargs = {
        "run_id": run_id,
        "warm_start": warm_start,
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "learning_rate": learning_rate,
        "seed": seed,
        "dev_size": dev_size,
        "limit": limit,
        "max_steps": max_steps,
        # The image strips .git, so compute git state locally and pass it into provenance.
        "commit": _local_git_commit(),
        "dirty": _local_git_dirty(),
    }
    if spawn:
        call = forced_revision_train.spawn(**kwargs)
        print(f"forced-revision training dispatched (spawn): function call {call.object_id}")
    else:
        print(f"forced-revision training complete: {forced_revision_train.remote(**kwargs)}")


@app.local_entrypoint()
def fr_eval(
    dataset: str = "spider",
    adapter: str | None = None,
    base_model: str = MODEL,
    limit: int | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    sampling_seed: int = 0,
    policy: str = "always",
    spawn: bool = False,
) -> None:
    """Launch the forced-revision eval. Use --policy error_only for error-gated revision."""
    kwargs = {
        "dataset": dataset,
        "adapter": adapter,
        "base_model": base_model,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "sampling_seed": sampling_seed,
        "policy": policy,
        # The image strips .git, so compute git state locally and pass it into provenance.
        "commit": _local_git_commit(),
        "dirty": _local_git_dirty(),
    }
    if spawn:
        call = forced_revision_eval.spawn(**kwargs)
        print(f"forced-revision eval dispatched (spawn): function call {call.object_id}")
    else:
        print(f"forced-revision eval complete: {forced_revision_eval.remote(**kwargs)}")


@app.local_entrypoint()
def mt_eval(
    dataset: str = "spider",
    adapter: str | None = None,
    base_model: str = MODEL,
    limit: int | None = None,
    max_new_tokens: int = 1024,
    spawn: bool = False,
) -> None:
    """Launch the multi-turn greedy eval (no RL). Use --detach --spawn for full-dev runs."""
    kwargs = {
        "dataset": dataset,
        "adapter": adapter,
        "base_model": base_model,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        # The image strips .git, so compute git state locally and pass it into provenance.
        "commit": _local_git_commit(),
        "dirty": _local_git_dirty(),
    }
    if spawn:
        call = multiturn_eval.spawn(**kwargs)
        print(f"multi-turn eval dispatched (spawn): function call {call.object_id}")
    else:
        print(f"multi-turn eval complete: {multiturn_eval.remote(**kwargs)}")


@app.local_entrypoint()
def main(
    dataset: str = "spider",
    split: str = "dev",
    few_shot_k: int = 0,
    few_shot_seed: int = 0,
    prompt_style: str = "cot",
    adapter: str | None = None,
    base_model: str = MODEL,
    limit: int | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    sampling_seed: int = 0,
    spawn: bool = False,
) -> None:
    """Launch a baseline generation on the GPU. Use --detach --spawn for long (full-dev) runs."""
    kwargs = {
        "dataset": dataset,
        "split": split,
        "few_shot_k": few_shot_k,
        "few_shot_seed": few_shot_seed,
        "prompt_style": prompt_style,
        "adapter": adapter,
        "base_model": base_model,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "sampling_seed": sampling_seed,
        # The image strips .git, so compute git state locally and pass it into provenance.
        "commit": _local_git_commit(),
        "dirty": _local_git_dirty(),
    }
    if spawn:
        call = baseline.spawn(**kwargs)
        print(f"baseline dispatched (spawn): function call {call.object_id}")
    else:
        print(f"baseline complete: {baseline.remote(**kwargs)}")


@app.local_entrypoint()
def traces(
    n: int = 8,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
    max_new_tokens: int = 1024,
    limit: int | None = None,
    base_model: str = MODEL,
    spawn: bool = False,
) -> None:
    """Launch Spider-train trace sampling. Use --detach --spawn for the full-train run."""
    kwargs = {
        "n": n,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "limit": limit,
        "base_model": base_model,
    }
    if spawn:
        call = sample_traces.spawn(**kwargs)
        print(f"trace sampling dispatched (spawn): function call {call.object_id}")
    else:
        print(f"trace sampling complete: {sample_traces.remote(**kwargs)}")


@app.local_entrypoint()
def sft(
    traces: str = "sft/traces.jsonl",
    run_id: str = "sft-r32",
    base_model: str = MODEL,
    model_revision: str | None = None,
    warm_start: str | None = None,
    render_tool_prompts: bool = False,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    seed: int = 0,
    spawn: bool = False,
) -> None:
    """Launch the LoRA SFT. `traces` is the dataset path relative to the runs Volume."""
    kwargs = {
        "traces": traces,
        "run_id": run_id,
        "base_model": base_model,
        "model_revision": model_revision,
        "warm_start": warm_start,
        "render_tool_prompts": render_tool_prompts,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        # The image strips .git, so compute git state locally and pass it into provenance.
        "commit": _local_git_commit(),
        "dirty": _local_git_dirty(),
    }
    if spawn:
        call = sft_train.spawn(**kwargs)
        print(f"SFT dispatched (spawn): function call {call.object_id}")
    else:
        print(f"SFT complete: {sft_train.remote(**kwargs)}")


@app.local_entrypoint()
def grpo(
    run_id: str = "grpo-s0",
    warm_start: str = "sft/sft-r32",
    base_model: str = MODEL,
    base_model_revision: str | None = None,
    reward_arm: str = "shaped",
    learning_rate: float = 1e-5,
    seed: int = 0,
    dev_size: int | None = None,
    limit: int | None = None,
    spawn: bool = False,
) -> None:
    """Launch single-turn GRPO. Use --detach --spawn for the headline/seed runs.

    modal run modal_app.py::grpo --run-id grpo-smoke --limit 32 --dev-size 8  # colocate smoke
    modal run --detach modal_app.py::grpo --run-id grpo-s0 --spawn            # a headline seed
    """
    kwargs = {
        "run_id": run_id,
        "warm_start": warm_start,
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "reward_arm": reward_arm,
        "learning_rate": learning_rate,
        "seed": seed,
        "dev_size": dev_size,
        "limit": limit,
        # The image strips .git, so compute git state locally and pass it into provenance.
        "commit": _local_git_commit(),
        "dirty": _local_git_dirty(),
    }
    if spawn:
        call = grpo_train.spawn(**kwargs)
        print(f"GRPO dispatched (spawn): function call {call.object_id}")
    else:
        print(f"GRPO complete: {grpo_train.remote(**kwargs)}")


@app.local_entrypoint()
def mt_smoke(
    run_id: str = "grpo-mt-smoke",
    warm_start: str | None = "grpo/grpo-s0",
    base_model: str = MODEL,
    base_model_revision: str | None = None,
    prompt_style: str = "mt_demo",
    limit: int = 32,
    dev_size: int = 8,
    max_steps: int | None = None,
    spawn: bool = False,
) -> None:
    """Launch the Stage-3 multi-turn de-risk smoke (~$1, blocks paid runs on failure).

    modal run modal_app.py::mt_smoke          # attended; ~15-25 min on one A100
    """
    kwargs = {
        "run_id": run_id,
        "warm_start": warm_start,
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "prompt_style": prompt_style,
        "limit": limit,
        "dev_size": dev_size,
        "max_steps": max_steps,
        # The image strips .git, so compute git state locally and pass it into provenance.
        "commit": _local_git_commit(),
        "dirty": _local_git_dirty(),
    }
    if spawn:
        call = multiturn_smoke.spawn(**kwargs)
        print(f"multi-turn smoke dispatched (spawn): function call {call.object_id}")
    else:
        print(f"multi-turn smoke complete: {multiturn_smoke.remote(**kwargs)}")


def _local_git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _local_git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return bool(out.stdout.strip())
