"""Modal app: GPU steps — baseline generation, trace sampling, and LoRA SFT (single A100).

    modal run --no-spawn modal_app.py::main --dataset spider --limit 8  # day-1 smoke (blocks)
    modal run --detach modal_app.py::main --dataset spider             # full Spider-dev baseline
    modal run --no-spawn modal_app.py::traces --limit 8                # trace-sampling smoke
    modal run --detach modal_app.py::traces                            # Spider-train candidates
    modal run --detach modal_app.py::sft                               # LoRA SFT on the traces

Long entrypoints (main, traces, sft, grpo, fr_train, fr_eval, mt_eval) spawn by default, so
`--detach` is safe to leave without remembering `--spawn`; the smoke/foreground entrypoints
(generate_raw, fr_smoke, mt_smoke) block on `.remote()` so you get the inline result. Pass
`--no-spawn` to force a long entrypoint to block (e.g. a tiny `--limit` smoke).

Generation/sampling write artifacts to a Modal Volume; pull with `modal volume get text2sql-runs
<name> ./runs/<name>` and filter/score on CPU (`llm-rlvr-sql build-traces` / `score` / `make
eval`). Run with the `modal` CLI (`uv tool install modal`), not as a project dependency. Public HF
datasets/model, so no HF token is needed.

CUDA / torch / vLLM versions must agree — expect to re-verify them against the box on the first
image build (the "re-verify at build").
"""

from __future__ import annotations

from pathlib import Path

import modal
from modal.helpers import (
    commit_volume,
    dispatch_remote,
    git_kwargs,
    load_eval_context,
    load_spider_train_context,
    volume_path,
)

APP_NAME = "llm-rlvr-sql-baseline"
CUDA_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
REMOTE_ROOT = "/root/llm-rlvr-sql"
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
# would also strip the `src/llm_rlvr_sql/data/` *package* from the image.
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
VOLUMES = {RUNS_DIR: runs, "/root/.cache/huggingface": hf_cache}


def gpu_function(*, timeout_hours: int):
    return app.function(
        image=image,
        gpu="A100-80GB",
        volumes=VOLUMES,
        timeout=timeout_hours * 60 * 60,
    )


def runs_path(*parts: str | Path) -> Path:
    return volume_path(RUNS_DIR, *parts)


def adapter_path(adapter: str | None) -> Path | None:
    return runs_path(adapter) if adapter else None


@gpu_function(timeout_hours=6)
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

    from llm_rlvr_sql.eval.completions import write_completion_set
    from llm_rlvr_sql.eval.generate import generate
    from llm_rlvr_sql.eval.provenance import DecodingConfig

    ctx = load_eval_context(
        dataset=dataset,
        split=split,
        data_root=runs_path("data"),
        few_shot_k=few_shot_k,
        limit=limit,
    )
    completion_set = generate(
        ctx.examples,
        ctx.schemas,
        model=base_model,
        backend="vllm",
        decoding=DecodingConfig(
            temperature=temperature, max_new_tokens=max_new_tokens, seed=sampling_seed
        ),
        dataset=ctx.dataset,
        dataset_revision=ctx.dataset_revision,
        split=split,
        db_archive_sha256=ctx.db_archive_sha256,
        train_pool=ctx.train_pool,
        few_shot_k=few_shot_k if ctx.dataset == "spider" else 0,
        few_shot_seed=few_shot_seed,
        prompt_style=prompt_style,
        adapter_path=adapter_path(adapter),
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
    out = write_completion_set(completion_set, runs_path(f"{name}.json"))
    commit_volume(runs, f"wrote {len(completion_set.items)} completions -> {out}")
    return str(out)


@gpu_function(timeout_hours=6)
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
    from llm_rlvr_sql.eval.completions import write_json
    from llm_rlvr_sql.train.traces import CandidateSet, TraceSamplingConfig, sample_candidates

    ctx = load_spider_train_context(data_root=runs_path("data"), limit=limit)

    sampling = TraceSamplingConfig(
        n=n, temperature=temperature, top_p=top_p, seed=seed, max_new_tokens=max_new_tokens
    )
    candidate_set = CandidateSet(
        base_model=base_model,
        sampling=sampling,
        dataset="spider",
        dataset_revision=ctx.dataset_revision,
        db_archive_sha256=ctx.db_archive_sha256,
        items=sample_candidates(ctx.examples, ctx.schemas, model=base_model, sampling=sampling),
    )
    tag = _model_tag(base_model)
    name = f"candidates-{tag}-n{n}-seed{seed}" + (f"-limit{limit}" if limit else "") + ".json"
    out = write_json(candidate_set, runs_path("sft", name))
    commit_volume(runs, f"wrote {len(candidate_set.items)} questions x {n} samples -> {out}")
    return str(out)


@gpu_function(timeout_hours=6)
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
    from llm_rlvr_sql.train.sft import SftHyperparams, run_sft

    out = run_sft(
        runs_path(traces),
        runs_path("sft", run_id),
        base_model=base_model,
        base_model_revision=model_revision,
        warm_start_adapter=adapter_path(warm_start),
        render_tool_prompts=render_tool_prompts,
        hyperparams=SftHyperparams(epochs=epochs, learning_rate=learning_rate, seed=seed),
        commit=commit,
        dirty=dirty,
    )
    commit_volume(runs, f"adapter + provenance -> {out}")
    return str(out)


@gpu_function(timeout_hours=8)
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
    from llm_rlvr_sql.rewards import RewardConfig
    from llm_rlvr_sql.train.grpo import GrpoHyperparams, run_grpo

    ctx = load_spider_train_context(data_root=runs_path("data"), limit=limit)

    hp_kwargs: dict[str, object] = {"learning_rate": learning_rate, "seed": seed}
    if dev_size is not None:  # smoke runs need a tiny dev slice so the train split isn't starved
        hp_kwargs["eval_slice_size"] = dev_size

    out = run_grpo(
        ctx.examples,
        ctx.schemas,
        ctx.database_root,
        runs_path("grpo", run_id),
        base_model=base_model,
        base_model_revision=base_model_revision,
        warm_start_adapter=adapter_path(warm_start),
        dataset="spider",
        dataset_revision=ctx.dataset_revision,
        hyperparams=GrpoHyperparams(**hp_kwargs),
        reward=RewardConfig(arm=reward_arm),
        commit=commit,
        dirty=dirty,
    )
    commit_volume(runs, f"GRPO adapter + provenance -> {out}")
    return str(out)


@gpu_function(timeout_hours=2)
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
    from llm_rlvr_sql.train.multiturn import MultiTurnHyperparams, run_multiturn_smoke

    ctx = load_spider_train_context(data_root=runs_path("data"), limit=limit)

    out = run_multiturn_smoke(
        ctx.examples,
        ctx.schemas,
        ctx.database_root,
        runs_path("grpo-mt", run_id),
        base_model=base_model,
        base_model_revision=base_model_revision,
        warm_start_adapter=adapter_path(warm_start),
        dataset="spider",
        dataset_revision=ctx.dataset_revision,
        # The Stage-2 smoke shape: a tiny dev slice so the train split isn't starved.
        hyperparams=MultiTurnHyperparams(eval_slice_size=dev_size),
        prompt_style=prompt_style,
        max_steps=max_steps,
        commit=commit,
        dirty=dirty,
    )
    commit_volume(runs, f"multi-turn smoke report -> {out}")
    return str(out)


@gpu_function(timeout_hours=4)
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
    from llm_rlvr_sql.eval.completions import write_completion_set, write_json
    from llm_rlvr_sql.eval.multiturn_generate import generate_multiturn
    from llm_rlvr_sql.eval.provenance import DecodingConfig

    ctx = load_eval_context(dataset=dataset, split="dev", data_root=runs_path("data"), limit=limit)
    completion_set, transcripts = generate_multiturn(
        ctx.examples,
        ctx.schemas,
        ctx.database_root,
        model=base_model,
        decoding=DecodingConfig(temperature=0.0, max_new_tokens=max_new_tokens),
        dataset=ctx.dataset,
        dataset_revision=ctx.dataset_revision,
        split="dev",
        adapter_path=adapter_path(adapter),
        db_archive_sha256=ctx.db_archive_sha256,
        commit=commit,
        dirty=dirty,
    )
    adapter_tag = f"-{Path(adapter).name}" if adapter else ""
    name = f"{dataset}-dev-mt{adapter_tag}" + (f"-n{limit}" if limit else "")
    out = write_completion_set(completion_set, runs_path("grpo-mt", "eval", f"{name}.json"))
    write_json(transcripts, runs_path("grpo-mt", "eval", f"{name}.transcripts.json"))
    commit_volume(runs, f"wrote {len(completion_set.items)} multi-turn completions -> {out}")
    return str(out)


@gpu_function(timeout_hours=2)
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
    from llm_rlvr_sql.revision import run_forced_revision_smoke
    from llm_rlvr_sql.train.grpo import GrpoHyperparams

    ctx = load_spider_train_context(data_root=runs_path("data"), limit=limit)

    out = run_forced_revision_smoke(
        ctx.examples,
        ctx.schemas,
        ctx.database_root,
        runs_path("grpo-fr", run_id),
        base_model=base_model,
        base_model_revision=base_model_revision,
        warm_start_adapter=runs_path(warm_start),
        dataset="spider",
        dataset_revision=ctx.dataset_revision,
        hyperparams=GrpoHyperparams(eval_slice_size=dev_size),
        max_steps=max_steps,
        commit=commit,
        dirty=dirty,
    )
    commit_volume(runs, f"forced-revision smoke report -> {out}")
    return str(out)


@gpu_function(timeout_hours=12)
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
    from llm_rlvr_sql.revision import run_forced_revision_train
    from llm_rlvr_sql.train.grpo import GrpoHyperparams

    ctx = load_spider_train_context(data_root=runs_path("data"), limit=limit)

    hp_kwargs: dict[str, object] = {"learning_rate": learning_rate, "seed": seed}
    if dev_size is not None:
        hp_kwargs["eval_slice_size"] = dev_size

    out = run_forced_revision_train(
        ctx.examples,
        ctx.schemas,
        ctx.database_root,
        runs_path("grpo-fr", run_id),
        base_model=base_model,
        base_model_revision=base_model_revision,
        warm_start_adapter=runs_path(warm_start),
        dataset="spider",
        dataset_revision=ctx.dataset_revision,
        hyperparams=GrpoHyperparams(**hp_kwargs),
        max_steps=max_steps,
        commit=commit,
        dirty=dirty,
    )
    commit_volume(runs, f"forced-revision adapter + provenance + D8 series -> {out}")
    return str(out)


@gpu_function(timeout_hours=4)
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
    from llm_rlvr_sql.eval.completions import write_completion_set, write_json
    from llm_rlvr_sql.eval.forced_revision_generate import (
        forced_revision_artifact_stem,
        generate_forced_revision,
        validate_forced_revision_policy,
    )
    from llm_rlvr_sql.eval.fr_metrics import compute_forced_revision_eval_report
    from llm_rlvr_sql.eval.provenance import DecodingConfig

    policy = validate_forced_revision_policy(policy)
    ctx = load_eval_context(dataset=dataset, split="dev", data_root=runs_path("data"), limit=limit)
    completion_set, transcripts = generate_forced_revision(
        ctx.examples,
        ctx.schemas,
        ctx.database_root,
        model=base_model,
        decoding=DecodingConfig(
            temperature=temperature, max_new_tokens=max_new_tokens, seed=sampling_seed
        ),
        dataset=ctx.dataset,
        dataset_revision=ctx.dataset_revision,
        split="dev",
        adapter_path=adapter_path(adapter),
        db_archive_sha256=ctx.db_archive_sha256,
        policy=policy,
        commit=commit,
        dirty=dirty,
    )
    report = compute_forced_revision_eval_report(
        completion_set, transcripts, ctx.database_root, policy=policy
    )
    name = forced_revision_artifact_stem(
        dataset=dataset,
        adapter=adapter,
        limit=limit,
        temperature=temperature,
        sampling_seed=sampling_seed,
        policy=policy,
    )
    out = write_completion_set(completion_set, runs_path("grpo-fr", "eval", f"{name}.json"))
    write_json(transcripts, runs_path("grpo-fr", "eval", f"{name}.transcripts.json"))
    write_json(report, runs_path("grpo-fr", "eval", f"{name}.report.json"))
    commit_volume(
        runs,
        f"wrote {len(completion_set.items)} forced-revision completions "
        f"(policy={policy}, second_generation={report.n_second_generation}) -> {out}",
    )
    return str(out)


@gpu_function(timeout_hours=2)
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

    from llm_rlvr_sql.eval.provenance import DecodingConfig
    from llm_rlvr_sql.eval.vllm_backend import make_vllm_engine, sample_vllm_completions

    rows = [
        json.loads(line) for line in runs_path(prompts).read_text(encoding="utf-8").splitlines()
    ]
    decoding = DecodingConfig(temperature=0.0, max_new_tokens=max_new_tokens)
    llm, chat_kwargs = make_vllm_engine(base_model, adapter_path=adapter_path(adapter))
    texts = sample_vllm_completions(
        llm, [row["prompt"] for row in rows], decoding, chat_kwargs=chat_kwargs
    )
    out_path = runs_path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps({"completion": t}, ensure_ascii=False) for t in texts) + "\n",
        encoding="utf-8",
    )
    commit_volume(runs, f"wrote {len(texts)} completions -> {out_path}")
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
    dispatch_remote(
        raw_generate,
        kwargs,
        spawn=spawn,
        label="raw generation",
        app_name=APP_NAME,
    )


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
        **git_kwargs(),
    }
    dispatch_remote(
        forced_revision_smoke,
        kwargs,
        spawn=spawn,
        label="forced-revision smoke",
        app_name=APP_NAME,
    )


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
    spawn: bool = True,
) -> None:
    """Launch forced-revision GRPO (pilot and full run). Spawns by default so `modal run --detach`
    is safe to leave; pass --no-spawn to block on the result (e.g. a tiny --limit smoke)."""
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
        **git_kwargs(),
    }
    dispatch_remote(
        forced_revision_train,
        kwargs,
        spawn=spawn,
        label="forced-revision training",
        app_name=APP_NAME,
    )


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
    spawn: bool = True,
) -> None:
    """Launch the forced-revision eval (--policy error_only for error-gated revision). Spawns by
    default so `modal run --detach` is safe to leave; pass --no-spawn for a tiny --limit smoke."""
    kwargs = {
        "dataset": dataset,
        "adapter": adapter,
        "base_model": base_model,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "sampling_seed": sampling_seed,
        "policy": policy,
        **git_kwargs(),
    }
    dispatch_remote(
        forced_revision_eval,
        kwargs,
        spawn=spawn,
        label="forced-revision eval",
        app_name=APP_NAME,
    )


@app.local_entrypoint()
def mt_eval(
    dataset: str = "spider",
    adapter: str | None = None,
    base_model: str = MODEL,
    limit: int | None = None,
    max_new_tokens: int = 1024,
    spawn: bool = True,
) -> None:
    """Launch the multi-turn greedy eval (no RL). Spawns by default so `modal run --detach` is
    safe to leave; pass --no-spawn to block on the inline result (e.g. a tiny --limit smoke)."""
    kwargs = {
        "dataset": dataset,
        "adapter": adapter,
        "base_model": base_model,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        **git_kwargs(),
    }
    dispatch_remote(
        multiturn_eval,
        kwargs,
        spawn=spawn,
        label="multi-turn eval",
        app_name=APP_NAME,
    )


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
    spawn: bool = True,
) -> None:
    """Launch a baseline generation on the GPU. Spawns by default so `modal run --detach` is
    safe to leave; pass --no-spawn to block on the inline result (e.g. a tiny --limit smoke)."""
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
        **git_kwargs(),
    }
    dispatch_remote(
        baseline,
        kwargs,
        spawn=spawn,
        label="baseline",
        app_name=APP_NAME,
    )


@app.local_entrypoint()
def traces(
    n: int = 8,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
    max_new_tokens: int = 1024,
    limit: int | None = None,
    base_model: str = MODEL,
    spawn: bool = True,
) -> None:
    """Launch Spider-train trace sampling. Spawns by default so `modal run --detach` is safe to
    leave; pass --no-spawn to block on the inline result (e.g. a tiny --limit smoke)."""
    kwargs = {
        "n": n,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "limit": limit,
        "base_model": base_model,
    }
    dispatch_remote(
        sample_traces,
        kwargs,
        spawn=spawn,
        label="trace sampling",
        app_name=APP_NAME,
    )


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
    spawn: bool = True,
) -> None:
    """Launch the LoRA SFT. `traces` is the dataset path relative to the runs Volume. Spawns by
    default so `modal run --detach` is safe to leave; pass --no-spawn to block on the result."""
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
        **git_kwargs(),
    }
    dispatch_remote(sft_train, kwargs, spawn=spawn, label="SFT", app_name=APP_NAME)


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
    spawn: bool = True,
) -> None:
    """Launch single-turn GRPO. Spawns by default so `modal run --detach` is safe to leave.

    modal run --no-spawn modal_app.py::grpo --run-id grpo-smoke --limit 32 --dev-size 8  # smoke
    modal run --detach modal_app.py::grpo --run-id grpo-s0                               # seed
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
        **git_kwargs(),
    }
    dispatch_remote(grpo_train, kwargs, spawn=spawn, label="GRPO", app_name=APP_NAME)


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
        **git_kwargs(),
    }
    dispatch_remote(
        multiturn_smoke,
        kwargs,
        spawn=spawn,
        label="multi-turn smoke",
        app_name=APP_NAME,
    )
