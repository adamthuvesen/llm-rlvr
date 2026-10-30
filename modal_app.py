"""Modal app: GPU entrypoints for generation, trace sampling, SFT, and GRPO.

    modal run modal_app.py::main --dataset spider --limit 8 --no-spawn  # small smoke
    modal run --detach modal_app.py::main --dataset spider             # full Spider-dev baseline
    modal run modal_app.py::traces --limit 8 --no-spawn                # small trace sampling
    modal run --detach modal_app.py::traces                            # Spider-train candidates
    modal run --detach modal_app.py::sft                               # LoRA SFT on the traces

Every entrypoint spends money on GPUs and refuses to launch without approval. The refusal prints
the approved command; run it only after Adam approves the run and its rough cost.

Long entrypoints (main, traces, sft, grpo) spawn by default, so `--detach` is safe to leave without
remembering `--spawn`. Pass `--no-spawn` to force a long entrypoint to block.

Generation/sampling write artifacts to a Modal Volume; pull with `modal volume get text2sql-runs
<name> ./runs/<name>` and filter/score on CPU (`llm-rlvr build-traces` / `score` / `make
eval`). Run with the `modal` CLI (`uv tool install modal`), not as a project dependency. Public HF
datasets/model, so no HF token is needed.

CUDA, torch, and vLLM versions must agree; verify the image on the target GPU before paid runs.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from modal.app import _FunctionDecoratorType

from llm_rlvr.modal_helpers import (
    GenerationRequest,
    GenerationRunRequest,
    GrpoRequest,
    SftRequest,
    TraceRequest,
    TraceRunRequest,
    commit_volume,
    dispatch_remote,
    git_kwargs,
    gpu_function_command,
    load_eval_context,
    load_spider_train_context,
    managed_run_path,
    require_approval,
    volume_path,
)

APP_NAME = "llm-rlvr-baseline"
CUDA_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
REMOTE_ROOT = "/root/llm-rlvr"
RUNS_DIR = "/runs"
MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"


app = modal.App(APP_NAME)

# Transient/local-only caches: never part of the image (they also mutate during a build and trip
# Modal's "modified during build" guard). .git is stripped — the entrypoint passes git state in.
# NOTE: the cache dirs are root-anchored (`runs/**`, not `**/runs`) — a deep glob like `**/data`
# would also strip the `src/llm_rlvr/data/` *package* from the image.
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
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.14")
    .apt_install("git")
    .pip_install("uv")
    .add_local_file(
        Path(__file__).parent / "pyproject.toml", f"{REMOTE_ROOT}/pyproject.toml", copy=True
    )
    .add_local_file(Path(__file__).parent / "uv.lock", f"{REMOTE_ROOT}/uv.lock", copy=True)
    .workdir(REMOTE_ROOT)
    .run_commands(
        "uv export --frozen --extra train --no-emit-project -o /tmp/reqs.txt",
        "uv pip install --system -r /tmp/reqs.txt",
    )
    .add_local_dir(Path(__file__).parent, remote_path=REMOTE_ROOT, copy=True, ignore=_IGNORE)
    .run_commands("uv pip install --system --no-deps -e .")
)

runs = modal.Volume.from_name("text2sql-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("text2sql-hf-cache", create_if_missing=True)
VOLUMES: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    RUNS_DIR: runs,
    "/root/.cache/huggingface": hf_cache,
}


def modal_gpu_entrypoint(*, timeout_hours: int) -> _FunctionDecoratorType:
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


@modal_gpu_entrypoint(timeout_hours=6)
def baseline(
    dataset: str = "spider",
    split: str = "dev",
    few_shot_k: int = 0,
    few_shot_seed: int = 0,
    prompt_style: str = "cot",
    adapter: str | None = None,
    base_model: str = MODEL,
    model_revision: str | None = None,
    limit: int | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    sampling_seed: int = 0,
    run_id: str | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
    approved: bool = False,
) -> str:
    """Generate base or base+adapter completions for one dataset on the GPU; persist to Volume.

    `adapter` is a LoRA adapter dir relative to the runs Volume (e.g. `sft/sft-r32`) — that is
    how the SFT/GRPO checkpoint is evaluated with the identical pipeline. `base_model` defaults to
    Qwen; set it to the Llama base for the cross-family control (the adapter's base must match).
    """
    require_approval(approved, label="baseline", command=gpu_function_command("baseline"))
    del approved  # locals() below feeds the request model, which forbids extra fields
    request = GenerationRequest.model_validate(locals())
    from llm_rlvr.artifacts.io import ensure_new_output
    from llm_rlvr.artifacts.provenance import DecodingConfig
    from llm_rlvr.artifacts.runs import begin_run, publish_complete, require_complete
    from llm_rlvr.eval.completions import read_completion_set, write_completion_set
    from llm_rlvr.eval.generate import generate
    from llm_rlvr.train.configuration import resolve_adapter_base, resolve_model_revision

    output_dir = managed_run_path(RUNS_DIR, "generation", request)
    ensure_new_output(output_dir)
    adapter_root = adapter_path(adapter)
    adapter_identity = None
    if adapter_root is not None:
        require_complete(adapter_root)
        resolved_revision, adapter_identity = resolve_adapter_base(
            adapter_root, base_model=base_model, requested_revision=model_revision
        )
    else:
        resolved_revision = resolve_model_revision(base_model, model_revision)
    ctx = load_eval_context(
        dataset=dataset,
        split=split,
        data_root=runs_path("data"),
        few_shot_k=few_shot_k,
        limit=limit,
    )
    begin_run(
        output_dir,
        kind="generation",
        request=GenerationRunRequest(
            generation=request,
            resolved_model_revision=resolved_revision,
            adapter_provenance_sha256=adapter_identity.provenance_sha256
            if adapter_identity
            else None,
            adapter_sha256=adapter_identity.sha256 if adapter_identity else None,
            selection_policy="prefix" if limit is not None else "all",
            data_source=ctx.data_source,
            db_archive_sha256=ctx.db_archive_sha256,
        ),
    )
    runs.commit()
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
        model_revision=resolved_revision,
        db_archive_sha256=ctx.db_archive_sha256,
        train_pool=ctx.train_pool,
        few_shot_k=few_shot_k,
        few_shot_seed=few_shot_seed,
        prompt_style=prompt_style,
        adapter_path=adapter_root,
        adapter_sha256=adapter_identity.sha256 if adapter_identity else None,
        adapter_provenance_sha256=adapter_identity.provenance_sha256 if adapter_identity else None,
        commit=commit,
        dirty=dirty,
        selection_policy="prefix" if limit is not None else "all",
    )
    out = write_completion_set(completion_set, output_dir / "completions.json")
    # Validate the saved payload before publishing completion. Managed readers require the marker.
    type(completion_set).model_validate_json(out.read_text(encoding="utf-8"))
    publish_complete(output_dir, artifacts=[out])
    read_completion_set(out)
    commit_volume(runs, f"wrote {len(completion_set.items)} completions -> {out}")
    return str(out)


@modal_gpu_entrypoint(timeout_hours=6)
def sample_traces(
    n: int = 8,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
    max_new_tokens: int = 1024,
    limit: int | None = None,
    base_model: str = MODEL,
    model_revision: str | None = None,
    run_id: str | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
    approved: bool = False,
) -> str:
    """Sample N `<think>` completions per Spider-train question; write a CandidateSet.

    `base_model` defaults to Qwen; set the Llama base for the cross-family control so the SFT
    cold-start traces are distilled from the same family that GRPO will then train (matched recipe).
    """
    require_approval(approved, label="sample_traces", command=gpu_function_command("sample_traces"))
    del approved  # locals() below feeds the request model, which forbids extra fields
    request = TraceRequest.model_validate(locals())
    from llm_rlvr.artifacts.io import atomic_write_json, ensure_new_output
    from llm_rlvr.artifacts.runs import begin_run, publish_complete
    from llm_rlvr.train.configuration import resolve_model_revision
    from llm_rlvr.train.traces import CandidateSet, TraceSamplingConfig, sample_candidates

    output_dir = managed_run_path(RUNS_DIR, "traces", request)
    ensure_new_output(output_dir)
    resolved_revision = resolve_model_revision(base_model, model_revision)
    ctx = load_spider_train_context(data_root=runs_path("data"), limit=limit)
    begin_run(
        output_dir,
        kind="traces",
        request=TraceRunRequest(
            sampling=request,
            resolved_model_revision=resolved_revision,
            selection_policy="prefix" if limit is not None else "all",
            data_source=ctx.data_source,
            db_archive_sha256=ctx.db_archive_sha256,
        ),
    )
    runs.commit()
    sampling = TraceSamplingConfig(
        n=n, temperature=temperature, top_p=top_p, seed=seed, max_new_tokens=max_new_tokens
    )
    candidate_set = CandidateSet(
        base_model=base_model,
        base_model_revision=resolved_revision,
        sampling=sampling,
        dataset="spider",
        dataset_revision=ctx.dataset_revision,
        db_archive_sha256=ctx.db_archive_sha256,
        items=sample_candidates(
            ctx.examples,
            ctx.schemas,
            model=base_model,
            model_revision=resolved_revision,
            sampling=sampling,
        ),
    )
    out = atomic_write_json(candidate_set, output_dir / "candidates.json")
    CandidateSet.model_validate_json(out.read_text(encoding="utf-8"))
    publish_complete(output_dir, artifacts=[out])
    commit_volume(runs, f"wrote {len(candidate_set.items)} questions x {n} samples -> {out}")
    return str(out)


@modal_gpu_entrypoint(timeout_hours=6)
def sft_train(
    traces: str = "sft/traces.jsonl",
    trace_manifest: str = "sft/traces.manifest.json",
    run_id: str = "sft-r32",
    base_model: str = MODEL,
    model_revision: str | None = None,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    seed: int = 0,
    commit: str | None = None,
    dirty: bool | None = None,
    approved: bool = False,
) -> str:
    """LoRA SFT on the trace dataset; adapter + provenance land under `sft/<run-id>/`.

    `base_model` defaults to Qwen; set the Llama base (with its own trace dataset) for the
    cross-family control so the adapter's base matches the later GRPO base.
    """
    require_approval(approved, label="sft_train", command=gpu_function_command("sft_train"))
    del approved  # locals() below feeds the request model, which forbids extra fields
    SftRequest.model_validate(locals())
    from llm_rlvr.artifacts.io import ensure_new_output
    from llm_rlvr.train.sft import SftHyperparams, run_sft

    ensure_new_output(runs_path("sft", run_id))
    out = run_sft(
        runs_path(traces),
        runs_path("sft", run_id),
        base_model=base_model,
        base_model_revision=model_revision,
        trace_manifest_path=runs_path(trace_manifest),
        on_begin=runs.commit,
        hyperparams=SftHyperparams(epochs=epochs, learning_rate=learning_rate, seed=seed),
        commit=commit,
        dirty=dirty,
    )
    commit_volume(runs, f"adapter + provenance -> {out}")
    return str(out)


@modal_gpu_entrypoint(timeout_hours=8)
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
    approved: bool = False,
) -> str:
    """Single-turn GRPO LoRA on Spider-train, warm-started from the SFT adapter.

    `warm_start` is a LoRA adapter dir relative to the runs Volume (e.g. `sft/sft-r32`). The
    `base_model` defaults to Qwen; set the Llama base + a Llama warm-start adapter for the control.
    `reward_arm` is `shaped` (headline) or `near_binary` (the ablation). Adapter + provenance land
    under `grpo/<run-id>/`.
    """
    require_approval(approved, label="grpo_train", command=gpu_function_command("grpo_train"))
    del approved  # locals() below feeds the request model, which forbids extra fields
    request = GrpoRequest.model_validate(locals())
    from llm_rlvr.artifacts.io import ensure_new_output
    from llm_rlvr.tasks.sql import RewardConfig
    from llm_rlvr.train.grpo import GrpoHyperparams, run_grpo

    ensure_new_output(runs_path("grpo", run_id))
    ctx = load_spider_train_context(data_root=runs_path("data"), limit=limit)

    hp_kwargs: dict[str, object] = {"learning_rate": learning_rate, "seed": seed}
    if dev_size is not None:  # Small runs still need a dev slice so the train split is nonempty.
        hp_kwargs["eval_slice_size"] = dev_size

    out = run_grpo(
        ctx.examples,
        ctx.schemas,
        ctx.database_root,
        runs_path("grpo", run_id),
        base_model=base_model,
        base_model_revision=base_model_revision,
        warm_start_adapter=runs_path(request.warm_start),
        dataset="spider",
        dataset_revision=ctx.dataset_revision,
        hyperparams=GrpoHyperparams.model_validate(hp_kwargs),
        reward=RewardConfig(arm=request.reward_arm),
        on_begin=runs.commit,
        commit=commit,
        dirty=dirty,
    )
    commit_volume(runs, f"GRPO adapter + provenance -> {out}")
    return str(out)


@app.local_entrypoint()
def main(
    dataset: str = "spider",
    split: str = "dev",
    few_shot_k: int = 0,
    few_shot_seed: int = 0,
    prompt_style: str = "cot",
    adapter: str | None = None,
    base_model: str = MODEL,
    model_revision: str | None = None,
    limit: int | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    sampling_seed: int = 0,
    run_id: str | None = None,
    spawn: bool = True,
    approved: bool = False,
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
        "model_revision": model_revision,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "sampling_seed": sampling_seed,
        "run_id": run_id,
        **git_kwargs(),
    }
    dispatch_remote(
        baseline,
        GenerationRequest.model_validate(kwargs),
        approved=approved,
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
    model_revision: str | None = None,
    run_id: str | None = None,
    spawn: bool = True,
    approved: bool = False,
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
        "model_revision": model_revision,
        "run_id": run_id,
        **git_kwargs(),
    }
    dispatch_remote(
        sample_traces,
        TraceRequest.model_validate(kwargs),
        approved=approved,
        spawn=spawn,
        label="trace sampling",
        app_name=APP_NAME,
    )


@app.local_entrypoint()
def sft(
    traces: str = "sft/traces.jsonl",
    trace_manifest: str = "sft/traces.manifest.json",
    run_id: str = "sft-r32",
    base_model: str = MODEL,
    model_revision: str | None = None,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    seed: int = 0,
    spawn: bool = True,
    approved: bool = False,
) -> None:
    """Launch the LoRA SFT. `traces` is the dataset path relative to the runs Volume. Spawns by
    default so `modal run --detach` is safe to leave; pass --no-spawn to block on the result."""
    kwargs = {
        "traces": traces,
        "trace_manifest": trace_manifest,
        "run_id": run_id,
        "base_model": base_model,
        "model_revision": model_revision,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        **git_kwargs(),
    }
    dispatch_remote(
        sft_train,
        SftRequest.model_validate(kwargs),
        approved=approved,
        spawn=spawn,
        label="SFT",
        app_name=APP_NAME,
    )


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
    approved: bool = False,
) -> None:
    """Launch single-turn GRPO. Spawns by default so `modal run --detach` is safe to leave.

    modal run modal_app.py::grpo --run-id grpo-small --limit 32 --dev-size 8 --no-spawn
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
    dispatch_remote(
        grpo_train,
        GrpoRequest.model_validate(kwargs),
        approved=approved,
        spawn=spawn,
        label="GRPO",
        app_name=APP_NAME,
    )
