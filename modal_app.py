"""Modal app: run the base-model baseline generation on a single A100 via vLLM.

    modal run modal_app.py --dataset spider --limit 8            # day-1 smoke
    modal run --detach modal_app.py --dataset spider --spawn     # full Spider-dev
    modal run --detach modal_app.py --dataset spider --few-shot-k 3 --spawn
    modal run --detach modal_app.py --dataset bird --spawn       # BIRD-dev (headline)

Generation is the only GPU step; it writes a `CompletionSet` to a Modal Volume. Pull it with
`modal volume get text2sql-runs <name> ./runs/<name>` and score on CPU (`make eval` / `text2sql-rlvr
score`). Run with the `modal` CLI (`uv tool install modal`), not as a project dependency. Public HF
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
        "uv export --frozen --extra gpu --no-emit-project -o /tmp/reqs.txt",
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
    limit: int | None = None,
    max_new_tokens: int = 512,
    commit: str | None = None,
    dirty: bool | None = None,
) -> str:
    """Generate the base-model baseline for one dataset on the GPU; persist to the Volume."""
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
        db_sha = SPIDER_DB_ARCHIVE.sha256 if SPIDER_DB_ARCHIVE else None
    elif dataset == "bird":
        bird = ensure_bird_dev(data_root / "bird")
        schemas = load_schemas(bird.tables_json)
        examples = list(load_bird_examples(bird.dev_json))
        # BIRD few-shot needs BIRD-train (not self-hosted yet); BIRD baseline is zero-shot for now.
        train_pool = []
        few_shot_k = 0
        dataset_revision = BIRD_ARCHIVE.revision if BIRD_ARCHIVE else "unknown"
        db_sha = BIRD_ARCHIVE.sha256 if BIRD_ARCHIVE else None
    else:
        raise ValueError(f"dataset must be 'spider' or 'bird', got {dataset!r}")

    if limit is not None:
        examples = examples[:limit]

    completion_set = generate(
        examples,
        schemas,
        model=MODEL,
        backend="vllm",
        decoding=DecodingConfig(temperature=0.0, max_new_tokens=max_new_tokens),
        dataset=dataset,
        dataset_revision=dataset_revision,
        split=split,
        db_archive_sha256=db_sha,
        train_pool=train_pool,
        few_shot_k=few_shot_k,
        few_shot_seed=few_shot_seed,
        prompt_style=prompt_style,
        commit=commit,
        dirty=dirty,
    )
    shot = f"fs{few_shot_k}" if few_shot_k else "zs"
    style_tag = "" if prompt_style == "cot" else f"-{prompt_style}"
    name = f"{dataset}-{split}-{shot}{style_tag}" + (f"-n{limit}" if limit else "")
    out = write_completion_set(completion_set, Path(RUNS_DIR) / f"{name}.json")
    runs.commit()
    print(f"wrote {len(completion_set.items)} completions -> {out}")
    return str(out)


@app.local_entrypoint()
def main(
    dataset: str = "spider",
    split: str = "dev",
    few_shot_k: int = 0,
    few_shot_seed: int = 0,
    prompt_style: str = "cot",
    limit: int | None = None,
    max_new_tokens: int = 512,
    spawn: bool = False,
) -> None:
    """Launch a baseline generation on the GPU. Use --detach --spawn for long (full-dev) runs."""
    kwargs = {
        "dataset": dataset,
        "split": split,
        "few_shot_k": few_shot_k,
        "few_shot_seed": few_shot_seed,
        "prompt_style": prompt_style,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        # The image strips .git, so compute git state locally and pass it into provenance.
        "commit": _local_git_commit(),
        "dirty": _local_git_dirty(),
    }
    if spawn:
        call = baseline.spawn(**kwargs)
        print(f"baseline dispatched (spawn): function call {call.object_id}")
    else:
        print(f"baseline complete: {baseline.remote(**kwargs)}")


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
