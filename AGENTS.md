# rlvr-sql-agent — agent instructions

An RL-trained, self-correcting text-to-SQL agent (GRPO + verifiable execution reward). Every
claimed gain is verified before it is claimed: controls, multi-seed CIs, and disclosed caveats
are part of the method, not overhead.

## Read this first, then route

| When you are… | Read |
|---|---|
| Touching any code, or new to the repo | `agents/docs/architecture.md` |
| Running baselines, training, or Modal commands | `RUNBOOK.md` |
| Citing a number, or pinning/locating an artifact | `ARTIFACTS.md` |
| Changing behavior or scope | the active change in `openspec/changes/` (work the tasks; don't diverge silently) |

## Commands

```bash
make check        # the gate: ruff lint + format-check + unit tests — green before any commit
make fmt          # auto-format + safe lint fixes
make eval         # re-derive the committed numbers from results/ (CPU, model-free)
make data         # fetch + verify the pinned Spider DBs
uv run pytest -m integration   # network/HF/model tests (deselected by default)
```

Everything runs through `uv` (`uv run …`, never bare `python`/`pip`). GPU work runs **only** via
`modal_app.py` entrypoints — never locally.

## Hard rules

- **Never break artifact re-derivation**: any number in a doc or table must reproduce from a
  committed artifact via `make eval`. New results land as CompletionSet JSON + score under
  `results/` with full provenance.
- **Never pool rollouts across seeds** — aggregate per-seed, then Student-t CI
  (`eval/aggregate.py`). Single-seed runs are point estimates; make no CI or significance claim.
- **Keep `guard_trivial_empty=True`** in training rewards and trace filtering — it blocks the
  always-empty-result reward hack.
- **Datasets are pinned + fetched, never committed** (HF mirrors under `HAL-9001/`; checksums in
  `data/artifacts.py`). Don't add data files to git.
- **Long Modal runs**: `modal run --detach modal_app.py::<entrypoint> … --spawn`, or a local
  disconnect kills the run. Stagger app creations (3 simultaneous hit a rate limit). The HF
  "set HF_TOKEN for higher rate limits" warning is benign noise, not an error.
- Paid GPU launches (training, full-dev evals) need explicit user approval of the run + rough
  cost first; smokes and pilots validate the rig before any expensive run.

## Conventions

- Typed records subclass `Frozen` (`_base.py`: pydantic frozen + `extra="forbid"`). No
  `dict[str, Any]` payloads for persisted artifacts.
- Pure, CPU-testable builders + thin GPU drivers (see architecture doc). Unit tests must not need
  a GPU or network; mark anything that does `@pytest.mark.integration`.
- Heavy/optional deps (`vllm`, `trl`, `peft`, `torch`) import lazily inside the functions that
  use them — the CPU eval layer must install and run without the GPU stack.
- Conventional commits (`type(scope): message`), one logical unit each.

## Current state (update when stages move)

Stage 3 (multi-turn GRPO, `openspec/changes/grpo-multi-turn/`) is active and **paused on a user
decision after rung 2 failed**. Four GPU smoke attempts, zero tool calls in 384 episodes: the
in-context rungs (`cot`/`mt`/`mt_demo`) failed (attempts 1–3), and the rung-2 tool-use SFT cold
start (`sft/tooluse-sft-r32`, executed 2026-06-12) also failed — its token mass was dominated by
the long Stage-1 direct traces, so it taught preamble-style direct answers instead of tool
calls, and cost −8.1pt Spider / −3.3pt BIRD single-turn (gate results + diagnosis + a concrete
fix list under task 2.1 in the change's `tasks.md`). Stage-2 artifacts are untouched
(`grpo/grpo-s0` only read). New since rung 2: a multi-turn inference eval
(`eval/multiturn_generate.py`, `mt_eval` entrypoint) and the tool-use dataset builder
(`train/tooluse.py`, `build-tooluse` CLI). **Next step: the user decides** — one diagnosed-fix
iteration of rung 2 (≈$1.50) or ship Stage 2 alone. Do not start paid Stage-3 training before a
smoke passes; task 2.1 stays unticked.
Stage 2 shipped and is archived with stages 0–1 in `openspec/changes/archive/`: 3-seed headline
(CI excludes zero on both benchmarks), near-binary ablation, and the Llama cross-family control.
