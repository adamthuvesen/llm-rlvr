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
decision**. CPU scaffolding done; three GPU smoke attempts cleared the TRL #4543 check and
measured the envelope, but the Stage-2 policy never calls `run_sql` (0/288 episodes — not under
the `cot` prompt, the `mt` instruction, or `mt_demo`'s full in-context tool demonstration,
verified rendered in the rollout log), so task 2.1's mask assertions are unvalidated — see the
journal under task 2.1 in the change's `tasks.md`. In-context seeding is exhausted. **Next step:
the user picks a rung** (recorded there): (2) small tool-use SFT cold-start, ~$5–10; (3) ship
Stage 2 alone; optional ~$0.50 diagnostic first — re-run the smoke warm-started from the SFT
adapter. Do not start paid Stage-3 training before a smoke passes.
Stage 2 shipped and is archived with stages 0–1 in `openspec/changes/archive/`: 3-seed headline
(CI excludes zero on both benchmarks), near-binary ablation, and the Llama cross-family control.
