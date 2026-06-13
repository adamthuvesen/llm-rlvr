# rlvr-sql-agent — agent instructions

Text-to-SQL post-training project: SFT + single-turn GRPO worked; the later self-correction
follow-up did not. Every claimed gain is backed by committed results, controls, multi-seed CIs,
and disclosed caveats.

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

**The project is complete at Stage 2; there is no active change.** The main result is the
Stage-2 single-turn GRPO result (3-seed CI excludes zero on both benchmarks, near-binary
ablation, Llama cross-family control; archived with stages 0–1). Two self-correction sequels
both closed honestly without a full training run, all archived in `openspec/changes/archive/`:

- **Stage 3 (`grpo-multi-turn`, tool choice) — CLOSED 2026-06-12 without training.** The 3B
  model never adopts the `run_sql` protocol: 0 tool calls in 672 episodes across base / SFT /
  GRPO / two tool-use-SFT checkpoints and every prompting strength; a two-arm probe ruled out
  RL formatting as the cause. TRL #4543 stayed clear throughout.
- **Stage 3b (`grpo-forced-revision`) — ABORTED 2026-06-12 at the pilot gate set before the run.**
  Tool choice removed (runner executes every answer; hand-written TRL `rollout_func` with
  loss-masked feedback splice — its smoke validated mask + IS-ratio on real spliced rollouts,
  the validation Stage 3 never reached). The probe passed (fix 14.9%, harm 6.3%), but the
  60-step pilot failed gates P1/P2: GRPO drifts toward identity-resubmission (frac_revising
  0.26→0.05, delta bonus below its untrained baseline, `frac_reward_zero_std` ≈ 0.6 — weak
  learning signal). Pilot-ckpt documentation evals (n=200): FR runner 0.720 vs Stage-2 anchor
  0.705 vs best-of-2 0.715 — small but inside noise, single-turn −2.5pt; the harm channel
  trained to 0%. Stage-3b spend ≈ $4.70. Evidence: ARTIFACTS Stage-3b section,
  `results/grpo-fr/`, the archived change's `tasks.md`.

The README's progression table is the public 30-second narrative — keep it in sync. The
self-correction support code (env tool loop, forced-revision rollout + eval runner, multi-turn
rewards + D8 diagnostics, `select-bestofn`, `fr-metrics`) stays in the tree, tested, for any
future attempt on a larger base.
