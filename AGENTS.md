# llm-rlvr-sql — agent instructions

Text-to-SQL post-training project: SFT + single-turn GRPO worked; the later self-correction
follow-up did not. Every claimed gain is backed by committed results, controls, multi-seed CIs,
and disclosed caveats.

Global tone, principles, and git etiquette live in the user's agent defaults; they are not
duplicated here. This file is project-specific facts.

## Read this first, then route

| When you are… | Read |
|---|---|
| Touching any code, or new to the repo | `docs/architecture.md` |
| Running baselines, training, or Modal commands | `RUNBOOK.md` |
| Citing a number, or pinning/locating an artifact | `ARTIFACTS.md` |
| Checking the current stage / what's done | `docs/status.md` |
| Changing behavior or scope | the active change in `openspec/changes/` (work the tasks; don't diverge silently) |

If a doc disagrees with code, fix the doc in the same change.

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

## Current state

The project is complete at Stage 2 (single-turn GRPO); two self-correction sequels closed
honestly without a full training run. Full stage-by-stage status → `docs/status.md`; keep the
README progression table in sync with it.
