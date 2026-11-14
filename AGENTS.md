# llm-rlvr - agent instructions

Text-to-SQL post-training project: SFT + single-turn GRPO, complete at Stage 2. The public
headline is backed by a tracked summary bundle, external raw artifacts, controls, multi-seed CIs,
and disclosed caveats. Status and history live in `docs/internal/status.md`.

## Read this first, then route

| When you are… | Read |
|---|---|
| Touching any code, or new to the repo | `docs/architecture.md` |
| Adding an RLVR task, or plugging in a different model | `docs/harness.md` |
| Running baselines, training, or Modal commands | `docs/internal/runbook.md` |
| Citing a number, or pinning/locating an artifact | `docs/internal/artifacts.md` |
| Checking the current stage / what's done | `docs/internal/status.md` |
| Choosing checks or assessing change quality | `docs/internal/verification.md` |

If a doc disagrees with code, fix the doc in the same change.

## Commands

```bash
make check        # fast gate (CI runs the same): lock, Ruff, types, imports, tests, installed wheel
make slow         # verify all, then verify control (CI runs the same)
make fix          # auto-format + safe lint fixes
make eval         # validate/render the tracked public summary (CPU, model-free)
make eval-full    # full re-scoring, requires every restored external artifact pair
make runtime-check # explicit training API check, missing dependencies fail
make data         # fetch + verify the pinned Spider DBs
make help         # every target
```

Prove changes with `make verify` (`verify-llm-rlvr` skill at `.agents/skills/verify-llm-rlvr/`).

Everything runs through `uv` (`uv run …`, never bare `python`/`pip`). GPU work runs **only** via
`modal_app.py` entrypoints. Never run GPU work locally.

## Enforced Rules

| Rule | Enforced by | Added |
|---|---|---|
| Public evidence validates from a clean checkout and an installed wheel | `tests/test_public_evidence.py::test_tracked_public_evidence_passes_and_preserves_headline`, `scripts/verify_package.py::main` | 2026-11-05 |
| Aggregate per seed, then Student-t CI; one seed reports no interval | `src/llm_rlvr/eval/aggregate.py::SeedLevelStat`, `tests/test_aggregate.py::test_single_seed_reports_a_point_not_an_interval`, `tests/test_public_evidence.py::test_seed_count_and_interval_claim_are_derived` | 2026-07-01 |
| Training reward and trace filter keep `guard_trivial_empty=True` | `tests/test_harness.py::test_sql_training_reward_requires_empty_guard`, `tests/test_traces.py::test_trace_filter_cannot_disable_empty_guard` | 2026-11-05 |
| README headline table matches the tracked public summary | `src/llm_rlvr/report/public.py::_check_readme_table`, `tests/test_public_evidence.py::test_tracked_public_evidence_passes_and_preserves_headline` | 2026-11-05 |
| Datasets, databases, weights, and large files are never tracked | `tests/test_repo_rules.py::test_no_data_files_are_tracked` | 2026-11-06 |
| Unit tests open no TCP sockets; network tests are marked `integration` | `tests/test_repo_rules.py::test_unit_tests_cannot_open_tcp_sockets`, `Makefile::test-integration` | 2026-11-06 |
| Harness, SQL task, artifacts, and prepared training keep their import direction | `Makefile::imports`, `pyproject.toml::tool.importlinter` | 2026-11-05 |
| Generic training and the CPU wheel import without SQL or the model stack | `tests/test_import_architecture.py::test_generic_training_imports_without_sql_or_heavy_dependencies`, `scripts/verify_package.py::main` | 2026-11-05 |
| Package, scripts, tests, and Modal app pass basedpyright strict with no baseline | `Makefile::typecheck`, `pyproject.toml::typeCheckingMode` | 2026-11-06 |
| Modal entrypoints and GPU functions refuse to launch without approval | `src/llm_rlvr/modal_helpers.py::require_approval`, `tests/test_modal_launch_approval.py::test_every_gpu_function_gates_on_approval_first`, `tests/test_modal_launch_approval.py::test_modal_calls_follow_the_gate_and_forward_approval`, `tests/test_modal_launch_approval.py::test_every_modal_entrypoint_passes_its_approval_flag`, `tests/test_modal_preflight.py::test_direct_gpu_function_call_refuses_without_approval` | 2026-11-06 |
| `make verify` reports a pass only when its control observes FAIL | `.agents/skills/verify-llm-rlvr/drives.py::control_score`, `.agents/skills/verify-llm-rlvr/verify::verdict`, `Makefile::slow` | 2026-11-06 |
| Every citation in this table is defined where it points | `tests/test_repo_rules.py::test_enforced_rules_cite_existing_targets` | 2026-11-06 |
| No file downgrades strict type checking with a `# pyright:` mode comment | `tests/test_strict_mode.py::test_no_file_downgrades_type_checking_mode` | 2026-11-07 |
| Every pyright ignore names its rule and a reason; no blanket `noqa`, file-wide `# ruff: noqa`, or `noqa` on a module ban | `pyproject.toml::reportIgnoreCommentWithoutRule`, `pyproject.toml::tool.ruff.lint`, `tests/test_strict_mode.py::test_ignores_name_a_rule_and_a_reason_and_no_noqa_lifts_a_ban` | 2026-11-09 |
| Process, network, paid SDK, and Modal modules are imported only by named owner files, each held to its own modules | `pyproject.toml::tool.ruff.lint.flake8-tidy-imports.banned-api`, `tests/test_banned_modules.py::test_every_file_uses_only_the_banned_modules_it_owns`, `tests/test_banned_modules.py::test_only_named_owners_hold_a_tid251_ignore` | 2026-11-07 |

## Judgment calls

- Raw CompletionSet JSON, ScoreReport JSON, databases, and adapters stay external under
  gitignored `results/` or the Modal Volume. Restore them and run `make eval-full` to re-score.
- Single-seed runs are point estimates. Make no CI or significance claim from them.
- A Modal refusal prints the approved command. Run it only after Adam approves that run and its
  rough cost. Smokes and pilots validate the rig before any expensive run.
- Long Modal runs: `modal run --detach modal_app.py::<entrypoint>`. Long entrypoints spawn by
  default, so a local disconnect does not kill them. Stagger app creations (3 simultaneous hit a
  rate limit). The HF "set HF_TOKEN for higher rate limits" warning is benign.

## Conventions

- Typed records subclass `Frozen` (`_base.py`: pydantic frozen + `extra="forbid"`). No
  `dict[str, Any]` payloads for persisted artifacts.
- Pure, CPU-testable builders + thin GPU drivers (see architecture doc). Unit tests must not need
  a GPU; mark network tests `@pytest.mark.integration`.
- Heavy/optional deps (`vllm`, `trl`, `peft`, `torch`) import lazily inside the functions that
  use them.
- Import each name from its defining module: provenance and atomic publication from
  `artifacts/`, SQL parsing, prompts, and verifier from `tasks/sql/<module>.py`. The SQL task and
  reward are the one package-level surface: `llm_rlvr.tasks.sql`.
- New outputs use fresh paths. Never publish a completion marker for an interrupted or unvalidated run.
- Full reproduction code is tracked under `report/`. Do not depend on an external reproduction script.

