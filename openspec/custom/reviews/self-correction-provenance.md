# Code Review: self-correction provenance rename

Scope: `main..HEAD` on branch `adam/refactor-self-correction-provenance`.

## Critical

No findings.

## High

No findings.

## Medium

No findings.

## Low

No findings.

## Areas Reviewed & Found Clean

- Rename coverage: `stage3_provenance.py` was renamed to `self_correction_provenance.py`, and the shared base class/type aliases were renamed consistently. `MultiTurnProvenance` and `ForcedRevisionProvenance` now inherit from `SelfCorrectionGrpoProvenance`; no tracked stale `stage3_provenance` or `Stage3GrpoProvenance` references remain.
- Provenance behavior: the shared fields (`grpo`, `environment`, `multiturn_reward`, `rollout`, `prompt_style`) stayed intact, and the narrower rollout literals in the multi-turn and forced-revision subclasses still preserve their existing defaults.
- CLI/docs wording: the CLI help now describes these commands as self-correction analysis utilities, and the architecture module map points at `self_correction_provenance.py`.
- Reproducibility/docs: artifact wording still preserves the pinned Spider/BIRD sources, checksums, public evidence policy, and Stage 1/2 result descriptions.
- Security: no new path handling, SQL construction, network calls, secrets handling, or permission changes were introduced.
- Performance: no runtime algorithm or data-path changes were introduced; the branch is a naming/docs refactor.
- DS/ML correctness: no changes to reward scoring, empty-result guard behavior, seed aggregation, train/dev splitting, hyperparameters, checkpoint selection, or eval math.
- Test coverage: existing provenance and training config tests cover the touched import paths and Pydantic round trips.

## Checks Run

- `uv sync --extra dev` — passed; installed/synced the declared dev environment without downloads.
- `uv run python -m pytest tests/test_forced_revision.py tests/test_multiturn.py tests/test_multiturn_diagnostics.py tests/test_sft.py -q` — passed: 69 passed, 1 skipped.
- `make check` — passed outside the sandbox after uv cache access was allowed: ruff check passed, format check passed, 241 passed, 1 skipped, 3 deselected.
- `uv run llm-rlvr --help` — passed; help output shows the updated self-correction analysis wording.
- `uv run python -c "... import SelfCorrectionGrpoProvenance ..."` — passed; both provenance subclasses resolve through the renamed base class.
- `git diff --check main..HEAD` — passed.

## Summary

| Severity | Count |
| --- | ---: |
| Critical | 0 |
| High | 0 |
| Medium | 0 |
| Low | 0 |

Overall: no bugs, security issues, performance regressions, DS/ML issues, missing-test concerns, or documentation mismatches found in this branch scope.
