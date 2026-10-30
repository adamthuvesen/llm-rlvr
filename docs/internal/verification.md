# Verification and change quality

A change is ready when its behavior, ownership, and verification are clear.
Use this rubric to assess a concrete diff. It is engineering judgment, rather than a measured product metric.

| Dimension | Required evidence |
| --- | --- |
| Discoverability | The architecture names the owner, CLI help names inputs, and `make help` exposes the relevant check |
| Ownership | Each rule has one implementation home, and callers import it from that module |
| Valid states | Typed records reject contradictory counts, metadata, runtime options, and artifact states |
| Boundaries | Import contracts enforce dependency direction, and runtime preflight precedes model loading or publication |
| Useful feedback | Errors identify the failing input, location, and corrective action without a false-success path |
| Behavioral correctness | Tests exercise actual SQL, artifact IO, CLI execution, and the reported claims |
| Future changes | Focused checks cover the owner, and an installed-wheel check catches source-checkout assumptions |

If a numeric assessment helps compare revisions, use 0 for missing evidence, 1 for partial evidence, and 2 for verified evidence.
These anchors define review language. They do not establish a target score or substitute for a demonstrated defect.

## Normal gate

```bash
make install
make check
```

`make check` checks the lockfile, Ruff, package and entrypoint types, import contracts, unit tests, and the installed wheel.
The wheel check (`make package-check`) installs locked CPU dependencies in a temporary environment outside the checkout.
It imports package modules with model dependencies blocked, then runs CLI help, public evidence, and the SQLite demo.

CI configures these checks on Python 3.14. A local pass does not establish a hosted-CI pass.

## Failure tests

`make check` runs every unit test in parallel. To iterate on one owner, select its files:

```bash
uv run --locked pytest tests/test_score.py tests/test_aggregate.py tests/test_report.py
uv run --locked pytest tests/test_public_evidence.py tests/test_full_reproduction.py
uv run --locked pytest tests/test_completions.py tests/test_run_lifecycle.py
uv run --locked pytest tests/test_sandbox.py tests/test_verifier.py tests/test_verifier_parity.py
uv run --locked pytest tests/test_modal_preflight.py tests/test_training_runtime.py
```

These checks reject forged rates, duplicate seeds, false positive-gain claims, truncated results, incomplete runs, and changed artifact hashes.
Full reproduction checks every raw input before scoring. Equal aggregate counts do not hide changed per-item outcomes when historical outcomes exist.
A legacy report without outcomes cannot establish per-item agreement.

Replay known failure classes when changing their owners: model-revision forwarding, BIRD split validation, SQLite URI paths, and clean-checkout public evidence.
Use the actual dataset integration checks for dataset changes:

```bash
make test-integration
```

## Optional runtime evidence

```bash
make runtime-check
```

This command requires installed training dependencies. A missing dependency fails with an install command instead of skipping.
The check constructs real training configurations. It does not download models or run GPU training.

Paid GPU launches still require approval and a cost estimate. A local mock or configuration check cannot establish live trainer behavior.

## Result evidence

`make eval` checks the tracked public summary, claim references, README table, and seed-level reporting.
It preserves the existing public counts and statistical conventions.

`make eval-full` requires the restored external completion/report pairs and matching databases.
It reports score deltas and reference failures while preserving historical reports.
A restored file hash identifies current bytes. It cannot supply a missing historical pin.
Unknown SFT lineage, database-archive identity, or historical per-item outcomes remain explicit limits.
