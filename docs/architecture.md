# Architecture

Core design: **the verifier that scores the baseline is the same code used as the RL training
reward.** A bug in eval is a bug in training, and vice versa. Generated result files carry enough
metadata to be scored without loading a model. The committed public bundle carries the compact
headline summary.

That verifier-as-reward idea lives in a **task-agnostic harness** (`harness/`): a task plugs in by
implementing two small protocols, `AnswerParser` (pull the answer out of a completion) and
`Verifier` (check it, return a `Verdict`), and bundling them as a `Task`. The shaped / near-binary
reward (`harness.score_answer`) is generic over those. **Text-to-SQL is the reference task**
(`tasks/sql/task.py`). To train your own model on your own verifiable problem, add a task. See
`docs/harness.md`.

The shipped result is single-turn RLVR on that SQL task: SFT cold-start followed by GRPO with
execution-verified rewards.

## Module owners and dependencies

| Module | Owns | Change here when |
| --- | --- | --- |
| `harness/` | Parser, verifier, task protocols, and reward shaping | A rule applies to every task |
| `tasks/sql/` | SQL parsing, prompts, execution-match, and the SQL task | A rule depends on SQL |
| `env/sandbox.py` | Read-only SQLite execution, time limits, and complete-result detection | SQLite safety or execution behavior changes |
| `data/` | Typed source examples, pinned loaders, database paths, and cache publication | Dataset identity or installation changes |
| `artifacts/` | Shared provenance, atomic files, and managed run manifests | Artifact identity or publication changes |
| `eval/` | SQL completion artifacts, generation, scoring, and seed aggregation | A saved evaluation result changes |
| `train/configuration.py` | Adapter identity and requested/effective runtime configuration | Model or training API compatibility changes |
| `train/prepared.py` | Prepared SFT rows, bound GRPO rewards, and final publication | Generic training execution changes |
| `train/checkpoints.py` | Independent periodic adapter provenance and completion | Checkpoint publication changes |
| `train/traces.py` | Candidate filtering, source lineage, and training rows | SFT trace eligibility changes |
| `train/sft.py`, `train/grpo.py` | Training entrypoints and task composition | A training request or its evidence changes |
| `report/` | Progression tables, public claims, and full re-scoring | A result is aggregated or reported |
| `cli.py`, `modal_helpers.py`, `modal_app.py` | Typed requests, dispatch, and execution boundaries | A local or remote command changes |

`harness/` has no SQL, data, evaluation, or training dependency. The SQL task does not depend on evaluation or training.
`train/prepared.py` accepts prepared rows and a reward function. It does not construct SQL resources.
Prepared SFT accepts validated conversations and a task-supplied eligibility proof. SQL SFT validates SQL trace lineage before composition.
Import contracts in `pyproject.toml` enforce these boundaries through `make imports`.

The SQL implementations live under `tasks/sql/`. The old `eval.verifier`, `eval.parsing`, and `eval.prompt` imports remain compatibility exports.
Shared provenance lives in `artifacts/provenance.py`. `eval.provenance` remains a compatibility export.
`artifacts/io.py` owns atomic publication. The old completion IO functions delegate to it.

Heavy dependencies import inside runtime functions. The installed CPU wheel can import every package module without the model stack.

## Data and result flow

```
Pinned datasets      generation / training       CPU scoring and reporting
Spider + BIRD    ->  saved completions       ->   score -> aggregate -> report
                                                results/public/ -> make eval
```

- **CompletionSet** (`eval/completions.py`) is the boundary object: `generate` (the *only*
  model-loading step) writes one JSON per (model, dataset, decoding) cell; `score` reads it anywhere,
  deterministically, with no generation backend.
- **Evidence split**: `results/public/` is the tracked clean-checkout summary that `make eval`
  validates. Raw CompletionSet/ScoreReport files are not part of the public quickstart.
- **Generation identity** records model revision, dataset revision, decoding, prompt recipe, selection policy, and available database provenance.
  Local seeded slices and remote prefix slices use different recorded policies.
  A database override has unknown archive identity unless the archive was actually verified.
- **Scoring identity** is separate from generation identity. New scores record the canonical completion-model digest, scorer policy, source digest, and actual database-file hashes.
  Version 1 completion identity uses a fixed field projection and its defined defaults. Future optional metadata cannot change that identity.
  Identity changes require explicit schema-version handling. These digests are not hashes of original JSON bytes.
  Source checkout Git identity never comes from the caller's unrelated repository. Installed wheels report unknown Git identity.
- **Legacy evidence** remains readable when new optional fields are absent. Unknown lineage, archive pins, or per-item outcomes stay unknown.

## Key types and patterns

- **`Frozen`** (`_base.py`) prevents attribute reassignment and rejects unknown keys.
  Score reasons and generation library versions use immutable mappings. Sequence evidence uses tuples.
  Score reports validate counts, reasons, rates, provenance, and available per-item outcomes.
  Progression rows validate their aggregates against the recorded seed values.
- **Pure builder + thin training driver**: config assembly is pure and CPU-testable
  (`grpo_config_kwargs`, `build_grpo_config`, `split_train_dev`, `filter_traces`); the `run_*`
  functions keep runtime work at the edge. Unit tests cover the builders; model-loading paths are
  integration-marked.
- **Lazy imports**: CLI subcommands and GPU backends (`vllm`, `trl`, `peft`, `torch`) import inside
  the functions that use them, so the CPU eval layer installs and runs without the training stack.
- **TRL version drift fails before training**. Unsupported requested fields produce an actionable error.
  Training provenance stores requested and effective TRL and LoRA settings. Warm-start adapter settings remain authoritative.
  GRPO defaults to the adapter's declared base revision. Explicit model refs resolve to commits before compatibility comparison.
  SQL population identity uses the fixed `sql-example-v1` semantic fields. Future optional metadata cannot change exposure comparisons.
  Known SFT exposure is recorded separately from the GRPO train/dev split. Historical missing lineage remains unknown.

## Artifact publication

Single-file CLI outputs publish atomically and reject existing paths. Library callers can request explicit overwrite.
Trace datasets publish their manifest last. SFT verifies the manifest hash, source lineage, filter counts, and trace bytes before model loading.

Managed remote and training runs use a fresh directory:

```text
request.json -> output files -> complete.json
```

The request exists before expensive execution. The completion marker lists validated output paths, sizes, and hashes.
Readers reject an incomplete managed directory or a changed output. Existing unmanaged historical artifacts remain readable.
Each periodic checkpoint receives its own request, provenance, and completion marker after successful training.
Checkpoint provenance records that checkpoint's weights and saved step. Population IDs do not claim which examples an intermediate step saw.
An incomplete checkpoint cannot inherit completion from its parent run.
Do not make an interrupted run look complete by creating its marker manually. Use a fresh run ID after resolving the failure.

Dataset archives use a temporary extraction tree. Validation precedes publication of the installed cache.
A partial extraction cannot become a valid cache merely because one expected file exists.

## The reward (Stage 2)

`harness.score_answer` composes one per-completion score from an `AnswerParser` + a `Verifier`;
`tasks/sql/task.py` binds those to SQL (the `SqlAnswerParser` + execution-match `SqlVerifier`) and adds
the empty-result guard. The shaping is task-agnostic; only the parser/verifier are SQL. Per
completion, from a single sandbox execution:

- **shaped arm** (SQL-R1-style): `format + executable + result_match + length`, config-driven
  weights (0.2 / 0.2 / 1.0 / 0.1). `format` gives full credit only when `<think>` *leads* the
  completion; a prose preamble caps at `preamble_credit=0.3`. The SFT model emits a preamble on
  ~99% of completions (~54% of generated tokens), and the reward pressure trains it toward the
  cheaper leading-`<think>` format.
- **near-binary arm** (Arctic-style): `1.0` exact / `0.1` executable / `0` else. This is the
  shaping-backfires ablation.
- **Empty-result guard** (`guard_trivial_empty=True` in training and trace filtering) rejects empty prediction/reference pairs.
  Public re-scoring retains the recorded verifier configuration.
- **Complete results**: truncated predictions fail verification. Truncated references produce a reference-query error.
  Inspect the example and database before changing `VerifierConfig.row_cap`.

## Evaluation discipline

- **Seed-level aggregation only** (`eval/aggregate.py`): mean + 95% Student-t CI over per-seed EX;
  rollouts are never pooled across seeds. The headline success check is a gain CI entirely above zero. Single-seed controls are point estimates with no CI claim
  (`report/progression.py` enforces both: ≥2 seeds renders a CI, n=1 renders "point").
- **EX is single-DB execution-match** with a disclosed ~6.5% false-positive caveat. Test-suite
  multi-DB accuracy would be stricter, but it is not part of the public result.
- Baselines report both `direct` (literature-comparable) and `<think>`-matched (before-RL anchor)
  prompts. The untrained base scores *lower* with `<think>`, and hiding that would inflate the gain.

## Supported Path

- Eval foundation: data loaders, sandbox, vendored verifier, and baseline.
- SFT cold-start: LoRA adapter that teaches the `<think>` + ```sql``` format.
- Single-turn GRPO: the shipped result with 3-seed confidence intervals, a near-binary ablation,
  and a Llama cross-family control.
