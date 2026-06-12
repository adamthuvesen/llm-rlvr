# rlvr-sql-agent

Text-to-SQL with RL on verifiable execution rewards (RLVR): **SFT cold-start → single-turn GRPO**,
gains verified across seeds, reward designs, and model families — then two carefully-instrumented
**negative results** on training the self-correction loop itself, each closed by a cheap
pre-registered probe or gate instead of an expensive run. Every claim re-derives from a committed
artifact (`make eval`); every paid run sat behind a smoke and a pilot with thresholds committed
before the spend.

This is the build-and-train companion to
[`grpo-gain-decomposition`](https://github.com/adamthuvesen/grpo-gain-decomposition), which measured
whether RL gains are real.

## The post-training progression

The shipped headline is **Stage 2**: Spider-dev 0.625 → 0.678 → **0.779**, BIRD-dev
0.313 → 0.344 → **0.406** (base → SFT → RL; 3 seeds, the RL-vs-SFT gain CI excludes zero on both
benchmarks). Every rung after it was gated, and the gates are the story:

| Rung | What was tried | Verdict | Record |
|---|---|---|---|
| **Baseline** (Qwen2.5-Coder-3B) | Greedy zero/few-shot, two prompt styles disclosed (`<think>` 0.625 / direct 0.669 Spider-dev) | reference | [`ARTIFACTS.md`](ARTIFACTS.md), [`RUNBOOK.md`](RUNBOOK.md) |
| **Stage 1 — SFT cold-start** | LoRA on self-generated, execution-filtered traces | **POSITIVE** — format 3%→99%, +5.3pt Spider / +3.1pt BIRD | [`archive/2026-06-10-sft-coldstart`](openspec/changes/archive/2026-06-10-sft-coldstart/) |
| **Stage 2 — single-turn GRPO** | Execution-reward RL on the SFT adapter | **SHIPPED** — +10.2pt Spider / +6.1pt BIRD over SFT (3-seed CI excludes 0); holds under near-binary reward ablation and on Llama-3.2-3B | [`archive/2026-06-11-grpo-single-turn`](openspec/changes/archive/2026-06-11-grpo-single-turn/), [`ARTIFACTS.md`](ARTIFACTS.md) |
| **Stage 3a — multi-turn tool election** | Offer a `run_sql` tool; train the loop with GRPO | **FALSIFIED** — the 3B policy never elects to call the tool: 0/672 episodes across base / SFT / GRPO / two tool-use-SFT checkpoints and every prompting strength (incl. in-context demos) | [`archive/2026-06-12-grpo-multi-turn`](openspec/changes/archive/2026-06-12-grpo-multi-turn/tasks.md) |
| ↳ tool-use SFT cold-start (×2 recipes) | 900 supervised tool-call targets, then a fully diagnosed v2 | **NEGATIVE, diagnosed** — token-mass imbalance taught the wrong turn-1 policy; v2 fixed every diagnosed fault and still 0 tool calls | same archive, task 2.1 journal |
| ↳ entrenchment probe (~$1, pre-registered) | Is RL formatting entrenchment the blocker? Raw base + pre-GRPO arms | **FALSIFIED** — raw base also 0/96: protocol non-adoption at this scale, not entrenchment. Dodged a ~$10–20 sequel | same archive + [`ARTIFACTS.md`](ARTIFACTS.md) |
| **Stage 3b — revision probe** (~$0.30, pre-registered) | Remove election: execute every answer, show feedback, measure *untrained* revision | **PASS** — fix rate 14.9% (≥10% gate), harm 6.3% (≤10% gate); untrained net −1.65pt | [`results/grpo-fr/revision-probe.json`](results/grpo-fr/revision-probe.json) |
| **Stage 3b — forced-revision GRPO** | Hand-written rollout: attempt → execute → feedback (loss-masked) → revise; smoke green (masking + IS-ratio validated on real spliced rollouts) | **NEGATIVE at the pre-registered pilot gate** — GRPO drifts toward identity-resubmission (revision rate 0.26→0.05, fix-channel bonus below its untrained baseline); aborted before the full run. At greedy eval the pilot ckpt does show net +4.5pt from revision with 0% harm on a 200-item slice — but inside noise, on a −2.5pt-degraded single-turn policy, with eroding dynamics | [`archive/2026-06-12-grpo-forced-revision`](openspec/changes/archive/2026-06-12-grpo-forced-revision/tasks.md), [`results/grpo-fr/`](results/grpo-fr/) |

Total spend on closing stage 3 honestly: ≈ $13 (vs ≥ $60 of unguarded full runs). The
self-correction *scaffolding* — env tool loop, forced-revision rollout, multi-turn rewards +
collapse diagnostics, best-of-N control, fix/harm metrics — stays in the tree, tested, for an
attempt on a larger base model.

`openspec/changes/archive/` holds the full decision record for every stage; numbers in this table
re-derive with `make data && make eval`.

## Layout

```
src/text2sql_rlvr/
  data/         Spider/BIRD loaders, typed examples, schema serialization
  env/          read-only SQLite sandbox (the safety boundary) + the run_sql tool env
  eval/         execution-match verifier, generation, scoring, provenance,
                forced-revision + multi-turn inference harnesses, best-of-N control
  rewards/      shaped + near-binary single-turn arms; multi-turn trajectory credit
  train/        STaR trace sampling, LoRA SFT, GRPO, forced-revision rollout, diagnostics
  report/       progression tables (seed-level CIs, disclosed caveats)
  third_party/  vendored Test-Suite SQL Eval (execution-match path only; Apache 2.0)
```

## Usage

```bash
make install   # CPU env: data, env, verifier, eval (no GPU stack)
make check     # ruff + unit tests (the Stage-0 gate)
make test-integration   # network/HuggingFace/model tests (deselected by default)
```

The baseline runs through the `text2sql-rlvr` CLI: `generate` writes a completion set (the only
model-loading step), then `score` reads it on a CPU box. See `RUNBOOK.md`.

## Data & licensing

Spider is CC-BY-SA-4.0 (attribute + share-alike); the vendored evaluator is Apache 2.0 (see
`src/text2sql_rlvr/third_party/test_suite_sql_eval/`). Datasets and databases are pinned + fetched,
never committed.

## License

MIT (this repository's own code).
