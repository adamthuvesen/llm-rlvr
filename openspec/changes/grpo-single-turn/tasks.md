# Tasks — grpo-single-turn (Stage 2: single-turn GRPO)

## 1. De-risk before paid training (GPU smoke + pairing)

- [x] 1.1 GPU colocate smoke RAN (2026-06-10, `grpo-smoke`, 32 train / 8 dev, 3 steps, exit 0): GRPO
  end-to-end on A100-80GB, vLLM colocate at `gpu_mem_util=0.3`, **no OOM**, CUDA graphs captured.
  Envelope confirmed (binary fits-on-A100-80GB; explicit peak-mem not instrumented — add an nvidia-smi
  log to the pilot if a precise number is wanted). RUNBOOK write lands with 8.2.
- [x] 1.1a Cost instruments captured: **step_time 13.4s** (per_device_batch 8, G=4), completions
  **mean 226 tok, clipped_ratio 0** (1024 cap comfortable). Extrapolated 3-seed headline ≈ **$8–22**
  (300→800 steps/seed at $2.50/hr) — far under the planning $50–60, and the preamble penalty will
  shrink completions during training. The LR pilot (6.1) pins steps-to-convergence; RUNBOOK in 8.2.
- [x] 1.1b Token-count check on the `sft-r32` warm-start: measured on the committed completions —
  the preamble is **~54% of every completion** (~180–195 tok of ~320–360 total, p95 ~500, so it does
  not truncate the 1024 cap but burns half the rollout budget). Trim approach decided + built: the
  reward `format` component (task 2.2) gives full credit only when `<think>` *leads* (preamble caps at
  `preamble_credit=0.3`), turning the waste into an RL signal toward the cheaper format. Verify the
  token drop at the LR pilot (6.1); add a prompt nudge there if the reward pressure alone is slow.
- [x] 1.2 TRL 1.5.1 ↔ vLLM 0.22.1 pairing VALIDATED on the smoke: runs GRPO steps cleanly;
  `sampling_logp_difference/mean` 0.0076 + `importance_sampling_ratio` mean 1.02 (min 0.43/max 2.25)
  → vLLM and trainer logprobs agree, so TRL's conservative "supports vLLM 0.12–0.18" warning is
  cosmetic for our locked combo. `frac_reward_zero_std` 0.042 (live signal), kl 4e-4 (stable
  warm-start). Also fixed: `build_grpo_config` now filters to GRPOConfig's accepted fields (the smoke
  caught `max_prompt_length` being rejected) — drift is a loud warning, not a crash.
- [x] 1.3 Resolve the multi-DB test-suite-DB question (design Open Questions). **Investigated:** the
  verifier already supports it — `execution_match(..., db_dir=...)` upgrades single-DB → test-suite
  accuracy with no code change; only the distilled Spider test-suite DB artifacts are missing.
  **DECIDED (2026-06-10): report single-DB EX with the ~6.5% false-positive caveat disclosed in the
  artifact.** Upgradeable later (verifier needs no code change) if we self-host the distilled DBs.
  → 5.3 scopes to caveat disclosure, not test-suite wiring.

## 2. Reward module (`reward-functions`)

- [x] 2.1 `src/text2sql_rlvr/rewards/`: a TRL reward-function over `(prompts, completions, gold)` that
  parses SQL, executes via the Stage-0 read-only sandbox, and scores result-match with the verifier —
  reusing `env.sandbox` + `eval.verifier`, never raising on unparseable/erroring SQL.
- [x] 2.2 Shaped arm (`format` + `executable` + `result_match` + `length`) with config-driven
  component weights; near-binary Arctic arm (`1.0`/`0.1`/`0`); arm + weights selected by a typed config.
- [x] 2.3 Empty-result guard: empty result earns full credit only when gold is genuinely empty.
- [x] 2.4 Unit tests against fixtures: result-match → full credit; non-executing → zero; empty-guard
  both directions; both arms return the specified tiers; determinism (same input → same reward); the
  sandbox is read-only (no mutation). No GPU needed — these run in `make check`.

## 3. GRPO training (`grpo-training`)

- [x] 3.1 `src/text2sql_rlvr/train/grpo.py`: a typed `GRPOConfig` wrapper (LoRA all-linear rank 32/
  alpha 64 bf16, `num_generations` 4, `max_completion_length` 1024–1536, LR + KL/beta + seed) and a
  trainer wrapper that warm-starts from the Stage-1 SFT adapter and wires in the reward module.
- [x] 3.2 `modal_app.py`: a `grpo_train` GPU function (single A100-80GB, vLLM colocate) + a local
  entrypoint mirroring the `sft` entrypoint; persist the adapter to the `text2sql-runs` Volume under a
  run-scoped path, base + SFT adapter untouched.
- [x] 3.3 Base-model-parametric: thread a base-model id through `grpo_train` (Qwen default,
  Llama-3.2-3B for the control) so one recipe trains both families.
- [x] 3.4 Held-out dev-EX slice + periodic checkpoints for dev-EX checkpoint-selection;
  `split_train_dev` holds out a disjoint, seed-deterministic slice (asserted disjoint + train-never-
  empty in tests), and `save_steps` writes periodic checkpoints. Selection itself is **post-hoc**
  (score the saved checkpoints on the held-out slice via the existing `generate`+`score` path —
  same pattern SFT used for per-epoch selection), exercised during the headline run (6.2).
- [x] 3.5 `eval/provenance.py`: record RL provenance — base + revision, warm-start adapter id, LoRA
  config, reward arm + weights, GRPO hyperparams + seed, dataset identity, TRL/vLLM/peft/transformers
  versions, git commit/dirty (back-compatible default when no reward config is present).
- [x] 3.6 Tests (unit, no GPU): config contract + completion-only/advantage wiring assertions, RL
  provenance recorded and round-trips, dev-slice disjointness check. Mark the live GPU GRPO step as an
  integration test deselected by default.

## 4. Base-model-parametric adapter eval (reuses `baseline-evaluation`)

- [x] 4.1 `eval/generate.py`: make the base-model id a parameter (defaults to Qwen — the base path
  stays byte-identical) so a Llama+adapter checkpoint evaluates through the existing adapter-eval path.
- [x] 4.2 Test: Qwen default path unchanged; a Llama base id routes to the Llama base + adapter; base
  identity recorded in provenance.

## 5. Multi-seed RL evaluation + reporting (`rl-evaluation`)

- [x] 5.1 Seed-level aggregation in `eval/aggregate.py`: `seed_level_ci` (mean + 95% Student-t CI over
  per-seed EX, point-only for n=1) and `rl_vs_sft_gain` (per-seed RL EX minus the greedy SFT anchor,
  with `excludes_zero` as the headline bar). Never pools rollouts across seeds; t-crit lookup table,
  no scipy. Exported from `text2sql_rlvr.eval`.
- [x] 5.2 DONE (2026-06-11): `make eval` renders base → SFT → RL on both benchmarks with
  seed-level CIs, the shaped-vs-near-binary ablation rows, AND the Llama cross-family staircase
  (two extra tables, 1-seed point-labeled).
- [x] 5.3 DONE: single-DB EX with the false-positive caveat emitted into every rendered table
  (`SINGLE_DB_CAVEAT` in `report/progression.py`; decision recorded in 1.3).
- [x] 5.4 Tests (unit, CPU) DONE: aggregation refuses cross-seed pooling, seed-level CI, 1-seed
  point-only, gain bar both directions (`tests/test_aggregate.py`); table renders rows + caveat
  from fixtures (`tests/test_report.py`); threaded scoring byte-identical at any worker count
  (`tests/test_score.py`).

## 6. LR pilot + headline run (paid GPU)

- [x] 6.1 LR pilot DONE (2026-06-10, `{5e-6,1e-5,5e-5}`, ~240 steps each on Spider-train limit 2000,
  eval on Spider-dev 300-slice). Monotonic gain over SFT (0.633): 5e-6 **0.713** (+8.0), 1e-5 **0.740**
  (+10.7), 5e-5 **0.783** (+15.0); all 100% well-formed, no instability (zero_std ≤0.29, smooth reward).
  **LOCKED LR = 5e-5** (user-approved: best measured EX; dev-EX early-stop + checkpoint-selection guards
  the KL-0.21 drift over the longer headline). GRPO improves EX, not just format — core thesis holds.
- [x] 6.2 3-seed Qwen headline DONE (2026-06-10, `grpo-s0/s1/s2` @ LR 5e-5 shaped, full Spider-train
  6500, warm-start sft-r32). Evaled full Spider-dev + BIRD-dev, `CompletionSet`s committed under
  `results/grpo/`. (Network blip killed the first non-detached attempt → re-ran with `--detach
  --spawn`. ~$22.)
- [x] 6.3 SUCCESS BAR MET on both benchmarks (seed-level 95% CI excludes zero):
  **Spider-dev 0.678→0.779 (+10.2pt, CI [+5.8,+14.5])**; **BIRD-dev 0.344→0.406 (+6.1pt, CI
  [+5.6,+6.7])** — headline gain transfers to BIRD, 100% well-formed all seeds, BIRD seeds tight
  (0.404/0.405/0.408). base→SFT→RL = 0.313→0.344→0.406. The thesis holds: RL improves EX.

## 7. Controls (paid GPU, folded into the Stage-2 deliverable)

- [x] 7.1 Cross-family control DONE (2026-06-11, ~$18): **FULL matched SFT→GRPO on Llama-3.2-3B**
  (user chose the rigorous arm over a confounded base→RL; ungated `unsloth/` mirror, byte-identical
  weights, recorded in provenance). Same recipe end-to-end: trace-sample (2501/7000 kept, 36% vs
  Qwen 42%) → SFT `sft-llama-r32` → GRPO `grpo-llama-s0` @ LR 5e-5 shaped. **REPLICATES, stronger
  than the original:** Spider 0.459→0.596→**0.738** (RL +14.2pt vs Qwen +10.2); BIRD
  0.179→0.233→**0.314** (RL +8.0pt vs Qwen +6.1); well-formed 2%→100%. Weaker base, bigger RL
  gain (headroom). The headline is NOT a Qwen artifact. 1-seed point estimates, labeled, no CI
  claim. Artifacts: `results/llama/` (6 CompletionSets + scores), rendered in `make eval`.
- [x] 7.2 Reward-shaping ablation DONE (2026-06-11, `grpo-nb-s0` @ LR 5e-5 near-binary seed 0,
  ~5h train + evals, ~$14; shaped arm = headline seed 0 as planned). Full-dev, matched config:
  **Spider 0.774 vs shaped-s0 0.779** (tie, inside the shaped seed band 0.762–0.797);
  **BIRD 0.377 vs 0.408** (−3.1pt, below the shaped seed band 0.404–0.408). Well-formed dips
  without the format term: 98.7% Spider / 95.3% BIRD (shaped: 100/100). Training-dynamics finding:
  near-binary starves its own signal as it improves — `frac_reward_zero_std` 0.59→0.78 vs ~0.04
  shaped (all-identical reward groups teach GRPO nothing). VERDICT: both arms beat SFT decisively
  (gains are not a shaping artifact — Arctic's worry doesn't bite here), and shaping wins on the
  harder headline benchmark + format robustness. 1-seed point estimates, no CI claim.
  Artifacts: `results/grpo/{spider,bird}-dev-zs-grpo-nb-s0.json` + `.score.json`.

## 8. Re-derivation, docs, and the gate

- [x] 8.1 DONE (2026-06-11): `make eval` re-derives all 20 committed sets (base/SFT/3 GRPO
  seeds/ablation/Llama staircase, both benchmarks) + renders all four progression tables in
  **9:03 wall** — `score_completion_set(workers=)` threads the DB execution (~5× on BIRD),
  byte-identical reports at any worker count (tested).
- [ ] 8.2 `RUNBOOK.md`: the GRPO procedure end-to-end (smoke → LR pilot → headline → controls), the
  Modal commands, the seed policy, the fallback ladder for colocate OOM.
- [ ] 8.3 `ARTIFACTS.md`: the GRPO adapters (headline 3 seeds + controls) as pinned artifacts, the
  locked LR, the reward config, and the committed RL-vs-SFT-vs-base table.
- [ ] 8.4 `make check` green: new unit tests pass; GPU / TRL GRPO tests are integration (deselected by
  default).
- [ ] 8.5 Record the **GATE** decision: is the single-turn gain strong/stable enough (CI excludes zero
  on BIRD) to approve Stage-3 spend, or do we ship single-turn only?
