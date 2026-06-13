# Reproducibility manifest

Every external artifact this project depends on, pinned. Code pins live in
[`src/text2sql_rlvr/data/artifacts.py`](src/text2sql_rlvr/data/artifacts.py); this file is the
human-readable mirror. The main result is reproducible from a committed `CompletionSet` with
`make data && make eval` (CPU only, no model).

## Datasets

| Artifact | Source | Pin | Integrity |
| --- | --- | --- | --- |
| Spider Q/SQL pairs | HF `xlangai/spider` (parquet) | revision `0c350918f3f29ec754f1181c65cdce76cd6c133c` | HF commit (immutable) |
| Spider SQLite databases | HF `HAL-9001/spider-databases` → `spider_data.zip` | commit `4a01bbac6520cd35b216db9e1724e5e1ada60aa4` | SHA256 `00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b` |
| Spider DB provenance origin | Yale (canonical) | Google Drive id `1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J` | re-hosted under CC-BY-SA-4.0 |
| BIRD-dev bundle (examples + schemas + databases) | HF `HAL-9001/bird-dev` → `dev.zip` | commit `837584317bf7e3d86a3f6d257bd5553b7e03e171` | SHA256 `cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630` |
| BIRD-dev provenance origin | bird-bench.github.io (2024-06-27 release) | `dev.zip` | re-hosted under **CC-BY-NC-4.0** (non-commercial) |
| Distilled test-suite DBs (eval-time test-suite accuracy) | Google Drive id `1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w` | **TBD** — R4 fast-follow | — |

`load_dataset("xlangai/spider")` ships only the parquet; the SQLite databases are the separate,
checksummed artifact above (design D2/R1). BIRD's `dev.zip` is one self-hosted bundle.

## Code / models

| Artifact | Pin | License |
| --- | --- | --- |
| Base model | `Qwen/Qwen2.5-Coder-3B-Instruct` at HF `main`; the SFT run pins commit `488639f1ff808d1d3d0ba301aef8c11461451ec5` | Apache 2.0 |
| Cross-family base | `unsloth/Llama-3.2-3B-Instruct` (ungated mirror, byte-identical weights to gated `meta-llama/Llama-3.2-3B-Instruct`; repo id recorded in every artifact's provenance) | Llama 3.2 Community |
| Stage-1 LoRA adapter `sft-r32` | Modal Volume `text2sql-runs` → `sft/sft-r32/` (provenance.json: base commit above, rank 32 all-linear, trace-dataset SHA256, trl 1.5.1 / peft 0.19.1) | — |
| Stage-1 trace dataset | 2,971 traces; manifest `results/sft/traces.manifest.json`; JSONL SHA256 `b02435e49f183425299f3167503883d0b006faf884f7705fc02abc3ad34613e6` (bulk on the Volume at `sft/traces.jsonl`) | self-generated |
| Stage-2 GRPO adapters (main result) | Volume → `grpo/grpo-s0`, `grpo/grpo-s1`, `grpo/grpo-s2` — **locked LR 5e-5** (pilot-measured), shaped reward, seeds 0/1/2, warm-start `sft/sft-r32`; full config in each `provenance.json` | — |
| Stage-2 reward ablation adapter | Volume → `grpo/grpo-nb-s0` — identical config, `near_binary` arm (1.0/0.1/0) | — |
| Cross-family control adapters | Volume → `sft/sft-llama-r32` (trace manifest `results/sft/traces-llama.manifest.json`, 2,501/7,000 kept) and `grpo/grpo-llama-s0` (same recipe @ LR 5e-5 shaped, warm-start `sft/sft-llama-r32`) | — |
| Reward config (shaped, the main arm) | `rewards/reward.py` `RewardConfig`: weights format 0.2 / executable 0.2 / result_match 1.0 / length 0.1; `preamble_credit=0.3` (full format credit only when `<think>` leads); `guard_trivial_empty=True` | — |
| Vendored verifier | `taoyds/test-suite-sql-eval` — execution-match path only (`exec_eval.py` + `parse.py`) | Apache 2.0 |

The verifier is parity-tested against upstream (`tests/test_verifier_parity.py`); the only
modifications to vendored code are documented in
`src/text2sql_rlvr/third_party/test_suite_sql_eval/ATTRIBUTION.md`.

## Committed evaluation artifacts

The base-model baselines (Coder-3B-Instruct, greedy, **full dev**), each a committed
`CompletionSet` + `ScoreReport`. `make eval` re-derives the whole table on CPU via
`results/baseline/reproduce.py`:

| File | Setting | EX |
| --- | --- | --- |
| `spider-dev-zs.json` | Spider-dev, `<think>` zero-shot | 0.6248 (646/1034) |
| `spider-dev-fs3.json` | Spider-dev, `<think>` few-shot k=3 | 0.6770 (700/1034) |
| `spider-dev-zs-direct.json` | Spider-dev, direct zero-shot | 0.6692 (692/1034) |
| `bird-dev-zs.json` | BIRD-dev, `<think>` zero-shot | 0.3129 (480/1534) |
| `bird-dev-zs-direct.json` | BIRD-dev, direct zero-shot | 0.3514 (539/1534) |

Each `CompletionSet` carries its own provenance (model, greedy decoding, dataset revision,
DB-archive SHA256, **`prompt_style`**, few-shot recipe, adapter, git commit), so every number is
traceable to exactly the inputs that produced it. The two prompt styles (`<think>` matched vs
direct) are reported deliberately — see RUNBOOK § *Methodology*. The 50-example `*-n50` slices
remain as the local wiring proof. `*.score.json` files are the committed `ScoreReport`s.

## Stage 1: SFT cold-start (`results/sft/`, adapter `sft-r32`)

Same pipeline with `--adapter sft/sft-r32`; `make eval` re-derives these rows too. Success bar
(D6): well-formed rate near-100% **and** no EX regression below the base `<think>` anchor — met.

| File | Setting | EX | Well-formed |
| --- | --- | --- | --- |
| `spider-dev-zs-sft-r32.json` | Spider-dev, `<think>` zero-shot + sft-r32 | 0.6779 (701/1034) | 0.9923 |
| `spider-dev-zs-direct-sft-r32.json` | Spider-dev, direct zero-shot + sft-r32 | 0.7118 (736/1034) | — |
| `bird-dev-zs-sft-r32.json` | BIRD-dev, `<think>` zero-shot + sft-r32 | 0.3442 (528/1534) | 0.9602 |

SFT-vs-base: Spider-dev `<think>` 0.6248 → **0.6779** (+5.3pt, format 3% → 99%); BIRD-dev
`<think>` 0.3129 → **0.3442** (+3.1pt, format 4% → 96%); the adapter also lifts direct mode
(0.6692 → 0.7118). The direct rows score 0 well-formed by construction — that prompt asks for
bare SQL, no `<think>` block.

## Stage 2: single-turn GRPO (`results/grpo/` + `results/llama/`)

Same eval pipeline with the GRPO adapters; `make eval` re-derives every row AND renders the four
progression tables (seed-level 95% CIs for the 3-seed main result; single-seed rows labeled point
estimates; the single-DB caveat emitted into each table). All EX is single-DB execution-match
with the documented ~6.5% false-positive caveat.

**Decoding disclosure:** all evals are greedy (temperature 0.0; recorded in each provenance), but
the RL/Llama evals used `max_new_tokens=1024` vs the base/SFT evals' 512. Quantified impact:
0/1034 Spider and 1–2/1534 BIRD completions per RL seed exceed the ~512-token envelope, bounding
the effect at ≤0.13% EX on BIRD — two orders of magnitude below the +6.1pt gain it could at most
inflate. Disclosed rather than re-run.

**Main result (Qwen, 3 seeds @ LR 5e-5 shaped)** — success check met on both benchmarks (the
RL-vs-SFT gain CI excludes zero):

| File(s) | Setting | EX | Well-formed |
| --- | --- | --- | --- |
| `grpo/spider-dev-zs-grpo-s{0,1,2}.json` | Spider-dev, `<think>` + grpo-s{0,1,2} | 0.7795 / 0.7621 / 0.7969 — mean **0.779**, Δ vs SFT **+10.2pt** CI [+5.8, +14.5] | 1.00 |
| `grpo/bird-dev-zs-grpo-s{0,1,2}.json` | BIRD-dev, `<think>` + grpo-s{0,1,2} | 0.4081 / 0.4048 / 0.4042 — mean **0.406**, Δ vs SFT **+6.1pt** CI [+5.6, +6.7] | 1.00 |

**Reward-shaping ablation (1 seed, point estimates)** — both arms beat SFT (the gain is not a
shaping artifact); shaped wins on the harder benchmark and keeps a higher well-formed rate. Training
dynamics: the near-binary arm starves its own signal as it improves (`frac_reward_zero_std`
0.59→0.78 vs ~0.04 shaped — all-identical reward groups teach GRPO nothing):

| File | Setting | EX | Well-formed |
| --- | --- | --- | --- |
| `grpo/spider-dev-zs-grpo-nb-s0.json` | Spider-dev, near-binary | 0.7737 (vs shaped-s0 0.7795 — tie, inside the seed band) | 0.9874 |
| `grpo/bird-dev-zs-grpo-nb-s0.json` | BIRD-dev, near-binary | 0.3768 (vs shaped-s0 0.4081 — −3.1pt, outside the seed band) | 0.9531 |

**Cross-family control (Llama-3.2-3B, full matched SFT→GRPO, 1 seed, point estimates)** — the
same recipe works on another 3B model, with a larger RL gain than Qwen's (weaker base, more
headroom). The main result is not just a Qwen quirk:

| File | Setting | EX | Well-formed |
| --- | --- | --- | --- |
| `llama/spider-dev-zs-llama-3.2-3b-instruct.json` | Spider-dev, Llama base `<think>` | 0.4594 | 0.0213 |
| `llama/spider-dev-zs-llama-3.2-3b-instruct-sft-llama-r32.json` | + SFT | 0.5957 (+13.6pt) | 0.9874 |
| `llama/spider-dev-zs-llama-3.2-3b-instruct-grpo-llama-s0.json` | + GRPO | **0.7379** (+14.2pt) | 1.0000 |
| `llama/bird-dev-zs-llama-3.2-3b-instruct.json` | BIRD-dev, Llama base `<think>` | 0.1793 | 0.0046 |
| `llama/bird-dev-zs-llama-3.2-3b-instruct-sft-llama-r32.json` | + SFT | 0.2334 (+5.4pt) | 0.9544 |
| `llama/bird-dev-zs-llama-3.2-3b-instruct-grpo-llama-s0.json` | + GRPO | **0.3136** (+8.0pt) | 0.9948 |

The full progression (Qwen): base 0.625 → SFT 0.678 → RL **0.779** on Spider-dev;
0.313 → 0.344 → **0.406** on BIRD-dev. RL contributed ~2× SFT's gain on both benchmarks and
both model families, using the same training questions — the learning signal, not new data.

## Stage 3: multi-turn GRPO — CLOSED without training (`results/grpo-mt/`)

**Decision (2026-06-12): ship Stage 2 standalone.** The change (archived under
`openspec/changes/archive/`) stopped after ≈ $8 of smoke/gate evidence; no multi-turn model was
trained, so the Stage-2 progression above remains the main result. The finding: **the 3B model
never adopts the `run_sql` tool protocol** — 0 tool calls in 672 sampled episodes across every
checkpoint and prompting strength. A two-arm probe ruled out RL formatting as the cause: the raw
base model also answers directly, 0/96. TRL #4543 (multi-turn IS correctness) was clear in our
envelope on every census. All files are committed with full metadata:

| File | What | Result |
| --- | --- | --- |
| `grpo-mt/smoke-report-attempt1.json` | GRPO ckpt, `cot` prompt | 0/96 tool calls; IS 0.979 in band |
| `grpo-mt/smoke-report-attempt2.json` | GRPO ckpt, `mt` instruction | 0/96; IS 0.963 |
| `grpo-mt/smoke-report-attempt3.json` | GRPO ckpt, `mt_demo` in-context demonstration | 0/96; IS 0.999 |
| `grpo-mt/smoke-report-attempt4.json` | + tool-use SFT v1 (`tooluse-sft-r32`) | 0/96; model imitates feedback text |
| `grpo-mt/smoke-report-attempt5.json` | + tool-use SFT v2 (`tooluse-sft2-r32`, all diagnosed fixes) | 0/96; degraded generation |
| `grpo-mt/smoke-report-probe-base.json` | PROBE: raw base (identity LoRA), `mt_demo` | 0/96 — ruled out RL formatting as the blocker |
| `grpo-mt/smoke-report-probe-sft.json` | PROBE: Stage-1 SFT ckpt, `mt_demo` | 0/96 |
| `grpo-mt/{spider,bird}-dev-zs-tooluse-sft-r32.json` | SFT-v1 forgetting gate (single-turn) | 0.6983 (−8.1pt) / 0.3748 (−3.3pt) |
| `grpo-mt/{spider,bird}-dev-mt-tooluse-sft-r32(.transcripts).json` | SFT-v1 attribution baseline (multi-turn, no RL) | 0.1151 / 0.0469; 0 tool calls, transcripts persisted |
| `grpo-mt/tooluse.manifest.json`, `grpo-mt/tooluse-v2.manifest.json` | tool-use SFT dataset manifests | 3,000 / 2,600 rows, sandbox-verified feedback |

The Stage-3 support code (env tool loop, multi-turn reward + diagnostics, smoke driver,
`build-tooluse` dataset builder, `mt_eval` multi-turn inference eval) remains in the tree,
tested, for any future attempt on a larger base model.

## Stage 3b: forced-revision GRPO — ABORTED at the pilot gate (`results/grpo-fr/`)

**Decision (2026-06-12): closed as an honest negative; no full run.** Sequel to the Stage-3
closure: tool choice was the failed step, so this change removed it. The runner executes every
answer and asks the model to revise after seeing feedback (SCoRe / ReEx-SQL shape), built as a
hand-written TRL `rollout_func` with feedback-token loss masking. The sequence ran
probe → build → smoke → pilot and stopped at the pilot gate set before the run. Stage-2 artifacts
untouched.

| File | What | Result |
| --- | --- | --- |
| `grpo-fr/revision-probe.json` | Untrained revision probe | **PASS** — fix 14.9% (G1 ≥ 10%), harm 6.3% (G2 ≤ 10%); untrained net −1.65pt |
| `grpo-fr/smoke-report.json` | Smoke test (`grpo-fr-smoke2`, 96 episodes) | **PASS** — feedback masked in 96/96 episodes, 0 gradient leaks; IS 0.982 in band; 0 NaNs; 15.0 s/step |
| `grpo-fr/pilot-d8-diagnostics.json` | Pilot D8 series (60 steps @ LR 5e-5, 1,920 episodes) | **GATE FAIL** — P1 turn-1 0.737 < 0.754; P2 delta-bonus 0.008 < 0.01 (below untrained 0.0094); frac_revising 0.26→0.05 (identity-resubmission drift); `frac_reward_zero_std` ≈ 0.56–0.71 (weak learning signal) |
| `grpo-fr/spider-dev-zs-grpo-fr-pilot-n200.json` | Pilot ckpt, single-turn greedy (n=200) | 0.6800 — −2.5pt vs the Stage-2 anchor |
| `grpo-fr/spider-dev-fr-grpo-fr-pilot-n200(.transcripts).json` | Pilot ckpt under the forced-revision runner (2 generations) | 0.7200; attempt 0.6750 → final 0.7200 (net +4.5pt); fix 9/65 (13.8%, all error-class), **harm 0/135** (untrained: 6.3%) |
| `grpo-fr/spider-dev-zs-grpo-fr-pilot-t1.0-seed{0,1}-n200.json` | The matched-compute control's sampled passes | 0.6500 / 0.6600 |
| `grpo-fr/spider-dev-bo2-grpo-fr-pilot-n200(.selection).json` | Best-of-2, gold-free selection (D6 control) | 0.7150 (selection report committed) |
| `grpo-fr/spider-dev-fr-grpo-fr-pilot-n200.metrics.json` | `fr-metrics` fix/harm decomposition | committed; re-derives via `make eval` |
| `grpo-fr/spider-dev-fr-error-only-grpo-fr-pilot-n200(.report).json` | CPU replay: revise only when first-attempt feedback is an error | 0.7200; same 144/200 EX as always-revise with 23/200 second generations (11.5%), 9 fixes, 0 harms |
| Volume `grpo-fr/grpo-fr-pilot/` | Pilot adapter + provenance + D8 series | not evaluated further (aborted) |

Stage-2 seed-0 anchor on the same 200 items: **0.7050**. The n=200 margins (FR +3 items vs the
anchor, +1 vs best-of-2) are far inside noise and sit on eroding training dynamics — the
abort rule held. The error-only replay is an inference-time cost note, not new training evidence.
The full verdict narrative, the audit, and the reopening notes live in the archived change's
`tasks.md`. All Stage-3b numbers re-derive on CPU via `make eval` (the `Stage 3b` block in
`results/baseline/reproduce.py`).
