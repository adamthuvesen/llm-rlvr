# Reproducibility manifest

Every external artifact this project depends on is pinned. Code pins live in
[`src/llm_rlvr/data/artifacts.py`](../../src/llm_rlvr/data/artifacts.py); this file is the
human-readable mirror. The public headline is checked from the tracked
[`results/public/`](../../results/public/) bundle with `make eval` (CPU only, no model). Full raw
re-scoring is external/local: restore the `CompletionSet` and `ScoreReport` bundle under
gitignored `results/`, then run `make data && make eval-full`.

## Datasets

| Artifact | Source | Pin | Integrity |
| --- | --- | --- | --- |
| Spider Q/SQL pairs | HF `xlangai/spider` (parquet) | revision `0c350918f3f29ec754f1181c65cdce76cd6c133c` | HF commit (immutable) |
| Spider SQLite databases | HF `HAL-9001/spider-databases` → `spider_data.zip` | commit `4a01bbac6520cd35b216db9e1724e5e1ada60aa4` | SHA256 `00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b` |
| Spider DB provenance origin | Yale (canonical) | Google Drive id `1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J` | re-hosted under CC-BY-SA-4.0 |
| BIRD-dev bundle (examples + schemas + databases) | HF `HAL-9001/bird-dev` → `dev.zip` | commit `837584317bf7e3d86a3f6d257bd5553b7e03e171` | SHA256 `cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630` |
| BIRD-dev provenance origin | bird-bench.github.io (2024-06-27 release) | `dev.zip` | re-hosted under **CC-BY-NC-4.0** (non-commercial) |
`load_dataset("xlangai/spider")` ships only the parquet; the SQLite databases are the separate,
checksummed artifact above. BIRD's `dev.zip` is one self-hosted bundle.

## Code / models

| Artifact | Pin | License |
| --- | --- | --- |
| Base model | `Qwen/Qwen2.5-Coder-3B-Instruct` at HF `main`; the SFT run pins commit `488639f1ff808d1d3d0ba301aef8c11461451ec5` | Apache 2.0 |
| Cross-family base | `unsloth/Llama-3.2-3B-Instruct` (ungated mirror, byte-identical weights to gated `meta-llama/Llama-3.2-3B-Instruct`; repo id recorded in every artifact's provenance) | Llama 3.2 Community |
| Stage-1 LoRA adapter `sft-r32` | Modal Volume `text2sql-runs` → `sft/sft-r32/` (provenance.json: base commit above, rank 32 all-linear, trace-dataset SHA256, trl 1.5.1 / peft 0.19.1) | - |
| Stage-1 trace dataset | 2,971 traces; manifest generated at `results/sft/traces.manifest.json`; JSONL SHA256 `b02435e49f183425299f3167503883d0b006faf884f7705fc02abc3ad34613e6` (bulk on the Volume at `sft/traces.jsonl`) | self-generated |
| Stage-2 GRPO adapters (main result) | Volume → `grpo/grpo-s0`, `grpo/grpo-s1`, `grpo/grpo-s2`; LR 5e-5, shaped reward, seeds 0/1/2, warm-start `sft/sft-r32`; full config in each `provenance.json` | - |
| Stage-2 reward ablation adapter | Volume → `grpo/grpo-nb-s0`; identical config, `near_binary` arm (1.0/0.1/0) | - |
| Cross-family control adapters | Volume → `sft/sft-llama-r32` (trace manifest `results/sft/traces-llama.manifest.json`, 2,501/7,000 kept) and `grpo/grpo-llama-s0` (same recipe @ LR 5e-5 shaped, warm-start `sft/sft-llama-r32`) | - |
| Reward config (shaped, the main arm) | `tasks/sql.py` `RewardConfig` (shaping fields inherited from `harness/reward.py`): weights format 0.2 / executable 0.2 / result_match 1.0 / length 0.1; `preamble_credit=0.3` (full format credit only when `<think>` leads); `guard_trivial_empty=True` | - |
| Vendored verifier | `taoyds/test-suite-sql-eval`; execution-match path only (`exec_eval.py` + `parse.py`) | Apache 2.0 |

The verifier is parity-tested against upstream (`tests/test_verifier_parity.py`); the only
modifications to vendored code are documented in
[`src/llm_rlvr/third_party/test_suite_sql_eval/ATTRIBUTION.md`](../../src/llm_rlvr/third_party/test_suite_sql_eval/ATTRIBUTION.md).

## Tracked public evidence

The committed public bundle is compact:

| File | Role |
| --- | --- |
| `results/public/claims.json` | Maps public README claims to committed evidence and external raw-artifact sources |
| `results/public/headline_summary.json` | Counts for the README headline table and seed-level CI check |
| `results/public/provenance.json` | States the raw-artifact policy and public/full reproduction modes |
| `results/public/reproduce.py` | Validates the claim map, JSON, README headline rows, and rendered progression tables |

`make eval` runs this public check. It does not need raw completions, databases, Modal, GPU, or
network access.

Public mode is a clean-clone check: it proves the committed headline table is derived from committed
summary/provenance, and it names the external raw inputs behind each claim. Full mode starts after
restoring the external `CompletionSet` and `ScoreReport` bundle under gitignored `results/`. Then
`make eval-full` re-scores those raw artifacts locally without loading a model.

## Generated evaluation artifacts

Except for `results/public/`, paths under `results/` below are external/local artifacts, not
committed files. The base-model baselines (Coder-3B-Instruct, greedy, **full dev**), each a
generated `CompletionSet` + `ScoreReport`, re-score in full-artifact mode via `make eval-full` after
the raw bundle is restored under gitignored `results/`:

| File | Setting | EX |
| --- | --- | --- |
| `spider-dev-zs.json` | Spider-dev, `<think>` zero-shot | 0.6248 (646/1034) |
| `spider-dev-fs3.json` | Spider-dev, `<think>` few-shot k=3 | 0.6770 (700/1034) |
| `spider-dev-zs-direct.json` | Spider-dev, direct zero-shot | 0.6692 (692/1034) |
| `bird-dev-zs.json` | BIRD-dev, `<think>` zero-shot | 0.3129 (480/1534) |
| `bird-dev-zs-direct.json` | BIRD-dev, direct zero-shot | 0.3514 (539/1534) |

Each `CompletionSet` carries its own provenance (model, greedy decoding, dataset revision,
DB-archive SHA256, **`prompt_style`**, few-shot recipe, adapter, git commit), so every number is
traceable to exactly the inputs that produced it. Reporting both prompt styles (`<think>` matched
and direct) keeps the RL gain from being measured against an artificially low base anchor. The
50-example `*-n50` slices remain as the local wiring proof. `*.score.json` files are the generated
`ScoreReport`s.

## Stage 1: SFT cold-start (`results/sft/`, adapter `sft-r32`)

Same pipeline with `--adapter sft/sft-r32`; `make eval-full` re-scores these rows when local
artifacts are present. The success bar was well-formed rate near-100% **and** no EX regression
below the base `<think>` anchor. Both were met.

| File | Setting | EX | Well-formed |
| --- | --- | --- | --- |
| `spider-dev-zs-sft-r32.json` | Spider-dev, `<think>` zero-shot + sft-r32 | 0.6779 (701/1034) | 0.9923 |
| `spider-dev-zs-direct-sft-r32.json` | Spider-dev, direct zero-shot + sft-r32 | 0.7118 (736/1034) | - |
| `bird-dev-zs-sft-r32.json` | BIRD-dev, `<think>` zero-shot + sft-r32 | 0.3442 (528/1534) | 0.9602 |

SFT-vs-base: Spider-dev `<think>` 0.6248 → **0.6779** (+5.3pt, format 3% → 99%); BIRD-dev
`<think>` 0.3129 → **0.3442** (+3.1pt, format 4% → 96%); the adapter also lifts direct mode
(0.6692 → 0.7118). The direct rows score 0 well-formed by construction because that prompt asks for
bare SQL, no `<think>` block.

## Stage 2: single-turn GRPO (`results/grpo/` + `results/llama/`)

Same eval pipeline with the GRPO adapters; `make eval-full` re-scores every row and renders the full
progression tables when the raw bundle is present. The tracked public bundle renders the headline
Qwen progression from `results/public/headline_summary.json`. All EX is single-DB execution-match
with the documented ~6.5% false-positive caveat.

**Decoding disclosure:** all evals are greedy (temperature 0.0; recorded in each provenance), but
the RL/Llama evals used `max_new_tokens=1024` vs the base/SFT evals' 512. Quantified impact:
0/1034 Spider and 1-2/1534 BIRD completions per RL seed exceed the ~512-token envelope, bounding
the effect at ≤0.13% EX on BIRD. That is two orders of magnitude below the +6.1pt gain it could at
most inflate. This is disclosed here rather than re-run.

**Main result (Qwen, 3 seeds @ LR 5e-5 shaped)**: success check met on both benchmarks (the
RL-vs-SFT gain CI excludes zero).

| File(s) | Setting | EX | Well-formed |
| --- | --- | --- | --- |
| `grpo/spider-dev-zs-grpo-s{0,1,2}.json` | Spider-dev, `<think>` + grpo-s{0,1,2} | 0.7795 / 0.7621 / 0.7969; mean **0.779**, Δ vs SFT **+10.2pt** CI [+5.8, +14.5] | 1.00 |
| `grpo/bird-dev-zs-grpo-s{0,1,2}.json` | BIRD-dev, `<think>` + grpo-s{0,1,2} | 0.4081 / 0.4048 / 0.4042; mean **0.406**, Δ vs SFT **+6.1pt** CI [+5.6, +6.7] | 1.00 |

**Reward-shaping ablation (1 seed, point estimates)**: both arms beat SFT, so the gain survives the
near-binary control. Shaped wins on the harder benchmark and keeps a higher well-formed rate.
Training dynamics: the near-binary arm starves its own signal as it improves
(`frac_reward_zero_std` 0.59→0.78 vs ~0.04 shaped). All-identical reward groups teach GRPO nothing.

| File | Setting | EX | Well-formed |
| --- | --- | --- | --- |
| `grpo/spider-dev-zs-grpo-nb-s0.json` | Spider-dev, near-binary | 0.7737 (vs shaped-s0 0.7795; tie, inside the seed band) | 0.9874 |
| `grpo/bird-dev-zs-grpo-nb-s0.json` | BIRD-dev, near-binary | 0.3768 (vs shaped-s0 0.4081; -3.1pt, outside the seed band) | 0.9531 |

**Cross-family control (Llama-3.2-3B, full matched SFT→GRPO, 1 seed, point estimates)**: the same
recipe works on another 3B model, with a larger RL gain than Qwen's (weaker base, more headroom).

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
both model families, using the same training questions. The gain comes from the learning signal
rather than new data.

## Analysis artifacts

Self-correction artifacts under `results/grpo-mt/` and `results/grpo-fr/` are local generated
analysis inputs. The public claim is the single-turn GRPO result above.
