## 1. Dependencies + module scaffold

- [x] 1.1 Add a `train` optional-dependency extra (`trl>=1.3` — prefer 1.5.1 — `peft`, `accelerate`) keyed off the existing `gpu` extra; resolve `uv.lock`; verify TRL ↔ transformers 5.x ↔ vLLM compatibility against the current pins (vllm 0.22.1 / torch 2.11.0 / transformers 5.10.2) and record the working triple.
- [x] 1.2 Scaffold `src/text2sql_rlvr/train/` (package `__init__`, frozen Pydantic record types for an SFT example + dataset/checkpoint provenance).

## 2. Cold-start trace dataset (coldstart-trace-dataset)

- [x] 2.1 Candidate generation: reuse the prompt builder + vLLM backend to sample `N` (default 8) completions per Spider-train question (`SamplingParams(n=N, temperature≈0.8, top_p 0.95, seed)`); raise loudly if pointed at a dev/test pool (train-only guard).
- [x] 2.2 Execution-filter + selection: score each candidate against the Spider-train DBs with the Stage-0 verifier; keep only candidates that are execution-correct **and** parse to one `<think>` + one ```sql```; select one per question (default: shortest), dedup exact.
- [x] 2.3 Serialize the SFT dataset (chat messages + selected completion, JSONL) plus a provenance manifest (base model + revision, sampling config, dataset revision + DB checksum, kept/total counts).
- [x] 2.4 Tests: filter drops execution-incorrect and malformed candidates; one-per-question selection picks the shortest and dedups; the train-only guard raises on a dev pool; manifest carries the required fields (unit, fixtures — no GPU).

## 3. LoRA SFT trainer (sft-training)

- [x] 3.1 `train/sft.py`: build the TRL `SFTConfig` + PEFT `LoraConfig` (`target_modules='all-linear'`, rank 32 / alpha 64 / dropout 0.05, bf16, LR ≈1e-4, cosine, warmup 0.03, ≤3 epochs, max_seq_len 2048) and load the trace dataset.
- [x] 3.2 Completion-only loss: mask the prompt (schema + question + instruction) via the chat template's assistant-only masking, so loss is computed only over the `<think>`+SQL completion.
- [x] 3.3 Adapter-checkpoint provenance: record base model + revision, LoRA config, trace-dataset hash/identity, library versions (TRL/peft/transformers), git commit + dirty.
- [x] 3.4 Tests: the config builder resolves `all-linear` target modules and the completion-only collator masks the prompt tokens; provenance fields present (unit, no GPU).

## 4. Modal SFT training function (modal_app)

- [x] 4.1 `modal_app.py`: add an `sft_train` `@app.function(gpu="A100-80GB")` that reads the trace dataset from the Volume, runs the LoRA SFT, and writes the adapter to the Volume under `sft/<run-id>/`; `@app.local_entrypoint()` passes git commit/dirty + config and supports `--detach --spawn`.
- [ ] 4.2 Day-1 SFT smoke: a few-step SFT on a tiny trace slice → adapter written to the Volume → loadable over the base (integration / GPU), before any full run.

## 5. Adapter evaluation (baseline-evaluation, modified)

- [ ] 5.1 `eval/generate.py`: load a LoRA adapter over the base — vLLM via `enable_lora` + `LoRARequest`, transformers via `PeftModel.from_pretrained` — threading an `adapter_path` through `generate()`, the eval CLI, and the Modal generation function; the base path (no adapter) stays byte-identical.
- [ ] 5.2 `eval/provenance.py`: record the adapter identity (+ LoRA config) in `GenerationProvenance`, back-compatible default when no adapter is used.
- [ ] 5.3 Well-formed-format rate: compute the fraction of completions parsing to exactly one `<think>` block + one ```sql``` fence in the score path; surface it on the `ScoreReport`.
- [ ] 5.4 Tests: adapter identity recorded in provenance; format-rate computed correctly on fixtures; base (no-adapter) provenance + scoring unchanged (unit).

## 6. Run the cold-start

- [ ] 6.1 Generate candidates on Spider-train via Modal (GPU), pull, filter, and build the trace dataset; commit its manifest + kept/total stats under `results/sft/` (raw traces live on the Volume; commit the manifest, not the bulk).
- [ ] 6.2 Pin the `Qwen/Qwen2.5-Coder-3B-Instruct` commit for the SFT run; run the LoRA SFT on Modal; record the adapter checkpoint id + training provenance.
- [ ] 6.3 Evaluate the SFT checkpoint on full Spider-dev + BIRD-dev (both `<think>` and direct anchors), pull + score; commit the `CompletionSet`s + `ScoreReport`s under `results/sft/`.
- [ ] 6.4 Extend `make eval` to re-derive the SFT-vs-base table (EX + well-formed-format rate) on CPU from the committed artifacts; confirm the success bar — format rate near-100% and no EX regression below the base `<think>` anchor.

## 7. Docs + gate

- [ ] 7.1 `RUNBOOK.md`: the cold-start procedure end-to-end (trace-gen → filter → SFT → adapter eval) with the Modal commands and the success bar.
- [ ] 7.2 `ARTIFACTS.md`: the adapter checkpoint + trace-dataset manifest as pinned artifacts, the model commit, and the committed SFT-vs-base table.
- [ ] 7.3 `make check` green: new unit tests pass; Modal / GPU / TRL tests are integration (deselected by default).
