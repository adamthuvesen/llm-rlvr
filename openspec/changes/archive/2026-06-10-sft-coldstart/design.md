## Context

Stage 0 produced a trustworthy execution-match verifier and an honest base-model baseline
(Qwen2.5-Coder-3B-Instruct: Spider-dev 0.625 `<think>` / 0.669 direct; BIRD-dev 0.313 / 0.351). The
key Stage-0 finding is behavioral, not capability: the base *can* write decent SQL but the `<think>`
prompt makes it ramble and lose accuracy, and it does not reliably emit the `<think>…</think>` +
```sql``` shape. Stage 2 GRPO will reward exactly that shape and warm-start from this stage's
checkpoint, so Stage 1's job is to **instill the format and lock in the in-distribution competence**
the base already has — not to maximize EX (that is RL's job).

This is the first stage that *trains*. It reuses the Stage-0/baseline harness wholesale (verifier,
sandbox, Spider loader, `generate()`/`score()`, provenance, the two-layer Modal image) and adds three
things: a self-distilled trace-dataset builder, a LoRA SFT training function on Modal, and the
ability to evaluate a fine-tuned adapter checkpoint with the same verifier.

## Goals / Non-Goals

**Goals:**
- A LoRA adapter checkpoint over Qwen2.5-Coder-3B-Instruct that reliably emits well-formed
  `<think>…</think>` + ```sql``` and does not regress EX below the base `<think>` anchor.
- The SFT dataset is built the project's own way: self-sampled, **execution-filtered by the Stage-0
  verifier** — the thesis ("the execution signal bootstraps the model") starts here.
- The checkpoint is a pinned, provenance-stamped artifact; the SFT-vs-base comparison (EX + format
  rate, both prompt anchors, Spider-dev + BIRD-dev) is reproducible.
- Minimal, on-plan, infra-reusing — the only genuinely new code is the trace builder and the SFT
  trainer wrapper.

**Non-Goals:**
- GRPO, the reward module, multi-turn, the cross-family / reward-shaping controls (Stage 2+).
- Beating SQL-R1's SFT number (that used 200K SynSQL; we keep it minimal and set expectations).
- SynSQL ingestion and teacher distillation — documented fallbacks only, not built here.
- Full-FT (LoRA is the decided parameterization); multi-seed SFT (1 seed; multi-seed is Stage 2).

## Decisions

### D1 — Self-distilled, execution-filtered traces (STaR), not gold-imitation or teacher distillation
Sample the **base** model on Spider-train with the `<think>` prompt, score every sample with the
Stage-0 verifier, and keep only execution-correct `(prompt, <think>+SQL)` pairs as SFT targets.
- *Why:* the kept traces are correct **by construction**, in the policy's **own distribution** (so
  Stage 2 GRPO starts with minimal KL shock), and built entirely from infrastructure we already have
  — the verifier that scores the project becomes the data filter. On-thesis and near-free.
- *Alternatives:* **gold-SQL format-only** (no reasoning → trains an empty/absent `<think>`, the weak
  init SCoRe warns collapses RL); **teacher distillation** (a 32B generates traces → capability
  injection but a teacher dependency, a foreign reasoning voice that widens the GRPO gap, and an
  off-thesis story). Both are kept as written-down fallbacks if the checkpoint is weak.

### D2 — Sampling + selection policy: small-N, keep the *shortest* correct trace per question
Generate `N` (default 8) samples per Spider-train question at temperature ≈ 0.8, top_p 0.95; among
the execution-correct samples keep the **shortest** one (lowest token count), dedup exact. One trace
per question.
- *Why:* temperature gives trace diversity and recovers some questions greedy misses; "shortest
  correct" selects the cleanest, least-rambling trace — exactly the format hygiene we want to instill,
  and a direct counter to the base's ramble failure mode. One-per-question (~5k from ~7k Spider-train
  at the base's ~70% keep-rate) is ample for a cold-start LoRA and keeps the set balanced.
- *Alternatives:* greedy single-sample (loses the hard tail and diversity); keep-all-correct (up to
  N per question → over-weights easy questions). Selection knobs are config, not hard-coded.

### D3 — LoRA SFT, completion-only loss, conservative budget
TRL `SFTTrainer` + PEFT LoRA: `target_modules='all-linear'`, rank 32 / alpha 64 / dropout 0.05,
bf16, **completion-only loss** (mask the prompt; train only on the `<think>`+SQL assistant turn via
the chat template's `assistant_only_loss`), max_seq_len 2048, LR ≈ 1e-4, cosine, warmup 0.03, ≤3
epochs, early-stop / checkpoint-select on a held-out dev-EX slice.
- *Why:* LoRA is the decided parameterization (matches full-FT in this regime, far less memory, fits
  a 3B on one A100 with headroom). Completion-only loss keeps the model from training on the schema
  and question text. Conservative LR + few epochs + LoRA all bound drift — a cold-start should nudge
  behavior, not overwrite the base. SFT LoRA LR (~1e-4) is higher than the GRPO LoRA LR (~5e-5) by
  the usual SFT-vs-RL gap.
- *Alternatives:* full-FT (rejected, plan); higher LR / more epochs (over-fit risk on a small set);
  packing (off — complicates completion-only masking for marginal throughput on short SQL).

### D4 — Generation reuses the baseline path; filtering stays on CPU
Trace generation is a Modal **GPU** run (the existing vLLM backend, `SamplingParams(n=N)`,
split=train); scoring + filtering + dataset assembly run on **CPU** (local or a Modal CPU function),
exactly mirroring the baseline's "generate on GPU, score on CPU" split. Spider-train SQLite DBs ship
in the same pinned archive as dev, so the verifier needs nothing new. The trace builder does not
disturb the `CompletionSet` contract — it reuses prompt-building + the vLLM backend and emits a
candidate-per-question structure of its own.
- *Why:* GPU time is the cost; keep the verifier's DB execution (the real eval cost) on cheap CPU,
  and reuse the proven generation path rather than forking it.

### D5 — Adapter checkpoint persisted to the Volume; eval loads LoRA over the base
SFT saves the **adapter** (`adapter_model.safetensors` + `adapter_config.json`) to the
`text2sql-runs` Volume under `sft/<run-id>/`, with a training-provenance record (base model + commit,
LoRA config, trace-dataset hash, TRL/peft versions, git commit). Evaluation extends `generate()` to
load a LoRA adapter over the base — vLLM via `enable_lora` + `LoRARequest`, transformers via
`PeftModel.from_pretrained` — and records base+adapter in the completion provenance, so the SFT
checkpoint is scored by the identical verifier and `make eval` re-derives the SFT-vs-base table.
- *Why:* an adapter is small and portable, keeps the base immutable, and is the natural Stage-2 GRPO
  starting point. Sharing the exact eval path guarantees the comparison is apples-to-apples.

### D6 — Success is format + no-regression, reported against both anchors
Definition of done for the checkpoint: **well-formed-format rate** (parses to a single `<think>`
block + one ```sql``` fence) rises to near-100% on dev, **and** EX does not regress below the base
`<think>` anchor (target ≥ the direct anchor). Reported on Spider-dev and BIRD-dev, 1 seed.
- *Why:* the cold-start optimizes behavior and in-distribution competence, not peak EX; tying success
  to format + no-regression (rather than an EX jump) keeps Stage 1 honest and sets correct
  expectations for where the real gain comes from (Stage 2).

## Risks / Trade-offs

- **Coverage skew — hard questions the base never solves contribute no traces** → accept and document;
  the cold-start's job is format + latent competence, and GRPO is designed to push the hard tail.
  Temperature sampling recovers some of it.
- **TRL ↔ transformers 5.x / vLLM version friction** (the pairing is historically fragile; current
  pins are vllm 0.22.1 / torch 2.11.0 / transformers 5.10.2) → pin a `train` triple and run a tiny
  SFT smoke (a handful of steps) before the full run; record the working versions.
- **Filter false-positives — execution-match admitting a degenerate trace** (e.g. empty==empty) →
  reuse the verifier's empty-result handling; optionally require a non-trivial gold result-set for a
  trace to qualify. Note the single-DB false-positive rate as a known limitation.
- **Over-fit / catastrophic forgetting on a small SFT** → low LR, ≤3 epochs, LoRA, dev-EX
  checkpoint-selection; the no-regression success bar catches it directly.
- **Adapter-eval mismatch** (vLLM loading the adapter differently than training) → a day-1 adapter
  smoke (train a few steps, load, generate, eyeball format) before the full eval.

## Migration Plan

Purely additive — no existing artifact changes meaning. The base baselines stay valid as the "before"
anchor. Rollback is trivial (don't use the adapter). If the checkpoint misses the success bar, the
documented fallback ladder is: (1) more/temperature-varied traces, (2) a small SynSQL pull, (3)
teacher-distilled traces — each a follow-up change, not a blocker for shipping base→GRPO.

## Open Questions

- Final `N` / temperature / epochs land after the trace-gen + SFT smoke (defaults above are the
  starting point; tune on the dev-EX curve).
- Whether to keep 1 vs up-to-2 correct traces per question — start at 1, revisit only if the set is
  too small for stable SFT.
