## 1. BIRD data layer (`bird-dataset`)

- [x] 1.1 Add optional `evidence: str | None = None` to `Example`; confirm Spider examples still round-trip (evidence unset) and existing tests pass.
- [x] 1.2 Pin BIRD-dev artifacts in `data/artifacts.py`: the HF revision for the Q/SQL pairs and a self-hosted, **checksummed** BIRD DB archive (`BIRD_DB_ARCHIVE`); add a fetch+verify+extract helper mirroring `ensure_spider_databases`.
- [x] 1.3 `data/bird.py`: parse BIRD `dev.json` + `dev_tables.json` into the shared `Example` / `DatabaseSchema` models; resolve `<db_id>/<db_id>.sqlite` (reusing `resolve_database`).
- [x] 1.4 Prompt builder: include `evidence` in the user content when present; Spider prompts (no evidence) byte-identical to before.
- [x] 1.5 Tests: BIRD `tables.json` parse + deterministic serialization (unit, fixture); evidence-in-prompt and Spider-unchanged; checksum-mismatch and missing-DB raise loudly; BIRD-dev live load (integration, network-gated).

## 2. vLLM GPU generation path

- [x] 2.1 Exercise + fix the existing `_run_vllm` backend against current vLLM (chat templating, `SamplingParams`, `llm.chat`); pin a compatible `gpu` extra version triple (vLLM / torch) and record it.
- [x] 2.2 Smoke: a tiny `--backend vllm` generate produces a valid `CompletionSet` (integration / GPU).

## 3. Modal runner (`modal-runner`)

- [x] 3.1 `modal_app.py` (run via `uv tool install modal`, not a project dep): two-layer image — layer 1 installs the pinned GPU deps keyed on `pyproject.toml`+`uv.lock`; layer 2 relinks the source editable. A single `@app.function(gpu="A100-80GB")` runs `generate(backend="vllm")` and writes the `CompletionSet` to a Modal Volume.
- [x] 3.2 `@app.local_entrypoint()` computes git commit/dirty locally (image strips `.git`) and passes them into provenance; support `--detach --spawn` for long runs.
- [x] 3.3 Day-1 Modal smoke: `--limit 8` generate end-to-end (build image → run on GPU → `modal volume get` → score locally) before any full pass.

## 4. Proper baseline run (`baseline-evaluation`)

- [x] 4.1 Pin a `Qwen/Qwen2.5-Coder-3B-Instruct` commit for the headline run; record it in provenance + `ARTIFACTS.md`.
- [x] 4.2 Run full Spider-dev (1034) zero-shot + few-shot on the Modal GPU; pull + score; commit the `CompletionSet`s + `ScoreReport`s under `results/baseline/`.
- [x] 4.3 Run BIRD-dev zero-shot + few-shot on the Modal GPU (few-shot exemplars from BIRD-train, R2 train-only); pull + score; commit as the **headline** reference artifacts.
- [x] 4.4 `make eval` re-derives all committed numbers from the `CompletionSet`s on CPU (extend it to cover Spider + BIRD).

## 5. Reproducibility & docs

- [x] 5.1 `RUNBOOK.md`: the Modal run procedure — auth, image build, the day-1 smoke, the full detached run, and pulling results from the Volume.
- [x] 5.2 `ARTIFACTS.md`: BIRD pins (HF revision + DB-archive SHA256 + CC-BY-NC attribution), the model commit, the Modal Volume, and the committed headline Spider/BIRD numbers.
- [x] 5.3 `make check` green: BIRD unit tests pass; Modal / GPU / vLLM tests are integration (deselected by default).

> Scope notes: BIRD **few-shot** is deferred (needs a self-hosted BIRD-train); the base model is recorded at HF `main`, not a pinned commit (greedy is deterministic — a strict-repro follow-up). Both documented in RUNBOOK/ARTIFACTS.
