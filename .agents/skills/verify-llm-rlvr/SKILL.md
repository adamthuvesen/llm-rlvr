---
name: verify-llm-rlvr
description: "Drives the real llm-rlvr CLI and Modal launcher on CPU with tiny SQLite fixtures. Use to run, launch, drive, or verify llm-rlvr, or to prove a change to it works."
---

# verify-llm-rlvr

One command proves a change works in the real entry points: the `llm-rlvr` CLI scores
completions, filters SFT traces, and renders the public evidence, and `modal_app.py` resolves
the paid launch plan and refuses it. Nothing reaches Modal, a GPU, Weights & Biases, or the
Hugging Face Hub.

## Run

```bash
make verify                       # drives for files changed against main; the smoke drive if none match
make verify ARGS=all              # every drive, about 2 s
make verify ARGS=list             # each drive, what it proves, the paths it covers
make verify ARGS=control          # one drive with a wrong expectation; exits 0 only on FAIL
make verify ARGS="launch-plan"    # named drives
```

Run `make install` once first.

## Reading results

- Each drive prints `PASS|FAIL <drive> <seconds>s <what it proved or why it failed>`.
- A FAIL names the expected value and the observed value.
- Evidence is in `.verify/runs/<utc>-<sha7>[-dirty]/`: per drive, the argv, redacted env,
  exit code, stdout, and stderr, plus the ScoreReport, trace files, `launch-plan.json`, and
  `summary.json`.
- The helper does not retry, so it never reports `FLAKY`. Run a failed drive again by name.

## Limits

- No drive trains or generates. The locked CPU environment has no torch, TRL, PEFT, or vLLM,
  and the real runs need the Qwen weights. Training proof is a paid Modal smoke, for example
  `modal run modal_app.py::grpo --run-id grpo-small --limit 32 --dev-size 8 --no-spawn --approved`,
  and only after Adam approves it and its rough cost.
- No drive scores real Spider or BIRD databases (`make data` downloads them). The score and
  trace drives use the `make demo` SQLite database. `make eval-full` needs restored artifacts.

## Gotchas

- The entrypoint refusal echoes `sys.argv`, so the launch probe sets it to the `modal run`
  command it emulates.
- Request validation runs before the approval gate. The probe checks that a bad GRPO request
  (`--dev-size` not below `--limit`) fails on validation, before any refusal.
- The child environment points `MODAL_SERVER_URL` and `HF_ENDPOINT` at a dead loopback port,
  and a tripwire on `PATH` fails the run if anything shells out to `modal`, `wandb`, or `hf`.
