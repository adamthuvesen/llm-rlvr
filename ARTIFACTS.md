# Reproducibility manifest

Every external artifact this project depends on, pinned. Code pins live in
[`src/text2sql_rlvr/data/artifacts.py`](src/text2sql_rlvr/data/artifacts.py); this file is the
human-auditable mirror. The headline number is reproducible from a committed `CompletionSet` with
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
| Base model | `Qwen/Qwen2.5-Coder-3B-Instruct` at HF `main` (greedy is deterministic; pin a commit for byte-strict repro) | Apache 2.0 |
| Vendored verifier | `taoyds/test-suite-sql-eval` — execution-match path only (`exec_eval.py` + `parse.py`) | Apache 2.0 |

The verifier is parity-tested against upstream (`tests/test_verifier_parity.py`); the only
modifications to vendored code are documented in
`src/text2sql_rlvr/third_party/test_suite_sql_eval/ATTRIBUTION.md`.

## Committed evaluation artifacts

The headline base-model baselines (Coder-3B-Instruct, greedy, **full dev**), each a committed
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
DB-archive SHA256, **`prompt_style`**, few-shot recipe, git commit), so every number is traceable
to exactly the inputs that produced it. The two prompt styles (`<think>` matched vs direct) are
reported deliberately — see RUNBOOK § *Methodology*. The 50-example `*_dev50*` slices remain as the
local wiring proof. `*.score.json` files are the committed `ScoreReport`s.
