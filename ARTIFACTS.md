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
| Distilled test-suite DBs (eval-time test-suite accuracy) | Google Drive id `1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w` | **TBD** — R4 fast-follow | — |
| BIRD-dev (reported headline) | TBD | **TBD** — task 2.4 (stretch) | — |

`load_dataset("xlangai/spider")` ships only the parquet; the SQLite databases are the separate,
checksummed artifact above (design D2/R1).

## Code / models

| Artifact | Pin | License |
| --- | --- | --- |
| Base model | `Qwen/Qwen2.5-Coder-3B-Instruct` (pin a model commit before the proper run) | Apache 2.0 |
| Vendored verifier | `taoyds/test-suite-sql-eval` — execution-match path only (`exec_eval.py` + `parse.py`) | Apache 2.0 |

The verifier is parity-tested against upstream (`tests/test_verifier_parity.py`); the only
modifications to vendored code are documented in
`src/text2sql_rlvr/third_party/test_suite_sql_eval/ATTRIBUTION.md`.

## Committed evaluation artifacts

| File | What | Reproduces |
| --- | --- | --- |
| `results/baseline/completions.json` | Coder-3B-Instruct zero-shot, Spider-dev 50-example slice (seed 0) | `make eval` → `execution_accuracy=0.7600 (38/50)` |
| `results/baseline/score.json` | the scored zero-shot `ScoreReport` (with full provenance) | — |
| `results/baseline/completions_fewshot.json` | same slice, few-shot k=3 (seed 0) | score → `0.7400 (37/50)` |
| `results/baseline/score_fewshot.json` | the scored few-shot `ScoreReport` | — |

The committed `CompletionSet` carries its own provenance (model, greedy decoding, dataset
revision, DB-archive SHA256, few-shot recipe), so the number is traceable to exactly the inputs
that produced it. The 50-example slice is the local wiring proof; the full-dev + BIRD figures
(the GPU "proper run", see RUNBOOK) replace it as the headline artifacts.
