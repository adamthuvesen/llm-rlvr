.DEFAULT_GOAL := check
.PHONY: help install lock fmt lint typecheck imports test test-integration runtime-check check eval eval-full demo package-check data check-verifier check-evidence check-training

RUN := uv run --locked
VERIFIER_TESTS := tests/test_sandbox.py tests/test_verifier.py tests/test_verifier_parity.py tests/test_parsing.py tests/test_harness.py
EVIDENCE_TESTS := tests/test_score.py tests/test_aggregate.py tests/test_provenance.py tests/test_completions.py tests/test_report.py tests/test_public_evidence.py tests/test_full_reproduction.py
TRAINING_TESTS := tests/test_sft.py tests/test_grpo.py tests/test_grpo_config.py tests/test_traces.py tests/test_training_runtime.py tests/test_modal_preflight.py tests/test_run_lifecycle.py tests/test_checkpoints.py tests/test_import_architecture.py tests/test_models.py

help:  ## List supported commands
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "%-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install the locked CPU development environment
	uv sync --locked --extra dev

lock:  ## Check pyproject.toml agrees with uv.lock
	uv lock --check

fmt:  ## Apply Ruff formatting and safe fixes
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

lint:  ## Check Ruff lint and formatting
	$(RUN) ruff check .
	$(RUN) ruff format --check .

typecheck:  ## Check package, Modal registration, demo, and maintained scripts
	$(RUN) mypy src/llm_rlvr scripts modal_app.py examples

imports:  ## Check architecture import contracts
	$(RUN) lint-imports

test:  ## Run CPU unit tests in parallel without network or training dependencies
	$(RUN) pytest -n auto

check-verifier:  ## Check SQLite safety, SQL parsing, comparison, and rewards
	$(RUN) pytest $(VERIFIER_TESTS)

check-evidence:  ## Check completion identity, scores, aggregation, and reproduction
	$(RUN) pytest $(EVIDENCE_TESTS)

check-training:  ## Check CPU training builders, preflight, traces, and run lifecycle
	$(RUN) pytest $(TRAINING_TESTS)

test-integration:  ## Run dataset and network checks explicitly
	$(RUN) pytest -m integration

runtime-check:  ## Validate installed training APIs, failing when dependencies are absent
	$(RUN) pytest -m runtime

check: lock lint typecheck imports test package-check  ## Fast gate; package-check also runs public evidence and the demo from the wheel

eval:  ## Validate and render tracked public evidence
	$(RUN) llm-rlvr reproduce --mode public

eval-full:  ## Re-score the complete restored external artifact bundle
	$(RUN) llm-rlvr reproduce --mode full

package-check:  ## Install and verify a wheel outside the source checkout
	$(RUN) python scripts/verify_package.py

data:  ## Fetch and verify the pinned Spider databases
	$(RUN) python -c "from llm_rlvr.data import ensure_spider_databases as e; print('Spider DBs at', e().database_root)"

demo:  ## Score a tiny temporary SQLite database without a model
	$(RUN) python examples/no_gpu_demo.py
