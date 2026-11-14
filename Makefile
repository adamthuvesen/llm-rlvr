.DEFAULT_GOAL := check
.PHONY: help install lock fix lint typecheck imports test test-integration runtime-check check verify eval eval-full demo package-check data

RUN := uv run --locked --extra dev

help:  ## List supported commands
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "%-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install the locked CPU development environment
	uv sync --locked --extra dev

lock:  ## Check pyproject.toml agrees with uv.lock
	uv lock --check

fix:  ## Apply Ruff formatting and safe fixes
	$(RUN) ruff check --fix . .agents/skills/verify-llm-rlvr/verify
	$(RUN) ruff format . .agents/skills/verify-llm-rlvr/verify

lint:  ## Check Ruff lint and formatting
	$(RUN) ruff check . .agents/skills/verify-llm-rlvr/verify
	$(RUN) ruff format --check . .agents/skills/verify-llm-rlvr/verify

typecheck:  ## Check package, Modal registration, demo, scripts, tests, and the verify helper (basedpyright strict)
	$(RUN) basedpyright
	$(RUN) basedpyright --project .agents/skills/verify-llm-rlvr

imports:  ## Check architecture import contracts
	$(RUN) lint-imports

test:  ## Run CPU unit tests in parallel without network or training dependencies
	$(RUN) pytest -n auto

test-integration:  ## Run dataset and network checks explicitly
	$(RUN) pytest -m integration --force-enable-socket

runtime-check:  ## Validate installed training APIs, failing when dependencies are absent
	$(RUN) pytest -m runtime

check: lock lint typecheck imports test package-check  ## Fast gate; package-check also runs public evidence and the demo from the wheel

verify:  ## Drive the real CLI and Modal launcher on CPU (ARGS=all|list|control|<drive>)
	@$(RUN) python .agents/skills/verify-llm-rlvr/verify $(ARGS)

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
