.DEFAULT_GOAL := check
.PHONY: install fmt lint test test-integration check eval demo

install:  ## Sync the CPU dev environment (data, env, verifier, eval — no GPU stack)
	uv sync --extra dev

fmt:  ## Auto-format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Lint and format-check (no changes)
	uv run ruff check .
	uv run ruff format --check .

test:  ## Run unit tests (network/integration deselected)
	uv run python -m pytest

test-integration:  ## Run the network/HuggingFace/model integration tests
	uv run python -m pytest -m integration

check: lint test  ## Project gate: lint + unit tests

data:  ## Fetch + verify + extract the pinned Spider databases (HF mirror; see docs/internal/runbook.md)
	uv run python -c "from llm_rlvr.data import ensure_spider_databases as e; print('Spider DBs at', e().database_root)"

eval:  ## Re-score local result artifacts after pulling them into gitignored results/
	@test -f results/baseline/reproduce.py || (echo "No local results/ reproduction script found. Pull or restore run artifacts first."; exit 1)
	uv run python results/baseline/reproduce.py

demo:  ## Run a tiny no-GPU verifier demo against a temporary SQLite database
	uv run python examples/no_gpu_demo.py
