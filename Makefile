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

data:  ## Fetch + verify + extract the pinned Spider databases (HF mirror; see RUNBOOK)
	uv run python -c "from llm_rlvr_sql.data import ensure_spider_databases as e; print('Spider DBs at', e().database_root)"

eval:  ## Re-derive committed numbers (minutes; not in make check — run when results/ or scoring change)
	uv run python results/baseline/reproduce.py

demo:  ## Run a tiny no-GPU verifier demo against a temporary SQLite database
	uv run python examples/no_gpu_demo.py
