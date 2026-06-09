.DEFAULT_GOAL := check
.PHONY: install fmt lint test test-integration check eval

install:  ## Sync the CPU dev environment (data, env, verifier, eval — no GPU stack)
	uv sync --extra dev

fmt:  ## Auto-format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Lint and format-check (no changes)
	uv run ruff check .
	uv run ruff format --check .

test:  ## Run unit tests (network/integration deselected)
	uv run pytest

test-integration:  ## Run the network/HuggingFace/model integration tests
	uv run pytest -m integration

check: lint test  ## The Stage-0 gate: lint + unit tests

eval:  ## Reproduce the baseline number from a committed completion set (see RUNBOOK)
	uv run text2sql-rlvr score --completions runs/baseline --out results/baseline.json
