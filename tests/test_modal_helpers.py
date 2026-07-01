"""Modal entrypoint helper tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_rlvr.modal_helpers import _validate_limit, load_eval_context


def test_validate_limit_rejects_zero_and_negative() -> None:
    for limit in (0, -1):
        with pytest.raises(ValueError, match="limit"):
            _validate_limit(limit)


def test_validate_limit_accepts_none_and_positive() -> None:
    _validate_limit(None)
    _validate_limit(1)


def test_load_eval_context_rejects_non_dev_bird_split() -> None:
    # Raises before any data fetch, so no network/data setup is needed.
    with pytest.raises(ValueError, match="bird has only a 'dev' split"):
        load_eval_context(dataset="bird", split="train", data_root=Path("unused"))


def test_load_eval_context_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="dataset"):
        load_eval_context(dataset="wikisql", split="dev", data_root=Path("unused"))
