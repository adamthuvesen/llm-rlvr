"""Modal entrypoint helper tests."""

from __future__ import annotations

import pytest

from llm_rlvr.modal_helpers import _validate_limit


def test_validate_limit_rejects_zero_and_negative() -> None:
    for limit in (0, -1):
        with pytest.raises(ValueError, match="limit"):
            _validate_limit(limit)


def test_validate_limit_accepts_none_and_positive() -> None:
    _validate_limit(None)
    _validate_limit(1)
