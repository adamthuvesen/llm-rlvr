"""Dataset split invariants shared by training-data builders."""

from __future__ import annotations

from collections.abc import Sequence

from llm_rlvr.data.models import Example


def assert_train_split(examples: Sequence[Example], *, context: str) -> None:
    """Raise if any example is outside the train split."""
    leaked = sorted({example.split for example in examples if example.split != "train"})
    if leaked:
        raise ValueError(f"{context} must use train-split examples only; got splits {leaked}")
