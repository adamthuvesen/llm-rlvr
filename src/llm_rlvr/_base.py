"""Shared strict Pydantic base: frozen + reject unknown keys (loud on malformed artifacts)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Frozen(BaseModel):
    """Immutable, extra-forbidding model — the default for typed records and persisted artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid")
