"""Dataset-neutral path and checksum helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_of(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA256 of a file; dataset archives can be large."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> None:
    """Raise loudly if a file's checksum does not match the pinned trust anchor."""
    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(
            f"checksum mismatch for {path}: expected {expected}, got {actual}. "
            "Refusing to use an unverified dataset archive."
        )


def resolve_database(db_id: str, db_root: str | Path) -> Path:
    """Return the SQLite path for a database id under a Spider/BIRD-style database root."""
    path = Path(db_root) / db_id / f"{db_id}.sqlite"
    if not path.exists():
        raise FileNotFoundError(
            f"database for db_id={db_id!r} not found at {path}. SQLite databases are separate "
            "checksummed artifacts — see docs/RUNBOOK.md."
        )
    return path
