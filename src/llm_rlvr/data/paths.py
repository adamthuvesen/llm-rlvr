"""Dataset-neutral path and checksum helpers."""

from __future__ import annotations

from pathlib import Path

from llm_rlvr.artifacts.io import sha256_of as sha256_of


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
    if not db_id.strip() or db_id in {".", ".."} or any(c in db_id for c in "/\\\0"):
        raise ValueError(f"db_id must be a single database name, got {db_id!r}")
    path = Path(db_root) / db_id / f"{db_id}.sqlite"
    if not path.resolve().is_relative_to(Path(db_root).resolve()):
        raise ValueError(f"database for db_id={db_id!r} resolves outside database root {db_root}")
    if not path.is_file():
        raise FileNotFoundError(
            f"database for db_id={db_id!r} not found at {path}. SQLite databases are separate "
            "checksummed artifacts — see docs/internal/runbook.md."
        )
    return path
