"""Publish complete single-file artifacts without exposing partial writes."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel


def ensure_new_output(path: str | Path) -> Path:
    out = Path(path)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"output already exists: {out}")
    return out


def atomic_write_text(path: str | Path, text: str, *, overwrite: bool = False) -> Path:
    out = Path(path)
    if not overwrite:
        ensure_new_output(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=out.parent,
            prefix=f".{out.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            temporary.replace(out)
        else:
            os.link(temporary, out)
        return out
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(model: BaseModel, path: str | Path, *, overwrite: bool = False) -> Path:
    return atomic_write_text(path, model.model_dump_json(indent=2), overwrite=overwrite)


def sha256_of(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Compute a streaming SHA256 for an artifact file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()
