"""Stage verified dataset archives before publishing a complete local installation."""

from __future__ import annotations

import errno
import stat
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

from llm_rlvr.data.artifacts import ArchivePin
from llm_rlvr.data.paths import verify_sha256


def extract_zip(archive: str | Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"unsafe archive member {member.filename!r}")
        bundle.extractall(destination)


def install_archive(
    cache: Path,
    pin: ArchivePin,
    prepare: Callable[[Path], None],
    validate: Callable[[Path], object],
    *,
    tree_name: str | None = None,
) -> Path:
    """Return a validated installation with a receipt for its verified source archive."""
    installed = cache / (tree_name or ".installed")
    receipt = installed / ".archive.sha256"
    if installed.exists():
        if receipt.exists() and (
            not receipt.is_file() or receipt.read_text(encoding="utf-8") != pin.sha256
        ):
            raise ValueError(f"invalid dataset installation receipt at {installed}")
        validate(installed)
        return installed

    from huggingface_hub import hf_hub_download

    archive = hf_hub_download(
        repo_id=pin.repo, filename=pin.filename, revision=pin.revision, repo_type="dataset"
    )
    verify_sha256(archive, pin.sha256)
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".install-", dir=cache) as temporary:
        staged = Path(temporary) / "tree"
        staged.mkdir()
        extract_zip(archive, staged)
        prepare(staged)
        selected = staged / tree_name if tree_name else staged
        validate(selected)
        (selected / ".archive.sha256").write_text(pin.sha256, encoding="utf-8")
        try:
            selected.rename(installed)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            if not receipt.is_file() or receipt.read_text(encoding="utf-8") != pin.sha256:
                raise ValueError(f"conflicting dataset installation at {installed}") from None
            validate(installed)
    return installed
