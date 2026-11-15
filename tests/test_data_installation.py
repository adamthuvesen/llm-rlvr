"""Dataset installation publishes complete trees and preserves source identity."""

from __future__ import annotations

import errno
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from llm_rlvr.data.artifacts import ArchivePin
from llm_rlvr.data.bird import ensure_bird_dev
from llm_rlvr.data.installation import extract_zip, install_archive
from llm_rlvr.data.spider import ensure_spider_databases


def _archive(tmp_path: Path, files: dict[str, str | bytes]) -> Path:
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)
    return archive


def _pin(archive: Path) -> ArchivePin:
    return ArchivePin(
        repo="test/data",
        revision="rev",
        filename="data.zip",
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )


def _download(monkeypatch: pytest.MonkeyPatch, archive: Path) -> None:
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **_kwargs: str(archive))


def test_spider_install_preserves_public_path_and_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(
        tmp_path, {"spider_data/database/placeholder": "", "spider_data/tables.json": "[]"}
    )
    pin = _pin(archive)
    monkeypatch.delenv("TEXT2SQL_SPIDER_ROOT", raising=False)
    monkeypatch.setattr("llm_rlvr.data.spider.SPIDER_DB_ARCHIVE", pin)
    _download(monkeypatch, archive)
    cache = tmp_path / "cache"
    installed = ensure_spider_databases(cache)
    assert installed.root == cache / "spider_data"
    assert installed.source == "archive"
    assert installed.archive_sha256 == pin.sha256
    assert (installed.root / ".archive.sha256").read_text() == pin.sha256
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda **_kwargs: pytest.fail("redownload")
    )
    assert ensure_spider_databases(cache) == installed
    monkeypatch.setenv("TEXT2SQL_SPIDER_ROOT", str(installed.root))
    override = ensure_spider_databases(cache)
    assert override.source == "override"
    assert override.archive_sha256 is None


def test_spider_existing_legacy_tree_is_not_claimed_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "spider_data"
    (root / "database").mkdir(parents=True)
    (root / "tables.json").write_text("[]")
    monkeypatch.delenv("TEXT2SQL_SPIDER_ROOT", raising=False)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda **_kwargs: pytest.fail("redownload")
    )
    installed = ensure_spider_databases(tmp_path)
    assert installed.root == root
    assert installed.source == "legacy"
    assert installed.archive_sha256 is None
    assert not (root / ".archive.sha256").exists()


def test_invalid_extraction_does_not_publish_tree_or_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(tmp_path, {"spider_data/tables.json": "[]"})
    monkeypatch.delenv("TEXT2SQL_SPIDER_ROOT", raising=False)
    monkeypatch.setattr("llm_rlvr.data.spider.SPIDER_DB_ARCHIVE", _pin(archive))
    _download(monkeypatch, archive)
    cache = tmp_path / "cache"
    with pytest.raises(FileNotFoundError, match="spider_data layout"):
        ensure_spider_databases(cache)
    assert list(cache.iterdir()) == []


def test_bird_nested_archive_is_validated_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as bundle:
        bundle.writestr("dev_databases/placeholder", "")
    archive = _archive(
        tmp_path,
        {
            "dev/dev.json": "[]",
            "dev/dev_tables.json": "[]",
            "dev/dev_databases.zip": nested.getvalue(),
        },
    )
    pin = _pin(archive)
    monkeypatch.delenv("TEXT2SQL_BIRD_ROOT", raising=False)
    monkeypatch.setattr("llm_rlvr.data.bird.BIRD_ARCHIVE", pin)
    _download(monkeypatch, archive)
    bird = ensure_bird_dev(tmp_path / "cache")
    assert bird.database_root.is_dir()
    assert bird.source == "archive"
    assert bird.archive_sha256 == pin.sha256
    monkeypatch.setenv("TEXT2SQL_BIRD_ROOT", str(bird.root))
    assert ensure_bird_dev().archive_sha256 is None


def test_bird_incomplete_nested_archive_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(tmp_path, {"dev/dev.json": "[]", "dev/dev_tables.json": "[]"})
    monkeypatch.delenv("TEXT2SQL_BIRD_ROOT", raising=False)
    monkeypatch.setattr("llm_rlvr.data.bird.BIRD_ARCHIVE", _pin(archive))
    _download(monkeypatch, archive)
    cache = tmp_path / "cache"
    with pytest.raises(FileNotFoundError, match="BIRD dev layout"):
        ensure_bird_dev(cache)
    assert list(cache.iterdir()) == []


def test_installer_handles_nonempty_concurrent_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(tmp_path, {"data/file": "complete"})
    pin = _pin(archive)
    _download(monkeypatch, archive)
    cache = tmp_path / "cache"
    target = cache / "data"

    def concurrent_rename(staged: Path, destination: Path) -> None:
        assert destination == target
        destination.mkdir()
        (destination / "file").write_text("complete")
        (destination / ".archive.sha256").write_text(pin.sha256)
        raise OSError(errno.ENOTEMPTY, "Directory not empty")

    monkeypatch.setattr(Path, "rename", concurrent_rename)
    installed = install_archive(
        cache, pin, lambda _root: None, lambda root: (root / "file").read_text(), tree_name="data"
    )
    assert installed == target
    assert (installed / "file").read_text() == "complete"


@pytest.mark.parametrize("name", ["../outside", "/absolute"])
def test_archive_paths_cannot_escape_staging(tmp_path: Path, name: str) -> None:
    archive = _archive(tmp_path, {name: "bad"})
    with pytest.raises(ValueError, match="unsafe archive member"):
        extract_zip(archive, tmp_path / "staging")
    assert not (tmp_path / "outside").exists()


def test_installation_validates_schema_database_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = {
        "db_id": "missing",
        "table_names_original": [],
        "column_names_original": [],
        "column_types": [],
    }
    archive = _archive(
        tmp_path,
        {"spider_data/database/placeholder": "", "spider_data/tables.json": json.dumps([schema])},
    )
    monkeypatch.delenv("TEXT2SQL_SPIDER_ROOT", raising=False)
    monkeypatch.setattr("llm_rlvr.data.spider.SPIDER_DB_ARCHIVE", _pin(archive))
    _download(monkeypatch, archive)
    cache = tmp_path / "cache"
    with pytest.raises(FileNotFoundError, match="not found"):
        ensure_spider_databases(cache)
    assert list(cache.iterdir()) == []
