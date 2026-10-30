"""Database paths stay within the requested dataset root."""

from pathlib import Path

import pytest

from llm_rlvr.data.paths import resolve_database


@pytest.mark.parametrize(
    "db_id", ["", " ", ".", "..", "../outside", "a/b", "a\\b", "/tmp/db", "a\0b"]
)
def test_database_id_rejects_paths(db_id: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="single database name"):
        resolve_database(db_id, tmp_path)


def test_database_symlink_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "database"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "concert.sqlite").touch()
    (root / "concert").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="outside database root"):
        resolve_database("concert", root)


def test_database_file_symlink_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "database"
    (root / "concert").mkdir(parents=True)
    outside = tmp_path / "outside.sqlite"
    outside.touch()
    (root / "concert" / "concert.sqlite").symlink_to(outside)

    with pytest.raises(ValueError, match="outside database root"):
        resolve_database("concert", root)


def test_database_path_must_be_file(tmp_path: Path) -> None:
    (tmp_path / "concert" / "concert.sqlite").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_database("concert", tmp_path)


def test_database_root_can_be_symlink(tmp_path: Path) -> None:
    root = tmp_path / "database"
    (root / "concert").mkdir(parents=True)
    (root / "concert" / "concert.sqlite").touch()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)

    assert resolve_database("concert", alias) == alias / "concert" / "concert.sqlite"
