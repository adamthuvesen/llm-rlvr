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


@pytest.mark.parametrize("path", ["/outside", "../outside", "safe/../../outside", ".", ""])
def test_volume_path_rejects_unconfined_paths(tmp_path: Path, path: str) -> None:
    from llm_rlvr.modal_helpers import volume_path

    with pytest.raises(ValueError, match="relative Volume path"):
        volume_path(tmp_path, path)


def test_volume_path_rejects_symlink_escape(tmp_path: Path) -> None:
    from llm_rlvr.modal_helpers import volume_path

    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        volume_path(tmp_path, "escape/file")


def test_bird_fewshot_rejected_before_data_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llm_rlvr.modal_helpers.ensure_bird_dev", lambda *_: pytest.fail("data fetched")
    )
    with pytest.raises(ValueError, match="unsupported"):
        load_eval_context(dataset="bird", split="dev", data_root=Path("unused"), few_shot_k=1)


def test_eval_context_does_not_claim_canonical_identity_for_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(
        "llm_rlvr.modal_helpers.ensure_spider_databases",
        lambda *_: SimpleNamespace(
            tables_json=tmp_path / "tables.json",
            database_root=tmp_path,
            archive_sha256=None,
            source="override",
        ),
    )
    monkeypatch.setattr("llm_rlvr.modal_helpers.load_schemas", lambda *_: {})
    monkeypatch.setattr("llm_rlvr.modal_helpers.load_examples", lambda *_: ())
    context = load_eval_context(dataset="spider", split="dev", data_root=tmp_path)
    assert context.db_archive_sha256 is None
    assert context.data_source == "override"


def test_generation_names_include_complete_request_identity(tmp_path: Path) -> None:
    from llm_rlvr.modal_helpers import GenerationRequest, managed_run_path

    first = GenerationRequest(base_model="org/model", max_new_tokens=512)
    other_tokens = first.model_copy(update={"max_new_tokens": 1024})
    other_revision = first.model_copy(update={"model_revision": "different"})
    paths = {
        managed_run_path(tmp_path, "generation", request)
        for request in (first, other_tokens, other_revision)
    }
    assert len(paths) == 3
