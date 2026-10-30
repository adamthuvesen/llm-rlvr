"""baseline-evaluation: provenance round-trips and captures library state."""

from __future__ import annotations

from llm_rlvr.artifacts.provenance import (
    GenerationProvenance,
    build_generation_provenance,
    library_versions,
)
from llm_rlvr.eval.score import ScoreReport
from llm_rlvr.tasks.sql.verifier import VerifierConfig


def test_provenance_round_trips(provenance: GenerationProvenance) -> None:
    restored = GenerationProvenance.model_validate_json(provenance.model_dump_json())
    assert restored == provenance


def test_adapter_identity_recorded_when_present() -> None:
    from llm_rlvr.artifacts.provenance import DecodingConfig

    prov = build_generation_provenance(
        model="m",
        backend="vllm",
        dataset="spider",
        dataset_revision="rev",
        split="dev",
        n_examples=1,
        decoding=DecodingConfig(),
        adapter="/runs/sft/sft-r32",
    )
    assert prov.adapter == "/runs/sft/sft-r32"


def test_no_adapter_provenance_defaults_to_none(provenance: GenerationProvenance) -> None:
    # The base (no-adapter) path is unchanged, and pre-adapter artifacts still validate.
    assert provenance.adapter is None
    restored = GenerationProvenance.model_validate_json(provenance.model_dump_json())
    assert restored.adapter is None


def test_library_versions_includes_this_package() -> None:
    versions = library_versions()
    assert versions["llm-rlvr"] != "not-installed"


def test_build_provenance_captures_git_and_libs() -> None:
    from llm_rlvr.artifacts.provenance import DecodingConfig

    prov = build_generation_provenance(
        model="m",
        backend="transformers",
        dataset="spider",
        dataset_revision="rev",
        split="dev",
        n_examples=3,
        decoding=DecodingConfig(),
    )
    assert isinstance(prov.git_commit, str) and prov.git_commit
    assert "llm-rlvr" in prov.library_versions


def test_provenance_records_resolved_exemplars() -> None:
    from llm_rlvr.artifacts.provenance import DecodingConfig

    prov = build_generation_provenance(
        model="m",
        backend="transformers",
        dataset="spider",
        dataset_revision="rev",
        split="dev",
        n_examples=1,
        decoding=DecodingConfig(),
        few_shot_k=2,
        few_shot_seed=3,
        few_shot_exemplars=[("db1", "q1"), ("db2", "q2")],
    )
    assert prov.few_shot_exemplars == (("db1", "q1"), ("db2", "q2"))


def test_score_report_round_trips(provenance: GenerationProvenance) -> None:
    report = ScoreReport(
        execution_accuracy=0.5,
        well_formed_rate=0.5,
        n=2,
        n_correct=1,
        n_gold_errors=0,
        reasons={"match": 1, "denotation_mismatch": 1},
        verifier=VerifierConfig(),
        provenance=provenance,
    )
    assert ScoreReport.model_validate_json(report.model_dump_json()) == report


def test_generation_versions_cannot_be_mutated(provenance: GenerationProvenance) -> None:
    import pytest

    with pytest.raises(TypeError):
        provenance.library_versions["torch"] = "changed"  # type: ignore[index]


def test_nonfinite_and_invalid_decoding_rejected() -> None:
    import pytest

    from llm_rlvr.artifacts.provenance import DecodingConfig

    for updates in (
        {"temperature": float("nan")},
        {"temperature": -1.0},
        {"top_p": float("inf")},
        {"top_p": 0},
        {"max_new_tokens": 0},
    ):
        with pytest.raises(ValueError):
            DecodingConfig.model_validate(updates)


def test_exemplar_count_rejected_but_legacy_recipe_reads(
    provenance: GenerationProvenance,
) -> None:
    import pytest

    data = provenance.model_dump()
    data["few_shot_k"] = 2
    assert GenerationProvenance.model_validate(data).few_shot_k == 2
    data["few_shot_exemplars"] = [["db", "question"]]
    with pytest.raises(ValueError, match="few_shot_exemplars"):
        GenerationProvenance.model_validate(data)


def test_git_identity_belongs_to_source_checkout_not_caller(
    monkeypatch,
    tmp_path,
) -> None:
    from llm_rlvr.artifacts import provenance as module

    expected = module.git_commit()
    assert expected != "unknown"
    monkeypatch.chdir(tmp_path)
    assert module.git_commit() == expected
    monkeypatch.setattr(
        module, "__file__", str(tmp_path / "site-packages/llm_rlvr/eval/provenance.py")
    )
    assert module.git_commit() == "unknown"
    assert module.git_is_dirty() is False
