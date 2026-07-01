"""baseline-evaluation: provenance round-trips and captures library state."""

from __future__ import annotations

from llm_rlvr.eval.provenance import (
    GenerationProvenance,
    build_generation_provenance,
    library_versions,
)
from llm_rlvr.eval.score import ScoreReport
from llm_rlvr.eval.verifier import VerifierConfig


def test_provenance_round_trips(provenance: GenerationProvenance) -> None:
    restored = GenerationProvenance.model_validate_json(provenance.model_dump_json())
    assert restored == provenance


def test_adapter_identity_recorded_when_present() -> None:
    from llm_rlvr.eval.provenance import DecodingConfig

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
    from llm_rlvr.eval.provenance import DecodingConfig

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
    from llm_rlvr.eval.provenance import DecodingConfig

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
