"""Full re-scoring checks the whole inventory before executing any query."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_rlvr.eval.completions import assemble_completion_set, write_completion_set
from llm_rlvr.eval.provenance import GenerationProvenance
from llm_rlvr.eval.score import score_completion_set
from llm_rlvr.report.full import (
    ExpectedGeneration,
    FullArtifact,
    FullManifest,
    build_manifest,
    main,
    reproduce,
)
from llm_rlvr.report.public import Count, PublicSummary


def _artifacts(root: Path, provenance: GenerationProvenance, scoring_root: Path) -> FullArtifact:
    from llm_rlvr.data.models import Example

    examples = [Example(db_id="concert", question="q", gold_sql="SELECT a FROM t", split="dev")]
    completions = assemble_completion_set(
        examples, ["```sql\nSELECT a FROM t\n```"], provenance.model_copy(update={"n_examples": 1})
    )
    completion = "results/tiny.json"
    historical_score = "results/tiny.score.json"
    write_completion_set(completions, root / completion)
    score = score_completion_set(completions, scoring_root)
    # Exercise old artifacts without scorer identity or per-item outcomes.
    legacy = score.model_dump(exclude={"scoring", "outcomes"})
    from llm_rlvr.eval.score import ScoreReport

    (root / historical_score).write_text(ScoreReport.model_validate(legacy).model_dump_json())
    return FullArtifact(
        completion=completion,
        historical_score=historical_score,
        dataset="spider",
        identity=ExpectedGeneration(model=provenance.model, adapter=None),
        expected=Count(correct=1, total=1, source="fixture"),
    )


def test_real_sqlite_rescore_retains_legacy_scores(
    provenance: GenerationProvenance, scoring_root: Path, tmp_path: Path
) -> None:
    artifact = _artifacts(tmp_path, provenance, scoring_root)
    historical = (tmp_path / artifact.historical_score).read_bytes()
    (result,) = reproduce(
        FullManifest(artifacts=(artifact,)),
        tmp_path,
        spider_db_root=scoring_root,
        bird_db_root=scoring_root,
    )
    assert result.matches
    assert result.rescored.execution_accuracy == 1
    assert result.rescored.scoring is not None
    assert result.changed_outcomes is None
    assert len(result.completion_sha256) == 64
    assert (tmp_path / artifact.historical_score).read_bytes() == historical


@pytest.mark.parametrize("malformed", [False, True])
def test_missing_or_malformed_later_artifact_prevents_all_scoring(
    provenance: GenerationProvenance,
    scoring_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformed: bool,
) -> None:
    artifact = _artifacts(tmp_path, provenance, scoring_root)
    later = artifact.model_copy(
        update={"completion": "results/later.json", "historical_score": "results/later.score.json"}
    )
    if malformed:
        (tmp_path / later.completion).write_text("{}")
        (tmp_path / later.historical_score).write_text("{}")

    def forbidden_score(*args, **kwargs):
        raise AssertionError("scoring started before inventory validation")

    monkeypatch.setattr("llm_rlvr.report.full.score_completion_set", forbidden_score)
    with pytest.raises(ValueError):
        reproduce(
            FullManifest(artifacts=(artifact, later)),
            tmp_path,
            spider_db_root=scoring_root,
            bird_db_root=scoring_root,
        )


def test_historical_hash_and_headline_mismatch_fail_before_scoring(
    provenance: GenerationProvenance, scoring_root: Path, tmp_path: Path
) -> None:
    artifact = _artifacts(tmp_path, provenance, scoring_root)
    for updated, message in (
        (artifact.model_copy(update={"completion_sha256": "0" * 64}), "hash mismatch"),
        (
            artifact.model_copy(update={"expected": Count(correct=0, total=1, source="fixture")}),
            "disagrees with public summary",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            reproduce(
                FullManifest(artifacts=(updated,)),
                tmp_path,
                spider_db_root=scoring_root,
                bird_db_root=scoring_root,
            )


def test_inventory_has_all_headline_and_documented_controls() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = PublicSummary.model_validate_json(
        (root / "results/public/headline_summary.json").read_text()
    )
    manifest = build_manifest(summary)
    expected_count = sum(2 + len(row.grpo_seeds) for row in summary.headline.rows)
    assert sum(item.expected is not None for item in manifest.artifacts) == expected_count
    assert len(manifest.artifacts) == expected_count + 12
    assert any("fs3" in item.completion for item in manifest.artifacts)
    assert sum("llama" in item.completion for item in manifest.artifacts) == 6


def test_missing_inventory_reports_all_files_without_partial_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    source = Path(__file__).resolve().parents[1] / "results/public/headline_summary.json"
    target = tmp_path / "results/public/headline_summary.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    assert main(tmp_path) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "restore the full external artifact bundle" in output.err
    assert "results/baseline/spider-dev-zs.json" in output.err
    assert "results/llama/bird-dev-zs-llama-3.2-3b-instruct-grpo-llama-s0.score.json" in output.err


def test_modern_score_input_identity_is_checked_before_rescoring(
    provenance: GenerationProvenance, scoring_root: Path, tmp_path: Path
) -> None:
    from llm_rlvr.eval.completions import read_completion_set

    artifact = _artifacts(tmp_path, provenance, scoring_root)
    completion = read_completion_set(tmp_path / artifact.completion)
    modern = score_completion_set(completion, scoring_root).model_dump()
    modern["scoring"]["completion_set_sha256"] = "0" * 64
    from llm_rlvr.eval.score import ScoreReport

    (tmp_path / artifact.historical_score).write_text(
        ScoreReport.model_validate(modern).model_dump_json()
    )
    with pytest.raises(ValueError, match="scoring input digest mismatch"):
        reproduce(
            FullManifest(artifacts=(artifact,)),
            tmp_path,
            spider_db_root=scoring_root,
            bird_db_root=scoring_root,
        )


def test_item_changes_are_detected_when_aggregate_counts_are_equal(
    provenance: GenerationProvenance, scoring_root: Path
) -> None:
    from llm_rlvr.data.models import Example
    from llm_rlvr.report.full import ScoreComparison

    example = Example(db_id="concert", question="q", gold_sql="SELECT a FROM t", split="dev")
    old = assemble_completion_set(
        [example, example], ["```sql\nSELECT a FROM t\n```", ""], provenance
    )
    new = assemble_completion_set(
        [example, example], ["", "```sql\nSELECT a FROM t\n```"], provenance
    )
    comparison = ScoreComparison(
        artifact=FullArtifact(
            completion="old.json",
            historical_score="old.score.json",
            dataset="spider",
            identity=ExpectedGeneration(model=provenance.model, adapter=None),
        ),
        completion_sha256="fixture",
        historical_score_sha256="fixture",
        historical=score_completion_set(old, scoring_root),
        rescored=score_completion_set(new, scoring_root),
    )
    assert comparison.historical.n_correct == comparison.rescored.n_correct
    assert comparison.changed_outcomes == (0, 1)
    assert not comparison.matches


def test_database_identity_change_fails_before_sql_even_if_query_results_are_equal(
    provenance: GenerationProvenance,
    scoring_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    from llm_rlvr.eval.completions import read_completion_set

    artifact = _artifacts(tmp_path, provenance, scoring_root)
    completion = read_completion_set(tmp_path / artifact.completion)
    modern = score_completion_set(completion, scoring_root)
    (tmp_path / artifact.historical_score).write_text(modern.model_dump_json())
    with sqlite3.connect(scoring_root / "concert/concert.sqlite") as connection:
        connection.execute("CREATE TABLE unrelated (value INTEGER)")

    def forbidden_score(*args, **kwargs):
        raise AssertionError("scoring started before historical database identity validation")

    monkeypatch.setattr("llm_rlvr.report.full.score_completion_set", forbidden_score)
    with pytest.raises(ValueError, match="database hash mismatch"):
        reproduce(
            FullManifest(artifacts=(artifact,)),
            tmp_path,
            spider_db_root=scoring_root,
            bird_db_root=scoring_root,
        )


def _role_artifact(path: str) -> FullArtifact:
    root = Path(__file__).resolve().parents[1]
    summary = PublicSummary.model_validate_json(
        (root / "results/public/headline_summary.json").read_text()
    )
    artifact = next(item for item in build_manifest(summary).artifacts if item.completion == path)
    return artifact.model_copy(update={"expected": Count(correct=1, total=1, source="fixture")})


def _restore_role(
    artifact: FullArtifact, root: Path, scoring_root: Path, updates: dict[str, object]
) -> None:
    from llm_rlvr.data.models import Example

    identity = artifact.identity
    metadata = {
        "model": identity.model,
        "backend": "none",
        "dataset": artifact.dataset,
        "dataset_revision": "fixture",
        "split": identity.split,
        "n_examples": 1,
        "prompt_style": identity.prompt_style,
        "adapter": identity.adapter,
        "few_shot_k": identity.few_shot_k,
        "decoding": {
            "temperature": identity.temperature,
            "max_new_tokens": identity.max_new_tokens or 1024,
        },
    }
    metadata.update(updates)
    provenance = GenerationProvenance.model_validate(metadata)
    example = Example(
        db_id="concert",
        question="q",
        gold_sql="SELECT a FROM t",
        dataset=provenance.dataset,
        split=provenance.split,
    )
    completion = assemble_completion_set([example], ["```sql\nSELECT a FROM t\n```"], provenance)
    write_completion_set(completion, root / artifact.completion)
    historical = score_completion_set(completion, scoring_root)
    path = root / artifact.historical_score
    path.write_text(historical.model_dump_json(exclude={"scoring", "outcomes"}))


@pytest.mark.parametrize(
    "role",
    [
        "results/baseline/spider-dev-zs.json",
        "results/sft/spider-dev-zs-sft-r32.json",
        "results/grpo/spider-dev-zs-grpo-s0.json",
    ],
)
@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "wrong-model"),
        ("split", "train"),
        ("prompt_style", "direct"),
        ("decoding.temperature", 1.0),
        ("adapter", "sft/wrong-adapter"),
        ("few_shot_k", 3),
        ("decoding.max_new_tokens", 256),
    ],
)
def test_wrong_generation_role_is_rejected_before_any_sql(
    role: str,
    field: str,
    value: object,
    tmp_path: Path,
    scoring_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _role_artifact(role)
    if field.startswith("decoding."):
        decoding = {
            "temperature": artifact.identity.temperature,
            "max_new_tokens": artifact.identity.max_new_tokens,
        }
        decoding[field.removeprefix("decoding.")] = value
        updates: dict[str, object] = {"decoding": decoding}
    else:
        updates = {field: value}
    _restore_role(artifact, tmp_path, scoring_root, updates)

    def forbidden_score(*args, **kwargs):
        raise AssertionError("SQL scoring started before generation role validation")

    monkeypatch.setattr("llm_rlvr.report.full.score_completion_set", forbidden_score)
    with pytest.raises(ValueError, match=field):
        reproduce(
            FullManifest(artifacts=(artifact,)),
            tmp_path,
            spider_db_root=scoring_root,
            bird_db_root=scoring_root,
        )


@pytest.mark.parametrize(
    "role",
    [
        "results/baseline/spider-dev-zs.json",
        "results/sft/spider-dev-zs-sft-r32.json",
        "results/grpo/spider-dev-zs-grpo-s2.json",
        "results/baseline/spider-dev-fs3.json",
        "results/baseline/bird-dev-zs-direct.json",
        "results/grpo/bird-dev-zs-grpo-nb-s0.json",
        "results/llama/spider-dev-zs-llama-3.2-3b-instruct.json",
        "results/llama/bird-dev-zs-llama-3.2-3b-instruct-sft-llama-r32.json",
        "results/llama/bird-dev-zs-llama-3.2-3b-instruct-grpo-llama-s0.json",
    ],
)
@pytest.mark.parametrize("modal_path", [False, True])
def test_known_role_identity_accepts_legacy_unknown_pins_and_documented_mount(
    role: str, modal_path: bool, tmp_path: Path, scoring_root: Path
) -> None:
    artifact = _role_artifact(role)
    adapter = artifact.identity.adapter
    updates = {"adapter": f"/runs/{adapter}"} if modal_path and adapter else {}
    _restore_role(artifact, tmp_path, scoring_root, updates)
    (result,) = reproduce(
        FullManifest(artifacts=(artifact,)),
        tmp_path,
        spider_db_root=scoring_root,
        bird_db_root=scoring_root,
    )
    assert result.matches
    assert result.historical.provenance.model_revision is None
    assert result.historical.provenance.adapter_sha256 is None
    assert result.historical.scoring is None


def test_matching_adapter_suffix_under_another_root_is_rejected(
    tmp_path: Path,
    scoring_root: Path,
) -> None:
    artifact = _role_artifact("results/sft/spider-dev-zs-sft-r32.json")
    _restore_role(artifact, tmp_path, scoring_root, {"adapter": "/other/sft/sft-r32"})
    with pytest.raises(ValueError, match="outside documented /runs root"):
        reproduce(
            FullManifest(artifacts=(artifact,)),
            tmp_path,
            spider_db_root=scoring_root,
            bird_db_root=scoring_root,
        )


@pytest.mark.parametrize(
    "recipe", ["unstructured adapter description", "sft/sft-r32 -> grpo/grpo-s{0}"]
)
def test_unsupported_or_incomplete_headline_adapter_recipe_fails(recipe: str) -> None:
    root = Path(__file__).resolve().parents[1]
    summary = PublicSummary.model_validate_json(
        (root / "results/public/headline_summary.json").read_text()
    )
    summary = summary.model_copy(
        update={"headline": summary.headline.model_copy(update={"adapter_path": recipe})}
    )
    with pytest.raises(ValueError, match="adapter_path"):
        build_manifest(summary)


@pytest.mark.parametrize(
    "role,adapter",
    [
        ("results/sft/spider-dev-zs-sft-r32.json", None),
        ("results/grpo/spider-dev-zs-grpo-s0.json", None),
        ("results/grpo/spider-dev-zs-grpo-s0.json", "grpo/grpo-s1"),
    ],
)
def test_missing_or_wrong_seed_adapter_cannot_fill_an_adapter_role(
    role: str,
    adapter: str | None,
    tmp_path: Path,
    scoring_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _role_artifact(role)
    _restore_role(artifact, tmp_path, scoring_root, {"adapter": adapter})

    def forbidden_score(*args, **kwargs):
        raise AssertionError("SQL scoring started before adapter role validation")

    monkeypatch.setattr("llm_rlvr.report.full.score_completion_set", forbidden_score)
    with pytest.raises(ValueError, match="adapter expected"):
        reproduce(
            FullManifest(artifacts=(artifact,)),
            tmp_path,
            spider_db_root=scoring_root,
            bird_db_root=scoring_root,
        )


def test_inventory_identities_match_recorded_headline_and_control_settings() -> None:
    expected = {
        "results/baseline/spider-dev-zs.json": (
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            "cot",
            None,
            0,
            512,
        ),
        "results/sft/spider-dev-zs-sft-r32.json": (
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            "cot",
            "sft/sft-r32",
            0,
            512,
        ),
        "results/grpo/spider-dev-zs-grpo-s1.json": (
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            "cot",
            "grpo/grpo-s1",
            0,
            1024,
        ),
        "results/baseline/spider-dev-fs3.json": (
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            "cot",
            None,
            3,
            512,
        ),
        "results/baseline/bird-dev-zs-direct.json": (
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            "direct",
            None,
            0,
            512,
        ),
        "results/sft/spider-dev-zs-direct-sft-r32.json": (
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            "direct",
            "sft/sft-r32",
            0,
            512,
        ),
        "results/grpo/bird-dev-zs-grpo-nb-s0.json": (
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            "cot",
            "grpo/grpo-nb-s0",
            0,
            1024,
        ),
        "results/llama/spider-dev-zs-llama-3.2-3b-instruct.json": (
            "unsloth/Llama-3.2-3B-Instruct",
            "cot",
            None,
            0,
            1024,
        ),
        "results/llama/bird-dev-zs-llama-3.2-3b-instruct-sft-llama-r32.json": (
            "unsloth/Llama-3.2-3B-Instruct",
            "cot",
            "sft/sft-llama-r32",
            0,
            1024,
        ),
        "results/llama/bird-dev-zs-llama-3.2-3b-instruct-grpo-llama-s0.json": (
            "unsloth/Llama-3.2-3B-Instruct",
            "cot",
            "grpo/grpo-llama-s0",
            0,
            1024,
        ),
    }
    for path, settings in expected.items():
        identity = _role_artifact(path).identity
        assert (
            identity.model,
            identity.prompt_style,
            identity.adapter,
            identity.few_shot_k,
            identity.max_new_tokens,
        ) == settings
        assert identity.split == "dev"
        assert identity.temperature == 0
