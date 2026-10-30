"""Re-score the complete restored artifact inventory without replacing historical reports."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from llm_rlvr._base import Frozen
from llm_rlvr.artifacts.provenance import GenerationProvenance
from llm_rlvr.artifacts.runs import require_complete
from llm_rlvr.data.paths import resolve_database, sha256_of
from llm_rlvr.eval.completions import CompletionSet, read_completion_set
from llm_rlvr.eval.score import ScoreReport, score_completion_set
from llm_rlvr.report.public import Count, PublicSummary


class ExpectedGeneration(Frozen):
    model: str
    split: str = "dev"
    prompt_style: Literal["cot", "direct"] = "cot"
    temperature: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    adapter: str | None
    few_shot_k: int = Field(default=0, ge=0)
    max_new_tokens: int | None = Field(default=None, gt=0)


class FullArtifact(Frozen):
    completion: str
    historical_score: str
    dataset: Literal["spider", "bird"]
    identity: ExpectedGeneration
    expected: Count | None = None
    completion_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    historical_score_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def paths_are_relative(self) -> Self:
        for value in (self.completion, self.historical_score):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("full reproduction paths must stay inside the project root")
        return self


class FullManifest(Frozen):
    schema_version: Literal[1] = 1
    artifacts: tuple[FullArtifact, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def artifacts_are_unique(self) -> Self:
        paths = [
            path for item in self.artifacts for path in (item.completion, item.historical_score)
        ]
        if len(paths) != len(set(paths)):
            raise ValueError("full reproduction artifact paths must be unique")
        return self


class ScoreComparison(Frozen):
    artifact: FullArtifact
    completion_sha256: str
    historical_score_sha256: str
    historical: ScoreReport
    rescored: ScoreReport

    @property
    def changed_outcomes(self) -> tuple[int, ...] | None:
        if self.historical.outcomes is None or self.rescored.outcomes is None:
            return None
        return tuple(
            old.index
            for old, new in zip(self.historical.outcomes, self.rescored.outcomes, strict=True)
            if (old.db_id, old.reason, old.matched, old.well_formed)
            != (new.db_id, new.reason, new.matched, new.well_formed)
        )

    @property
    def matches(self) -> bool:
        previous, current = self.historical, self.rescored
        return (
            previous.n == current.n
            and previous.n_correct == current.n_correct
            and previous.n_gold_errors == current.n_gold_errors
            and previous.reasons == current.reasons
            and previous.well_formed_rate == current.well_formed_rate
            and not self.changed_outcomes
        )


def build_manifest(summary: PublicSummary) -> FullManifest:
    adapter_recipe = re.fullmatch(
        r"(?P<sft>sft/[A-Za-z0-9_.-]+) -> (?P<grpo>grpo/[A-Za-z0-9_.-]+)"
        r"\{(?P<seeds>\d+(?:,\d+)*)\}",
        summary.headline.adapter_path,
    )
    if adapter_recipe is None:
        raise ValueError("unsupported headline adapter_path. Expected SFT -> GRPO{seed,...}")
    seeds = [int(seed) for seed in adapter_recipe["seeds"].split(",")]
    if len(seeds) != len(set(seeds)):
        raise ValueError("headline adapter_path contains duplicate seed ids")
    sft_adapter = adapter_recipe["sft"]
    grpo_adapters = {seed: f"{adapter_recipe['grpo']}{seed}" for seed in seeds}
    artifacts = []
    for row in summary.headline.rows:
        dataset: Literal["spider", "bird"]
        if row.benchmark == "Spider-dev":
            dataset = "spider"
        elif row.benchmark == "BIRD-dev":
            dataset = "bird"
        else:
            raise ValueError(f"unsupported full reproduction benchmark: {row.benchmark}")
        if any(seed.seed not in grpo_adapters for seed in row.grpo_seeds):
            raise ValueError(f"headline adapter_path does not cover {row.benchmark} seed ids")
        roles = [
            (row.base, None),
            (row.sft, sft_adapter),
            *((seed, grpo_adapters[seed.seed]) for seed in row.grpo_seeds),
        ]
        for count, adapter in roles:
            prefix, suffix = "external ", " + score"
            if not count.source.startswith(prefix) or not count.source.endswith(suffix):
                raise ValueError(f"unsupported external artifact source: {count.source}")
            completion = count.source[len(prefix) : -len(suffix)]
            artifacts.append(
                FullArtifact(
                    completion=completion,
                    historical_score=Path(completion).with_suffix(".score.json").as_posix(),
                    dataset=dataset,
                    identity=ExpectedGeneration(
                        model=summary.headline.model,
                        prompt_style=summary.headline.prompt_style,
                        adapter=adapter,
                        max_new_tokens=512 if adapter in (None, sft_adapter) else 1024,
                    ),
                    expected=count,
                )
            )
    # Settings and token budgets are recorded in docs/internal/artifacts.md.
    qwen_base = ExpectedGeneration(
        model="Qwen/Qwen2.5-Coder-3B-Instruct", adapter=None, max_new_tokens=512
    )
    qwen_sft_direct = qwen_base.model_copy(
        update={"adapter": "sft/sft-r32", "prompt_style": "direct"}
    )
    qwen_near_binary = qwen_base.model_copy(
        update={"adapter": "grpo/grpo-nb-s0", "max_new_tokens": 1024}
    )
    llama_base = ExpectedGeneration(
        model="unsloth/Llama-3.2-3B-Instruct", adapter=None, max_new_tokens=1024
    )
    llama_sft = llama_base.model_copy(update={"adapter": "sft/sft-llama-r32"})
    llama_grpo = llama_base.model_copy(update={"adapter": "grpo/grpo-llama-s0"})
    controls = (
        ("results/baseline/spider-dev-fs3.json", qwen_base.model_copy(update={"few_shot_k": 3})),
        (
            "results/baseline/spider-dev-zs-direct.json",
            qwen_base.model_copy(update={"prompt_style": "direct"}),
        ),
        (
            "results/baseline/bird-dev-zs-direct.json",
            qwen_base.model_copy(update={"prompt_style": "direct"}),
        ),
        ("results/sft/spider-dev-zs-direct-sft-r32.json", qwen_sft_direct),
        ("results/grpo/spider-dev-zs-grpo-nb-s0.json", qwen_near_binary),
        ("results/grpo/bird-dev-zs-grpo-nb-s0.json", qwen_near_binary),
        ("results/llama/spider-dev-zs-llama-3.2-3b-instruct.json", llama_base),
        ("results/llama/spider-dev-zs-llama-3.2-3b-instruct-sft-llama-r32.json", llama_sft),
        ("results/llama/spider-dev-zs-llama-3.2-3b-instruct-grpo-llama-s0.json", llama_grpo),
        ("results/llama/bird-dev-zs-llama-3.2-3b-instruct.json", llama_base),
        ("results/llama/bird-dev-zs-llama-3.2-3b-instruct-sft-llama-r32.json", llama_sft),
        ("results/llama/bird-dev-zs-llama-3.2-3b-instruct-grpo-llama-s0.json", llama_grpo),
    )
    for completion, identity in controls:
        artifacts.append(
            FullArtifact(
                completion=completion,
                historical_score=Path(completion).with_suffix(".score.json").as_posix(),
                dataset="bird" if Path(completion).name.startswith("bird-") else "spider",
                identity=identity,
            )
        )
    return FullManifest(artifacts=tuple(artifacts))


def _check_generation_identity(
    expected: ExpectedGeneration, actual: GenerationProvenance, path: Path
) -> None:
    for field in ("model", "split", "prompt_style", "few_shot_k"):
        wanted, recorded = getattr(expected, field), getattr(actual, field)
        if wanted != recorded:
            raise ValueError(
                f"generation identity mismatch: {path}: "
                f"{field} expected {wanted!r}, got {recorded!r}"
            )
    if actual.decoding.temperature != expected.temperature:
        raise ValueError(
            f"generation identity mismatch: {path}: decoding.temperature "
            f"expected {expected.temperature}, got {actual.decoding.temperature}"
        )
    if (
        expected.max_new_tokens is not None
        and actual.decoding.max_new_tokens != expected.max_new_tokens
    ):
        raise ValueError(
            f"generation identity mismatch: {path}: decoding.max_new_tokens "
            f"expected {expected.max_new_tokens}, got {actual.decoding.max_new_tokens}"
        )
    adapter = actual.adapter
    if adapter is not None:
        adapter_path = Path(adapter)
        if adapter_path.is_absolute():
            if not adapter_path.is_relative_to("/runs"):
                raise ValueError(
                    f"generation identity mismatch: {path}: "
                    f"adapter is outside documented /runs root: {adapter!r}"
                )
            adapter_path = adapter_path.relative_to("/runs")
        adapter = adapter_path.as_posix()
    if adapter != expected.adapter:
        raise ValueError(
            f"generation identity mismatch: {path}: "
            f"adapter expected {expected.adapter!r}, got {actual.adapter!r}"
        )


def _require_artifact_files(manifest: FullManifest, root: Path) -> None:
    missing = [
        relative
        for item in manifest.artifacts
        for relative in (item.completion, item.historical_score)
        if not (root / relative).is_file()
    ]
    if missing:
        raise ValueError(
            "restore the full external artifact bundle. Missing:\n" + "\n".join(missing)
        )


def reproduce(
    manifest: FullManifest,
    root: Path,
    *,
    spider_db_root: Path,
    bird_db_root: Path,
    workers: int = 1,
) -> tuple[ScoreComparison, ...]:
    """Validate every input before scoring. Legacy artifacts have no claimed historical hash."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    _require_artifact_files(manifest, root)
    roots = {"spider": spider_db_root, "bird": bird_db_root}
    database_hashes: dict[Path, str] = {}
    loaded: list[tuple[FullArtifact, CompletionSet, ScoreReport, str, str]] = []
    for artifact in manifest.artifacts:
        completion_path = root / artifact.completion
        historical_path = root / artifact.historical_score
        for path in (completion_path, historical_path):
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"artifact resolves outside project root: {path}")
            require_complete(path)
        completion_hash = sha256_of(completion_path)
        historical_hash = sha256_of(historical_path)
        for supplied, actual, path in (
            (artifact.completion_sha256, completion_hash, completion_path),
            (artifact.historical_score_sha256, historical_hash, historical_path),
        ):
            if supplied is not None and supplied != actual:
                raise ValueError(f"artifact hash mismatch: {path}")
        completions = read_completion_set(completion_path)
        historical = ScoreReport.model_validate_json(historical_path.read_text(encoding="utf-8"))
        if completions.provenance != historical.provenance:
            raise ValueError(f"generation provenance mismatch: {historical_path}")
        _check_generation_identity(artifact.identity, completions.provenance, completion_path)
        if historical.scoring is not None and (
            historical.scoring.completion_schema_version != completions.schema_version
            or historical.scoring.completion_set_sha256 != completions.canonical_sha256
        ):
            raise ValueError(f"historical scoring input digest mismatch: {completion_path}")
        if completions.provenance.dataset != artifact.dataset:
            raise ValueError(f"dataset mismatch: {completion_path}")
        if artifact.expected is not None and (
            historical.n != artifact.expected.total
            or historical.n_correct != artifact.expected.correct
        ):
            raise ValueError(f"historical score disagrees with public summary: {historical_path}")
        for db_id in {item.example.db_id for item in completions.items}:
            resolve_database(db_id, roots[artifact.dataset])
        if historical.scoring is not None:
            used_ids = {
                item.example.db_id for item in completions.items if item.samples[0].sql is not None
            }
            if {database.db_id for database in historical.scoring.databases} != used_ids:
                raise ValueError(
                    f"historical scoring database inventory mismatch: {historical_path}"
                )
            for database in historical.scoring.databases:
                path = resolve_database(database.db_id, roots[artifact.dataset]).resolve()
                if path not in database_hashes:
                    database_hashes[path] = sha256_of(path)
                if database_hashes[path] != database.sha256:
                    raise ValueError(f"historical scoring database hash mismatch: {path}")
        loaded.append((artifact, completions, historical, completion_hash, historical_hash))
    return tuple(
        ScoreComparison(
            artifact=artifact,
            completion_sha256=completion_hash,
            historical_score_sha256=historical_hash,
            historical=historical,
            rescored=score_completion_set(
                completions, roots[artifact.dataset], config=historical.verifier, workers=workers
            ),
        )
        for artifact, completions, historical, completion_hash, historical_hash in loaded
    )


def main(
    root: Path | None = None,
    *,
    spider_db_root: Path | None = None,
    bird_db_root: Path | None = None,
    workers: int = 1,
) -> int:
    root = root or Path.cwd()
    try:
        summary = PublicSummary.model_validate_json(
            (root / "results/public/headline_summary.json").read_text(encoding="utf-8")
        )
        manifest = build_manifest(summary)
        _require_artifact_files(manifest, root)
        spider_root = (
            spider_db_root
            or Path(os.environ.get("TEXT2SQL_SPIDER_ROOT", str(root / "data/spider/spider_data")))
            / "database"
        )
        if bird_db_root is None:
            override = os.environ.get("TEXT2SQL_BIRD_ROOT")
            candidates = (
                [Path(override) / "dev_databases"]
                if override
                else sorted((root / "data/bird").glob("**/dev_databases"))
            )
            if len(candidates) > 1:
                raise ValueError(
                    "multiple BIRD database roots found. Set --bird-db-root explicitly"
                )
            bird_db_root = candidates[0] if candidates else root / "data/bird/dev_databases"
        comparisons = reproduce(
            manifest, root, spider_db_root=spider_root, bird_db_root=bird_db_root, workers=workers
        )
    except (OSError, ValueError) as exc:
        print(f"full reproduction failed: {exc}", file=sys.stderr)
        return 1
    print(f"full reproduction: {len(comparisons)} artifact pairs checked")
    for comparison in comparisons:
        old, new = comparison.historical, comparison.rescored
        pin = (
            "pinned"
            if comparison.artifact.completion_sha256 and comparison.artifact.historical_score_sha256
            else "no historical file hash pins"
        )
        database_identity = "verified" if old.scoring is not None else "not recorded historically"
        outcomes = comparison.changed_outcomes
        outcome_text = "not recorded historically" if outcomes is None else str(outcomes)
        print(
            f"{comparison.artifact.completion}: {old.n_correct}/{old.n} -> "
            f"{new.n_correct}/{new.n}, "
            f"delta={new.execution_accuracy - old.execution_accuracy:+.6f}, "
            f"gold_errors={old.n_gold_errors}->{new.n_gold_errors}, "
            f"outcome_changes={outcome_text}, database_identity={database_identity}, {pin}"
        )
        for outcome in new.outcomes or ():
            if outcome.reason == "gold_error":
                print(f"  item={outcome.index} db_id={outcome.db_id}: {outcome.detail}")
    success = all(item.matches and item.rescored.n_gold_errors == 0 for item in comparisons)
    if not success:
        print("full reproduction found score changes or reference failures", file=sys.stderr)
    return 0 if success else 1
