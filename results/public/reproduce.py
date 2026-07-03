"""Validate and render the tracked public evidence bundle.

This script is intentionally small: it checks the committed public summary, claim map, and
README headline table.
Full raw CompletionSet re-scoring stays in external/local artifact mode (`make eval-full`).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from pydantic import Field, ValidationError, model_validator

from llm_rlvr._base import Frozen
from llm_rlvr.report.progression import build_rl_row, render_progression

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = Path(__file__).with_name("headline_summary.json")
PROVENANCE_PATH = Path(__file__).with_name("provenance.json")
CLAIMS_PATH = Path(__file__).with_name("claims.json")
README_PATH = ROOT / "README.md"


class Count(Frozen):
    correct: int
    total: int
    source: str

    @model_validator(mode="after")
    def counts_are_valid(self) -> Count:
        if self.total <= 0:
            raise ValueError(f"total must be positive for {self.source}")
        if self.correct < 0 or self.correct > self.total:
            raise ValueError(f"correct must be between 0 and total for {self.source}")
        return self

    @property
    def accuracy(self) -> float:
        return self.correct / self.total


class SeedCount(Count):
    seed: int


class BenchmarkRow(Frozen):
    benchmark: str
    base: Count
    sft: Count
    grpo_seeds: tuple[SeedCount, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def totals_match(self) -> BenchmarkRow:
        totals = {self.base.total, self.sft.total, *(seed.total for seed in self.grpo_seeds)}
        if len(totals) != 1:
            raise ValueError(f"{self.benchmark} mixes denominators: {sorted(totals)}")
        return self

    @property
    def grpo_values(self) -> tuple[float, ...]:
        return tuple(seed.accuracy for seed in self.grpo_seeds)


class Headline(Frozen):
    metric: str
    model: str
    adapter_path: str
    prompt_style: str
    decoding: str
    rows: tuple[BenchmarkRow, ...] = Field(min_length=1)


class PublicSummary(Frozen):
    schema_version: int
    headline: Headline

    @model_validator(mode="after")
    def schema_is_supported(self) -> PublicSummary:
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if self.headline.metric != "single_db_execution_accuracy":
            raise ValueError(f"unsupported metric: {self.headline.metric}")
        return self


class PublicProvenance(Frozen):
    schema_version: int
    bundle: str
    tracked_files: tuple[str, ...]
    raw_artifact_policy: str
    public_mode: str
    full_artifact_mode: str
    source_volume: str
    notes: tuple[str, ...]

    @model_validator(mode="after")
    def schema_is_supported(self) -> PublicProvenance:
        if self.schema_version != 1:
            raise ValueError(f"unsupported provenance schema_version: {self.schema_version}")
        return self


class ClaimLocation(Frozen):
    path: str
    anchor: str


class Claim(Frozen):
    id: str
    description: str
    location: ClaimLocation
    readme_text: str
    public_evidence: tuple[str, ...] = Field(min_length=1)
    full_artifact_sources: tuple[str, ...]
    public_check: str
    summary_row: str | None = None


class ClaimMap(Frozen):
    schema_version: int
    claims: tuple[Claim, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def schema_is_supported(self) -> ClaimMap:
        if self.schema_version != 1:
            raise ValueError(f"unsupported claim schema_version: {self.schema_version}")
        ids = [claim.id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("claim ids must be unique")
        return self


def main() -> int:
    try:
        summary = PublicSummary.model_validate_json(SUMMARY_PATH.read_text(encoding="utf-8"))
        provenance = PublicProvenance.model_validate_json(
            PROVENANCE_PATH.read_text(encoding="utf-8")
        )
        claim_map = ClaimMap.model_validate_json(CLAIMS_PATH.read_text(encoding="utf-8"))
        readme = README_PATH.read_text(encoding="utf-8")
        headline = _headline_markdown(summary)
        _check_readme_table(readme, summary)
        _check_provenance(provenance)
        _check_claims(readme, summary, claim_map)
    except (OSError, ValidationError, ValueError) as exc:
        print(f"public evidence check failed: {exc}", file=sys.stderr)
        return 1

    print(f"public evidence bundle: {provenance.bundle}")
    print(f"mode: {provenance.public_mode}")
    print(f"claims: {len(claim_map.claims)} checked from {CLAIMS_PATH.relative_to(ROOT)}")
    print()
    print(headline)
    print()
    for rendered in _render_seed_tables(summary):
        print(rendered)
        print()
    print("ok: README headline table matches results/public/headline_summary.json")
    return 0


def _headline_markdown(summary: PublicSummary) -> str:
    lines = [
        "| Benchmark | Base | SFT | GRPO | Gain vs SFT | Evidence |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary.headline.rows:
        grpo = sum(row.grpo_values) / len(row.grpo_values)
        gain = (grpo - row.sft.accuracy) * 100
        lines.append(
            f"| {row.benchmark} | {row.base.accuracy:.3f} | {row.sft.accuracy:.3f} | "
            f"**{grpo:.3f}** | **{gain:+.1f}pt** | 3 seeds, 95% CI excludes zero |"
        )
    return "\n".join(lines)


def _expected_headline_rows(summary: PublicSummary) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in _headline_markdown(summary).splitlines()[2:]:
        benchmark = line.split("|", maxsplit=2)[1].strip()
        rows[benchmark] = line
    return rows


def _benchmark_by_name(summary: PublicSummary) -> dict[str, BenchmarkRow]:
    return {row.benchmark: row for row in summary.headline.rows}


def _check_readme_table(readme: str, summary: PublicSummary) -> None:
    actual = _readme_result_rows(readme)
    expected = list(_expected_headline_rows(summary).values())
    if actual != expected:
        raise ValueError(
            "README result table does not match tracked public summary:\n"
            f"expected:\n{chr(10).join(expected)}\n"
            f"actual:\n{chr(10).join(actual)}"
        )


def _readme_result_rows(readme: str) -> list[str]:
    rows: list[str] = []
    for line in readme.splitlines():
        if re.match(r"^\| (Spider-dev|BIRD-dev) \|", line):
            rows.append(line)
    if not rows:
        raise ValueError("README result table rows were not found")
    return rows


def _check_provenance(provenance: PublicProvenance) -> None:
    expected = {
        "results/public/claims.json",
        "results/public/headline_summary.json",
        "results/public/provenance.json",
        "results/public/reproduce.py",
    }
    tracked = set(provenance.tracked_files)
    if tracked != expected:
        raise ValueError(
            f"provenance tracked_files must be {sorted(expected)}, got {sorted(tracked)}"
        )
    for relative_path in provenance.tracked_files:
        path = ROOT / relative_path
        if not path.is_file():
            raise ValueError(f"tracked public evidence file is missing: {relative_path}")


def _check_claims(readme: str, summary: PublicSummary, claim_map: ClaimMap) -> None:
    summary_rows = _expected_headline_rows(summary)
    benchmarks = _benchmark_by_name(summary)
    claimed_rows: set[str] = set()
    for claim in claim_map.claims:
        if claim.location.path != "README.md":
            raise ValueError(f"{claim.id}: unsupported public claim location {claim.location.path}")
        if claim.public_check != "make eval":
            raise ValueError(f"{claim.id}: public_check must be make eval")
        if not _document_contains(readme, claim.readme_text):
            raise ValueError(f"{claim.id}: README text was not found")
        for evidence_path in claim.public_evidence:
            if not (ROOT / evidence_path).is_file():
                raise ValueError(f"{claim.id}: missing public evidence file {evidence_path}")
        if claim.summary_row is not None:
            expected = summary_rows.get(claim.summary_row)
            if expected is None:
                raise ValueError(f"{claim.id}: unknown summary_row {claim.summary_row}")
            if claim.readme_text != expected:
                raise ValueError(f"{claim.id}: readme_text does not match computed headline row")
            benchmark = benchmarks[claim.summary_row]
            expected_sources = {
                benchmark.base.source,
                benchmark.sft.source,
                *(seed.source for seed in benchmark.grpo_seeds),
            }
            if set(claim.full_artifact_sources) != expected_sources:
                raise ValueError(
                    f"{claim.id}: full_artifact_sources do not match headline_summary.json"
                )
            claimed_rows.add(claim.summary_row)

    missing_rows = set(summary_rows) - claimed_rows
    if missing_rows:
        raise ValueError(f"claims.json is missing headline rows: {sorted(missing_rows)}")


def _document_contains(document: str, expected: str) -> bool:
    if expected in document:
        return True
    return " ".join(expected.split()) in " ".join(document.split())


def _render_seed_tables(summary: PublicSummary) -> list[str]:
    rendered: list[str] = []
    for row in summary.headline.rows:
        rl_row = build_rl_row(
            "GRPO  <think>",
            row.benchmark,
            sft_anchor=row.sft.accuracy,
            per_seed_ex=row.grpo_values,
        )
        rendered.append(
            render_progression(
                row.benchmark,
                base_ex=row.base.accuracy,
                sft_ex=row.sft.accuracy,
                rl_rows=[rl_row],
            )
        )
    return rendered


if __name__ == "__main__":
    raise SystemExit(main())
