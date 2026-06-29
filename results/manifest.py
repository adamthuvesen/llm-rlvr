"""Registry of CompletionSets re-derived by ``make eval``.

Stage-3 multi-turn smoke artifacts under ``results/grpo-mt/`` are documented in
``ARTIFACTS.md`` only — they are smoke JSON, not CompletionSets scored here.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from llm_rlvr_sql._base import Frozen

RESULTS = Path(__file__).parent
BASELINE = RESULTS / "baseline"
SFT = RESULTS / "sft"
GRPO = RESULTS / "grpo"
LLAMA = RESULTS / "llama"

GRPO_SEEDS = (0, 1, 2)
LLAMA_TAG = "llama-3.2-3b-instruct"


class DbKind(StrEnum):
    SPIDER = "spider"
    BIRD = "bird"


class ReproRow(Frozen):
    benchmark: str
    setting: str
    completion_path: Path
    db: DbKind


def _row(benchmark: str, setting: str, path: Path, db: DbKind) -> ReproRow:
    return ReproRow(benchmark=benchmark, setting=setting, completion_path=path, db=db)


def baseline_rows() -> tuple[ReproRow, ...]:
    return (
        _row("Spider-dev", "cot zero-shot", BASELINE / "spider-dev-zs.json", DbKind.SPIDER),
        _row("Spider-dev", "cot few-shot k=3", BASELINE / "spider-dev-fs3.json", DbKind.SPIDER),
        _row(
            "Spider-dev", "direct zero-shot", BASELINE / "spider-dev-zs-direct.json", DbKind.SPIDER
        ),
        _row("Spider-dev", "cot zs + sft-r32", SFT / "spider-dev-zs-sft-r32.json", DbKind.SPIDER),
        _row(
            "Spider-dev",
            "direct zs + sft-r32",
            SFT / "spider-dev-zs-direct-sft-r32.json",
            DbKind.SPIDER,
        ),
        _row("BIRD-dev", "cot zero-shot", BASELINE / "bird-dev-zs.json", DbKind.BIRD),
        _row("BIRD-dev", "direct zero-shot", BASELINE / "bird-dev-zs-direct.json", DbKind.BIRD),
        _row("BIRD-dev", "cot zs + sft-r32", SFT / "bird-dev-zs-sft-r32.json", DbKind.BIRD),
    )


def grpo_rows() -> tuple[ReproRow, ...]:
    rows: list[ReproRow] = []
    for seed in GRPO_SEEDS:
        label = f"grpo-s{seed} (shaped)"
        rows.append(
            _row("Spider-dev", label, GRPO / f"spider-dev-zs-grpo-s{seed}.json", DbKind.SPIDER)
        )
        rows.append(_row("BIRD-dev", label, GRPO / f"bird-dev-zs-grpo-s{seed}.json", DbKind.BIRD))
    nb = "grpo-nb-s0 (near-bin)"
    rows.append(_row("Spider-dev", nb, GRPO / "spider-dev-zs-grpo-nb-s0.json", DbKind.SPIDER))
    rows.append(_row("BIRD-dev", nb, GRPO / "bird-dev-zs-grpo-nb-s0.json", DbKind.BIRD))
    return tuple(rows)


def llama_rows() -> tuple[ReproRow, ...]:
    rows: list[ReproRow] = []
    for setting, suffix in (
        ("llama base cot zs", ""),
        ("llama + sft-llama-r32", "-sft-llama-r32"),
        ("llama + grpo-llama-s0", "-grpo-llama-s0"),
    ):
        rows.append(
            _row(
                "Spider-dev",
                setting,
                LLAMA / f"spider-dev-zs-{LLAMA_TAG}{suffix}.json",
                DbKind.SPIDER,
            )
        )
        rows.append(
            _row(
                "BIRD-dev",
                setting,
                LLAMA / f"bird-dev-zs-{LLAMA_TAG}{suffix}.json",
                DbKind.BIRD,
            )
        )
    return tuple(rows)


def repro_manifest() -> tuple[ReproRow, ...]:
    return baseline_rows() + grpo_rows() + llama_rows()


def stage3b_completion_paths() -> tuple[Path, ...]:
    """Stage-3b artifacts scored inside ``reproduce._stage3b_forced_revision``."""
    fr = RESULTS / "grpo-fr"
    return (
        fr / "spider-dev-zs-grpo-fr-pilot-n200.json",
        fr / "spider-dev-fr-grpo-fr-pilot-n200.json",
        fr / "spider-dev-zs-grpo-fr-pilot-t1.0-seed0-n200.json",
        fr / "spider-dev-zs-grpo-fr-pilot-t1.0-seed1-n200.json",
        fr / "spider-dev-fr-error-only-grpo-fr-pilot.json",
        fr / "bird-dev-fr-error-only-grpo-fr-pilot.json",
    )


def all_manifest_paths() -> Sequence[Path]:
    paths = [row.completion_path for row in repro_manifest()]
    paths.extend(stage3b_completion_paths())
    paths.append(GRPO / "spider-dev-zs-grpo-s0.json")  # sliced in stage3b
    return paths
