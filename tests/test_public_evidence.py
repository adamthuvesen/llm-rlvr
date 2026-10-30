"""The public bundle derives its claims from validated counts and seed identities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_rlvr.report.public import (
    ClaimMap,
    PublicSummary,
    _check_claims,
    _check_readme_table,
    _headline_markdown,
    main,
)

ROOT = Path(__file__).resolve().parents[1]


def _summary() -> PublicSummary:
    return PublicSummary.model_validate_json(
        (ROOT / "results/public/headline_summary.json").read_text()
    )


def test_tracked_public_evidence_passes_and_preserves_headline(
    capsys: pytest.CaptureFixture,
) -> None:
    assert main(ROOT) == 0
    output = capsys.readouterr().out
    assert "3 seeds, 95% CI excludes zero" in output
    assert "**0.779** | **+10.2pt**" in output
    assert "**0.406** | **+6.1pt**" in output
    assert "✓>0" in output


def test_duplicate_seed_ids_rejected() -> None:
    data = _summary().model_dump()
    seeds = data["headline"]["rows"][0]["grpo_seeds"]
    seeds[1]["seed"] = seeds[0]["seed"]
    with pytest.raises(ValueError, match="seed ids must be unique"):
        PublicSummary.model_validate(data)


def test_duplicate_benchmarks_rejected() -> None:
    data = _summary().model_dump()
    data["headline"]["rows"][1]["benchmark"] = data["headline"]["rows"][0]["benchmark"]
    with pytest.raises(ValueError, match="benchmarks must be unique"):
        PublicSummary.model_validate(data)


@pytest.mark.parametrize("seed_count", [1, 2])
def test_seed_count_and_interval_claim_are_derived(seed_count: int) -> None:
    data = _summary().model_dump()
    for row in data["headline"]["rows"]:
        row["grpo_seeds"] = row["grpo_seeds"][:seed_count]
    summary = PublicSummary.model_validate(data)
    headline = _headline_markdown(summary)
    assert "3 seeds" not in headline
    assert f"{seed_count} seed" in headline
    if seed_count == 1:
        assert "point estimate" in headline and "CI" not in headline.splitlines()[2]
    with pytest.raises(ValueError, match="does not match"):
        _check_readme_table((ROOT / "README.md").read_text(), summary)


@pytest.mark.parametrize("gain", [-20, 0])
def test_zero_or_negative_gain_cannot_retain_positive_claim(gain: int) -> None:
    data = _summary().model_dump()
    for row in data["headline"]["rows"]:
        for seed in row["grpo_seeds"]:
            seed["correct"] = row["sft"]["correct"] + gain
    summary = PublicSummary.model_validate(data)
    headline = _headline_markdown(summary)
    assert "CI excludes zero" not in headline
    assert ("CI below zero" if gain < 0 else "CI includes zero") in headline
    with pytest.raises(ValueError, match="does not match"):
        _check_readme_table((ROOT / "README.md").read_text(), summary)


def test_denominators_and_source_identities_must_match() -> None:
    data = _summary().model_dump()
    data["headline"]["rows"][0]["sft"]["total"] += 1
    with pytest.raises(ValueError, match="mixes denominators"):
        PublicSummary.model_validate(data)
    data = _summary().model_dump()
    row = data["headline"]["rows"][0]
    row["grpo_seeds"][0]["source"] = row["base"]["source"]
    with pytest.raises(ValueError, match="sources must be unique"):
        PublicSummary.model_validate(data)


def test_claim_references_are_validated() -> None:
    claims = json.loads((ROOT / "results/public/claims.json").read_text())
    claims["claims"][0]["public_evidence"] = ["../outside.json"]
    with pytest.raises(ValueError, match="unknown public evidence"):
        _check_claims(
            (ROOT / "README.md").read_text(),
            _summary(),
            ClaimMap.model_validate(claims),
            ROOT,
        )


def test_public_failure_has_no_partial_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main(tmp_path) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "public evidence check failed:" in captured.err


@pytest.mark.parametrize("field,value", [("decoding", "sampled"), ("prompt_style", "direct")])
def test_public_metadata_cannot_contradict_greedy_think_tables(field: str, value: str) -> None:
    data = _summary().model_dump()
    data["headline"][field] = value
    with pytest.raises(ValueError, match=field):
        PublicSummary.model_validate(data)
