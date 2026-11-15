"""Cross-stage SQL example identities remain stable when optional metadata changes."""

import json
from pathlib import Path

from llm_rlvr.data.models import EXAMPLE_IDENTITY_POLICY, Example, example_identity


def test_example_identity_v1_matches_golden_and_ignores_future_metadata() -> None:
    payload = json.loads(
        (Path(__file__).parent / "fixtures" / "example_identity_v1.json").read_text()
    )
    example = Example.model_validate(payload["example"])

    class FutureExample(Example):
        optional_metadata: str = "new metadata"

    assert payload["policy"] == EXAMPLE_IDENTITY_POLICY
    assert example_identity(example) == payload["sha256"]
    assert example_identity(FutureExample(**payload["example"])) == payload["sha256"]


def test_example_identity_distinguishes_semantic_source_fields() -> None:
    example = Example(db_id="concert", question="q", gold_sql="SELECT a FROM t", split="train")
    for changes in (
        {"dataset": "bird"},
        {"split": "dev"},
        {"db_id": "other"},
        {"question": "other"},
        {"gold_sql": "SELECT b FROM t"},
        {"evidence": "external hint"},
    ):
        assert example_identity(example) != example_identity(example.model_copy(update=changes))
