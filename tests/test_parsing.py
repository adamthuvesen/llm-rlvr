"""baseline-evaluation: SQL extraction from raw completions."""

from __future__ import annotations

from llm_rlvr.tasks.sql.parsing import extract_sql, well_formed


def test_sql_block() -> None:
    assert extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"


def test_think_then_sql_block() -> None:
    assert extract_sql("<think>reason</think>\n```sql\nSELECT a\nFROM t\n```") == "SELECT a\nFROM t"


def test_last_block_wins() -> None:
    assert extract_sql("```sql\nA\n```\nthen\n```sql\nB\n```") == "B"


def test_answer_tag() -> None:
    assert extract_sql("<answer>SELECT 3</answer>") == "SELECT 3"


def test_plain_fence_fallback() -> None:
    assert extract_sql("```\nSELECT 2\n```") == "SELECT 2"


def test_no_sql_returns_none() -> None:
    assert extract_sql("I cannot answer that.") is None


def test_sqlite_language_tag_not_corrupted() -> None:
    # The ```sqlite language line must be consumed, not partially captured ("ite\nSELECT 1").
    assert extract_sql("```sqlite\nSELECT 1\n```") == "SELECT 1"


def test_unterminated_fence_is_recovered() -> None:
    # A generation truncated at max_new_tokens after the opening fence still yields its query.
    truncated = "<think>reason</think>\n```sql\nSELECT * FROM t WHERE x = 1"
    assert extract_sql(truncated) == "SELECT * FROM t WHERE x = 1"


def test_non_sql_fenced_block_is_not_returned_as_sql() -> None:
    # A stray non-SQL code block must read as no_sql, not be mis-scored as a wrong query.
    assert extract_sql("output:\n```\nfoo bar baz\n```") is None


def test_well_formed_canonical_trace() -> None:
    assert well_formed("<think>reason</think>\n```sql\nSELECT 1\n```") is True
    assert well_formed("<think>r</think>\n```sqlite\nSELECT 1\n```") is True


def test_well_formed_rejects_missing_or_duplicated_think() -> None:
    assert well_formed("```sql\nSELECT 1\n```") is False
    assert well_formed("<think>a</think><think>b</think>\n```sql\nSELECT 1\n```") is False


def test_well_formed_rejects_wrong_fences() -> None:
    assert well_formed("<think>r</think>\n```python\nprint(1)\n```") is False  # non-sql fence
    assert well_formed("<think>r</think>\n```sql\nA\n```\n```sql\nB\n```") is False  # two fences
    assert well_formed("<think>r</think>\n```sql\nSELECT 1") is False  # unterminated
    assert well_formed("<think>r</think> no fence at all") is False


def test_well_formed_requires_think_before_query() -> None:
    assert well_formed("```sql\nSELECT 1\n```\n<think>after</think>") is False
