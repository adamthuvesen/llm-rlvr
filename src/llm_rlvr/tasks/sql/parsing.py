"""Extract the SQL query from a model's raw completion.

The trained format is `<think>…</think>` followed by a ```sql … ``` block, but the base model
(pre-SFT) is unreliable, so extraction is defensive. Order of preference:
  1. the last *closed* ```sql / ```sqlite block (the language line is consumed, so `sqlite` works);
  2. an `<answer>` block;
  3. the last closed fenced block whose body looks like SQL (avoids a stray ```python block);
  4. an *unterminated* trailing fence (a generation truncated at max_new_tokens mid-query).
Returns None when no SQL can be found — so a truncated-but-usable query is recovered, not lost.
"""

from __future__ import annotations

import re

# A closed fence: an optional language label on the opening line, then the body, up to the close.
_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
# An unterminated fence at the very end of the string (truncated generation).
_OPEN_FENCE = re.compile(r"```([^\n`]*)\n(.*)\Z", re.DOTALL)
_ANSWER = re.compile(r"<answer>\s*(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_SQL_KEYWORDS = ("select", "with", "insert", "update", "delete", "create")
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)


def _lang_is_sql(lang: str) -> bool:
    return lang.strip().lower().startswith("sql")


def _looks_like_sql(body: str) -> bool:
    return body.strip().lower().startswith(_SQL_KEYWORDS)


def well_formed(text: str) -> bool:
    """True iff `text` is exactly one `<think>…</think>` block then one closed ```sql block.

    The trained output format (and the SFT format-rate metric): a single think block, followed by
    a single sql-tagged fenced query. Anything else — missing/duplicated think tags, zero or
    multiple fences, a non-sql fence, or a fence before the reasoning — is malformed.
    """
    opens = [m.start() for m in _THINK_OPEN.finditer(text)]
    closes = [m.start() for m in _THINK_CLOSE.finditer(text)]
    if len(opens) != 1 or len(closes) != 1 or closes[0] < opens[0]:
        return False
    fences = list(_FENCE.finditer(text))
    if len(fences) != 1:
        return False
    fence = fences[0]
    return _lang_is_sql(fence.group(1)) and fence.start() > closes[0]


def extract_sql(text: str) -> str | None:
    """Return the predicted SQL, or None if the completion contains none."""
    closed: list[tuple[str, str]] = [
        (lang.strip(), body.strip()) for lang, body in _FENCE.findall(text)
    ]

    for lang, body in reversed(closed):  # 1. last explicitly sql-tagged block
        if _lang_is_sql(lang):
            return body

    answers: list[str] = _ANSWER.findall(text)  # 2. an <answer> block
    if answers:
        inner = answers[-1].strip()
        inner_blocks: list[tuple[str, str]] = [
            (lang.strip(), body.strip()) for lang, body in _FENCE.findall(inner)
        ]
        for lang, body in reversed(inner_blocks):
            if _lang_is_sql(lang) or _looks_like_sql(body):
                return body
        return inner.strip("` ").strip() or None

    for _lang, body in reversed(closed):  # 3. last block whose body looks like SQL
        if _looks_like_sql(body):
            return body

    match = _OPEN_FENCE.search(text)  # 4. unterminated trailing fence (truncated)
    if match:
        lang, body = match.group(1).strip(), match.group(2).strip()
        if _lang_is_sql(lang) or _looks_like_sql(body):
            return body or None

    return None
