"""A file-level `# pyright:` mode comment must not turn strict type checking off for that file."""

import io
import os
import re
import subprocess
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNGRADE = re.compile(
    r"^\s*#\s*pyright:\s*(?:[^#\n]*,\s*)?(?:basic|standard|off|recommended|all|strict\s*=\s*false)\b"
)


def is_python_file(path: Path) -> bool:
    if path.suffix:
        return path.suffix == ".py"
    with path.open("rb") as handle:
        first_line = handle.readline()
    return first_line.startswith(b"#!") and b"python" in first_line


def checkout_python_files() -> list[Path]:
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=True,
    ).stdout.decode()
    paths = (ROOT / name for name in listing.split("\0") if name)
    return [path for path in paths if path.is_file() and is_python_file(path)]


def test_no_file_downgrades_type_checking_mode() -> None:
    downgrades = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in checkout_python_files()
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1)
        if DOWNGRADE.match(line)
    ]
    assert not downgrades, "file-level `# pyright:` comment turns strict mode off: " + ", ".join(
        downgrades
    )


def test_downgrade_pattern_finds_a_known_mode_comment() -> None:
    assert DOWNGRADE.match("# pyright: basic")
    assert DOWNGRADE.match("#pyright: reportMissingImports=false, strict=false")
    assert not DOWNGRADE.match("# pyright: ignore[reportUnknownMemberType]")


IGNORE_WITH_REASON = re.compile(r"ignore\[[A-Za-z]+(?:,\s*[A-Za-z]+)*\]\s+#\s*\S")
FILE_WIDE_NOQA = re.compile(r"#\s*ruff:\s*noqa", re.IGNORECASE)
NOQA_ON_BAN = re.compile(r"#\s*noqa\b.*\bTID251", re.IGNORECASE)


def suppression_problems(source: str) -> list[tuple[int, str]]:
    """Real comments only: a suppression inside a string literal does not count."""
    problems: list[tuple[int, str]] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        pyright = re.search(r"#\s*pyright:\s*(?P<directive>.*)", token.string)
        if pyright and pyright.group("directive").startswith("ignore"):
            if not IGNORE_WITH_REASON.match(pyright.group("directive")):
                problems.append((token.start[0], "`# pyright: ignore[rule]  # reason` needs both"))
        elif FILE_WIDE_NOQA.search(token.string):
            problems.append((token.start[0], "file-wide `# ruff: noqa`; fix the cause"))
        elif NOQA_ON_BAN.search(token.string):
            problems.append((token.start[0], "`noqa` on a module ban; add an owner in pyproject"))
    return problems


def test_ignores_name_a_rule_and_a_reason_and_no_noqa_lifts_a_ban() -> None:
    problems = [
        f"{path.relative_to(ROOT)}:{line}: {message}"
        for path in checkout_python_files()
        for line, message in suppression_problems(path.read_text(errors="replace"))
    ]
    assert not problems, "\n".join(problems)


def test_suppression_detector_finds_known_cases() -> None:
    def lines(source: str) -> list[int]:
        return [line for line, _ in suppression_problems(source)]

    assert lines("x = 1  # pyright: ignore[reportAny]\n") == [1]
    assert lines("x = 1  # pyright: ignore[reportAny]  # runtime proof\n") == []
    assert lines("# ruff: noqa\n") == [1]
    assert lines("import subprocess  # noqa: TID251\n") == [1]
    assert lines("import os  # noqa: F401\n") == []
    assert lines("x = '# noqa: TID251'\n") == []
