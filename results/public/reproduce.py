"""Validate the tracked evidence using the installed CPU report package."""

from pathlib import Path

from llm_rlvr.report.public import main

if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parents[2]))
