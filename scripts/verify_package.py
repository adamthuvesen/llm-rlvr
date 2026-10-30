"""Install the built wheel in a temporary environment using locked CPU dependencies."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
    }

    def run(command: list[str], cwd: Path) -> str:
        result = subprocess.run(
            command, cwd=cwd, env=env, capture_output=True, text=True, check=True
        )
        return result.stdout

    try:
        run(["uv", "lock", "--check"], root)
        with tempfile.TemporaryDirectory(prefix="llm-rlvr-package-") as temporary:
            work = Path(temporary)
            run(
                [
                    "uv",
                    "build",
                    "--python",
                    sys.executable,
                    "--wheel",
                    "--no-build-isolation",
                    "--out-dir",
                    str(work),
                ],
                root,
            )
            (wheel,) = work.glob("*.whl")
            requirements = work / "requirements.txt"
            requirements.write_text(
                run(
                    ["uv", "export", "--locked", "--no-dev", "--no-editable", "--no-emit-project"],
                    root,
                ),
                encoding="utf-8",
            )
            environment = work / "environment"
            run(["uv", "venv", "--python", sys.executable, str(environment)], work)
            python = environment / "bin/python"
            run(["uv", "pip", "sync", "--python", str(python), str(requirements)], work)
            run(["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)], work)
            smoke = """
import builtins
import importlib
import pkgutil
from pathlib import Path
import sys

blocked = {"torch", "transformers", "vllm", "peft", "trl"}
original_import = builtins.__import__
def cpu_import(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise AssertionError(f"CPU package import required optional dependency: {name}")
    return original_import(name, *args, **kwargs)
builtins.__import__ = cpu_import
import llm_rlvr
assert Path(llm_rlvr.__file__).is_relative_to(Path(sys.prefix))
for module in pkgutil.walk_packages(llm_rlvr.__path__, llm_rlvr.__name__ + "."):
    importlib.import_module(module.name)
"""
            run([str(python), "-I", "-c", smoke], work)
            run([str(python), "-I", "-m", "llm_rlvr.cli", "--help"], work)
            run(
                [
                    str(python),
                    "-I",
                    "-m",
                    "llm_rlvr.cli",
                    "reproduce",
                    "--mode",
                    "public",
                    "--root",
                    str(root),
                ],
                work,
            )
            run([str(python), "-I", str(root / "examples/no_gpu_demo.py")], work)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        print(f"package verification failed: {detail}", file=sys.stderr)
        return 1
    print(
        "package verification: wheel installation, CPU imports, CLI, evidence, "
        "and SQLite demo passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
