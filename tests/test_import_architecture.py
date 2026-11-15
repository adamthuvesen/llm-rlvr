"""Generic training imports must not initialize SQL or optional runtime packages."""

from __future__ import annotations

import subprocess
import sys


def test_generic_training_imports_without_sql_or_heavy_dependencies(tmp_path) -> None:
    script = """
import importlib.abc
import sys

class BlockDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = (
            'llm_rlvr.tasks', 'llm_rlvr.data', 'llm_rlvr.eval',
            'torch', 'trl', 'peft', 'transformers', 'datasets',
        )
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise ImportError('generic training import reached blocked dependency ' + fullname + '; import it lazily inside the function that needs it')

sys.meta_path.insert(0, BlockDependencies())
from llm_rlvr.harness import RewardConfig
from llm_rlvr.train.prepared import execute_prepared_grpo, execute_prepared_sft
assert callable(execute_prepared_grpo)
assert callable(execute_prepared_sft)
assert RewardConfig().arm == 'shaped'
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
