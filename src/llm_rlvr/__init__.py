"""llm-rlvr: execution-verified RL for text-to-SQL.

The package contains data loaders, a read-only SQLite sandbox, an execution-match verifier, and the
SFT/GRPO training path. The verifier that scores evaluation artifacts is also the training reward.
"""

__version__ = "0.1.0"
