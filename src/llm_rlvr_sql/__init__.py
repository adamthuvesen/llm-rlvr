"""llm-rlvr-sql: a self-correcting text-to-SQL agent trained with RL.

Stage-0 layer (this package): data loaders, a read-only SQLite sandbox, an execution-match
verifier, and a reproducible baseline. The verifier that scores the baseline is the same one
that becomes the Stage-2 training reward, so its correctness is load-bearing.
"""

__version__ = "0.1.0"
