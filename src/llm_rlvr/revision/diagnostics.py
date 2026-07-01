"""Forced-revision diagnostics logging."""

from __future__ import annotations

from llm_rlvr._base import Frozen
from llm_rlvr.rewards.multiturn import TrajectoryBreakdown
from llm_rlvr.train.diagnostics import CollapseDiagnostics, compute_diagnostics

REVISION_DIAGNOSTICS_FILENAME = "d8-diagnostics.json"
REVISION_DIAGNOSTICS_WINDOW = 128


class RevisionDiagnosticsSeries(Frozen):
    window_episodes: int
    windows: tuple[CollapseDiagnostics, ...]
    overall: CollapseDiagnostics | None


class RevisionEpisodeLog:
    def __init__(self) -> None:
        self.episodes: list[tuple[list[str | None], TrajectoryBreakdown]] = []
        self.windows: list[CollapseDiagnostics] = []

    def __call__(self, sqls: list[str | None], breakdown: TrajectoryBreakdown) -> None:
        self.episodes.append((sqls, breakdown))
        if len(self.episodes) % REVISION_DIAGNOSTICS_WINDOW == 0:
            window = compute_diagnostics(self.episodes[-REVISION_DIAGNOSTICS_WINDOW:])
            self.windows.append(window)
            print(f"[fr-d8] episodes={len(self.episodes)} {window.model_dump_json()}")

    def series(self) -> RevisionDiagnosticsSeries:
        return RevisionDiagnosticsSeries(
            window_episodes=REVISION_DIAGNOSTICS_WINDOW,
            windows=tuple(self.windows),
            overall=compute_diagnostics(self.episodes) if self.episodes else None,
        )
