"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Canonical AFK runner assembled from focused mixins.
"""

from __future__ import annotations

from ...memory import create_memory_store_from_env, new_id  # noqa: F401
from .api import RunnerAPIMixin
from .execution import RunnerExecutionMixin
from .interaction import RunnerInteractionMixin
from .internals import RunnerInternalsMixin
from .types import RunnerConfig, RunnerDebugConfig


class Runner(
    RunnerExecutionMixin,
    RunnerInteractionMixin,
    RunnerInternalsMixin,
    RunnerAPIMixin,
):
    """
    Canonical runtime runner for AFK agents.

    Composition:
        - `RunnerAPIMixin`: public API (`run`, `resume`, `run_handle`, compact)
        - `RunnerExecutionMixin`: main execution loop and orchestration
        - `RunnerInteractionMixin`: policy + HITL + subagent control
        - `RunnerInternalsMixin`: persistence/serialization/budget helpers
    """


__all__ = ["Runner", "RunnerConfig", "RunnerDebugConfig"]
