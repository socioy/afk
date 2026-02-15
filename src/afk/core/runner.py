"""
Canonical AFK runner assembled from focused mixins.
"""

from __future__ import annotations

from ..memory import create_memory_store_from_env, new_id  # noqa: F401
from .runner_api import RunnerAPIMixin
from .runner_execution import RunnerExecutionMixin
from .runner_internals import RunnerInternalsMixin
from .runner_interaction import RunnerInteractionMixin
from .runner_types import RunnerConfig


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


__all__ = ["Runner", "RunnerConfig"]
