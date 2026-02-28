"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Protocol interfaces for runtime hooks.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol

from ...llms.types import JSONValue
from .common import AgentState
from .config import RouterDecision, RouterInput
from .policy import PolicyDecision, PolicyEvent


class InstructionRole(Protocol):
    """
    Hook protocol for dynamic instruction augmentation.

    Implementations receive run context and current state and can return:
    one string, a list of strings, or ``None``.
    """

    def __call__(
        self,
        context: dict[str, JSONValue],
        state: AgentState,
    ) -> str | list[str] | None | Awaitable[str | list[str] | None]:
        """
        Return additional instruction text for current state.

        Args:
            context: JSON-safe run context.
            state: Current runtime state.

        Returns:
            Optional instruction text chunks.
        """
        ...


class PolicyRole(Protocol):
    """
    Hook protocol for runtime policy decisions.

    Implementations can deny/defer/rewrite runtime actions before execution.
    """

    def __call__(
        self,
        event: PolicyEvent,
    ) -> PolicyDecision | Awaitable[PolicyDecision]:
        """
        Return a policy decision for the given runtime event.

        Args:
            event: Runtime event payload under policy evaluation.

        Returns:
            Policy decision for the event.
        """
        ...


class SubagentRouter(Protocol):
    """
    Hook protocol for selecting subagents during execution.

    Router implementations decide target subagents and whether they run in
    parallel for the current step.
    """

    def __call__(
        self,
        data: RouterInput,
    ) -> RouterDecision | Awaitable[RouterDecision]:
        """
        Return a routing decision for subagent execution.

        Args:
            data: Router input payload with context and transcript.

        Returns:
            Router decision containing targets and parallelism flag.
        """
        ...
