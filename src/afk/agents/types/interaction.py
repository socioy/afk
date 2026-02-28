"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Human-in-the-loop interaction types.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

from ...llms.types import JSONValue
from .common import DecisionKind
from .result import AgentResult


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """
    Request payload for human approval interaction.

    Attributes:
        run_id: Run identifier.
        thread_id: Thread identifier.
        step: Current execution step.
        reason: Reason shown to the approver.
        payload: Additional JSON-safe context for approval UI.
    """

    run_id: str
    thread_id: str
    step: int
    reason: str
    payload: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UserInputRequest:
    """
    Request payload for human user-input interaction.

    Attributes:
        run_id: Run identifier.
        thread_id: Thread identifier.
        step: Current execution step.
        prompt: Prompt text for human response.
        payload: Additional JSON-safe context for the input request.
    """

    run_id: str
    thread_id: str
    step: int
    prompt: str
    payload: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DeferredDecision:
    """
    Deferred interaction token returned by interaction providers.

    Attributes:
        token: Opaque token used to resolve deferred decision later.
        message: Optional provider message for logs/UI.
    """

    token: str
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """
    Resolved decision for an approval request.

    Attributes:
        kind: Decision outcome (`allow`, `deny`, or `defer`).
        reason: Optional explanation of the decision.
    """

    kind: DecisionKind
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class UserInputDecision:
    """
    Resolved decision for a user-input request.

    Attributes:
        kind: Decision outcome (`allow`, `deny`, or `defer`).
        value: User-provided text value when available.
        reason: Optional explanation or fallback reason.
    """

    kind: DecisionKind
    value: str | None = None
    reason: str | None = None


class AgentRunHandle(Protocol):
    """Protocol for asynchronous run lifecycle controls."""

    @property
    def events(self) -> AsyncIterator[AgentRunEvent]:
        """
        Event stream for the run lifecycle.

        Returns:
            Async iterator of `AgentRunEvent`.
        """
        ...

    async def pause(self) -> None:
        """Pause cooperative execution at safe boundaries."""
        ...

    async def resume(self) -> None:
        """Resume execution after pause."""
        ...

    async def cancel(self) -> None:
        """Cancel run and terminate with no result."""
        ...

    async def interrupt(self) -> None:
        """Interrupt in-flight operations where supported."""
        ...

    async def await_result(self) -> AgentResult | None:
        """
        Await terminal result.

        Returns:
            `AgentResult` on completion, or `None` when cancelled.
        """
        ...


# Import for forward reference used by AgentRunHandle.events
from .policy import AgentRunEvent  # noqa: E402, F401
