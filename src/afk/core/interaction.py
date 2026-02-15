"""
Pluggable interaction providers for human-in-the-loop workflows.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol

from ..agents.types import (
    AgentRunEvent,
    ApprovalDecision,
    ApprovalRequest,
    DeferredDecision,
    DecisionKind,
    UserInputDecision,
    UserInputRequest,
)


class InteractionProvider(Protocol):
    """
    Runtime-portable interaction contract.

    Providers can be implemented by API servers, desktop apps, or autonomous
    controllers.

    Every method is async so implementations can call external systems
    (webhooks, message queues, UI event loops) without blocking the runner.
    """

    async def request_approval(
        self,
        request: ApprovalRequest,
    ) -> ApprovalDecision | DeferredDecision:
        """
        Request approval for a gated action.

        Args:
            request: Structured approval request payload from the runner.

        Returns:
            `ApprovalDecision` for immediate allow/deny decisions, or
            `DeferredDecision` when the provider needs asynchronous user action.
        """
        ...

    async def request_user_input(
        self,
        request: UserInputRequest,
    ) -> UserInputDecision | DeferredDecision:
        """
        Request user input for the active run.

        Args:
            request: Structured prompt request payload from the runner.

        Returns:
            `UserInputDecision` for immediate responses, or
            `DeferredDecision` when input will arrive later.
        """
        ...

    async def await_deferred(
        self,
        token: str,
        *,
        timeout_s: float,
    ) -> ApprovalDecision | UserInputDecision | None:
        """
        Wait for a deferred interaction decision.

        Args:
            token: Deferred token returned from `request_approval` or
                `request_user_input`.
            timeout_s: Maximum time to wait for a resolved decision.

        Returns:
            Resolved decision for the deferred token, or `None` on timeout.
        """
        ...

    async def notify(self, event: AgentRunEvent) -> None:
        """
        Receive lifecycle notifications emitted by the runner.

        Args:
            event: Run lifecycle event emitted by the runtime.
        """
        ...


@dataclass(slots=True)
class HeadlessInteractionProvider:
    """
    Non-blocking default interaction provider for autonomous/runtime-server use.

    Attributes:
        approval_fallback: Immediate decision used for approval requests.
        input_fallback: Immediate decision used for user-input requests.
    """

    approval_fallback: DecisionKind = "deny"
    input_fallback: DecisionKind = "deny"

    async def request_approval(
        self,
        request: ApprovalRequest,
    ) -> ApprovalDecision | DeferredDecision:
        """
        Return an immediate fallback approval decision.

        Args:
            request: Approval request emitted by the runner.

        Returns:
            An immediate `ApprovalDecision` based on `approval_fallback`.
        """
        _ = request
        return ApprovalDecision(kind=self.approval_fallback, reason="headless_fallback")

    async def request_user_input(
        self,
        request: UserInputRequest,
    ) -> UserInputDecision | DeferredDecision:
        """
        Return an immediate fallback user-input decision.

        Args:
            request: User-input request emitted by the runner.

        Returns:
            An immediate `UserInputDecision` based on `input_fallback`.
        """
        _ = request
        return UserInputDecision(kind=self.input_fallback, reason="headless_fallback")

    async def await_deferred(
        self,
        token: str,
        *,
        timeout_s: float,
    ) -> ApprovalDecision | UserInputDecision | None:
        """
        Resolve deferred tokens in headless mode.

        Headless mode never stores deferred decisions, so this always returns
        `None`.

        Args:
            token: Deferred token identifier.
            timeout_s: Maximum wait time (ignored in headless mode).

        Returns:
            Always `None`.
        """
        _ = token
        _ = timeout_s
        return None

    async def notify(self, event: AgentRunEvent) -> None:
        """
        Receive lifecycle notifications.

        Args:
            event: Runtime event emitted by the runner.
        """
        _ = event
        return None


@dataclass(slots=True)
class InMemoryInteractiveProvider:
    """
    In-memory provider useful for tests/local development.

    This provider stores deferred decisions and notifications in process.
    It is deterministic and convenient for unit/integration tests.
    """

    _deferred: dict[str, ApprovalDecision | UserInputDecision] = field(
        default_factory=dict
    )
    _notifications: list[AgentRunEvent] = field(default_factory=list)

    async def request_approval(
        self,
        request: ApprovalRequest,
    ) -> ApprovalDecision | DeferredDecision:
        """
        Create a deferred approval token.

        Args:
            request: Approval request payload.

        Returns:
            Deferred token that can later be resolved via `set_deferred_result`.
        """
        return DeferredDecision(token=f"approval:{request.run_id}:{request.step}")

    async def request_user_input(
        self,
        request: UserInputRequest,
    ) -> UserInputDecision | DeferredDecision:
        """
        Create a deferred user-input token.

        Args:
            request: User-input request payload.

        Returns:
            Deferred token that can later be resolved via `set_deferred_result`.
        """
        return DeferredDecision(token=f"input:{request.run_id}:{request.step}")

    async def await_deferred(
        self,
        token: str,
        *,
        timeout_s: float,
    ) -> ApprovalDecision | UserInputDecision | None:
        """
        Resolve a deferred decision from memory or time out.

        Args:
            token: Deferred token to resolve.
            timeout_s: Sleep duration before timing out when unresolved.

        Returns:
            Resolved decision, or `None` when no decision is available.
        """
        if token in self._deferred:
            return self._deferred.pop(token)
        await asyncio.sleep(timeout_s)
        return None

    async def notify(self, event: AgentRunEvent) -> None:
        """
        Store emitted events for assertions/debugging.

        Args:
            event: Runtime event emitted by the runner.
        """
        self._notifications.append(event)

    def set_deferred_result(
        self,
        token: str,
        decision: ApprovalDecision | UserInputDecision,
    ) -> None:
        """
        Set a deferred decision to be returned by `await_deferred`.

        Args:
            token: Deferred token identifier.
            decision: Decision value to return for the token.
        """
        self._deferred[token] = decision

    def notifications(self) -> list[AgentRunEvent]:
        """
        Return captured notification events.

        Returns:
            Snapshot copy of received run events.
        """
        return list(self._notifications)
