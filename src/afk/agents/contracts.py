"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Agent-to-agent communication contracts.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol

from ..llms.types import JSONValue

ProtocolEventType = Literal[
    "queued",
    "dispatched",
    "acked",
    "nacked",
    "dead_letter",
    "completed",
    "failed",
    "cancelled",
    "ignored_late_response",
]


@dataclass(frozen=True, slots=True)
class AgentInvocationRequest:
    """Typed request envelope for one agent-to-agent invocation."""

    run_id: str
    thread_id: str
    conversation_id: str
    correlation_id: str
    idempotency_key: str
    source_agent: str
    target_agent: str
    payload: dict[str, JSONValue] = field(default_factory=dict)
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    causation_id: str | None = None
    timeout_s: float | None = None


@dataclass(frozen=True, slots=True)
class AgentInvocationResponse:
    """Normalized response for one agent-to-agent invocation."""

    run_id: str
    thread_id: str
    conversation_id: str
    correlation_id: str
    idempotency_key: str
    source_agent: str
    target_agent: str
    success: bool
    output: JSONValue | None = None
    error: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentProtocolEvent:
    """Observable protocol event emitted during request delivery."""

    type: ProtocolEventType
    request: AgentInvocationRequest
    response: AgentInvocationResponse | None = None
    details: dict[str, JSONValue] = field(default_factory=dict)
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass(frozen=True, slots=True)
class AgentDeadLetter:
    """Dead-letter record for exhausted invocation retries."""

    request: AgentInvocationRequest
    error: str
    attempts: int
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class AgentCommunicationProtocol(Protocol):
    """Protocol abstraction for internal/external agent communication transports."""

    protocol_id: str

    async def invoke(self, request: AgentInvocationRequest) -> AgentInvocationResponse:
        """Send one request and return one terminal response."""
        ...

    async def invoke_stream(
        self,
        request: AgentInvocationRequest,
    ) -> AsyncIterator[AgentProtocolEvent]:
        """Send one request and stream protocol events until terminal state."""
        ...

    async def get_task(self, task_id: str) -> dict[str, JSONValue]:
        """Fetch task metadata by task identifier."""
        ...

    async def cancel_task(self, task_id: str) -> dict[str, JSONValue]:
        """Request task cancellation by task identifier."""
        ...
