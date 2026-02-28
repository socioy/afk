"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Internal A2A protocol wrapper for in-process agent orchestration.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from ...llms.types import JSONValue
from ..contracts import (
    AgentCommunicationProtocol,
    AgentDeadLetter,
    AgentInvocationRequest,
    AgentInvocationResponse,
    AgentProtocolEvent,
)
from .delivery import A2ADeliveryStore, InMemoryA2ADeliveryStore

EnvelopeType = Literal["request", "response", "event"]


@dataclass(frozen=True, slots=True)
class InternalA2AEnvelope:
    """Typed internal message envelope with end-to-end correlation IDs."""

    message_type: EnvelopeType
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
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class InternalA2AProtocol(AgentCommunicationProtocol):
    """At-least-once in-process A2A protocol with dedupe and dead-letter support."""

    protocol_id = "internal.a2a.v1"

    def __init__(
        self,
        *,
        dispatch: Callable[
            [AgentInvocationRequest], Awaitable[AgentInvocationResponse]
        ],
        delivery_store: A2ADeliveryStore | None = None,
    ) -> None:
        self._dispatch = dispatch
        self._delivery_store = delivery_store or InMemoryA2ADeliveryStore()
        self._event_log: list[AgentProtocolEvent] = []
        self._dead_letters: list[AgentDeadLetter] = []
        self._tasks: dict[str, dict[str, JSONValue]] = {}
        self._lock = asyncio.Lock()

    def events(self) -> list[AgentProtocolEvent]:
        """Return a snapshot of emitted protocol events."""
        return list(self._event_log)

    def dead_letters(self) -> list[AgentDeadLetter]:
        """Return a snapshot of accumulated dead-letter entries."""
        return list(self._dead_letters)

    async def invoke(self, request: AgentInvocationRequest) -> AgentInvocationResponse:
        """Invoke one target agent request with dedupe-aware at-least-once semantics."""
        events: list[AgentProtocolEvent] = []
        response = await self._invoke_internal(request, sink=events.append)
        async with self._lock:
            self._event_log.extend(events)
        return response

    async def invoke_stream(
        self,
        request: AgentInvocationRequest,
    ) -> AsyncIterator[AgentProtocolEvent]:
        """Invoke and stream emitted protocol events for this single request."""
        events: list[AgentProtocolEvent] = []
        await self._invoke_internal(request, sink=events.append)
        async with self._lock:
            self._event_log.extend(events)
        for event in events:
            yield event

    async def _invoke_internal(
        self,
        request: AgentInvocationRequest,
        *,
        sink: Callable[[AgentProtocolEvent], None],
    ) -> AgentInvocationResponse:
        cached = await self._delivery_store.get_success(request.idempotency_key)
        if cached is not None:
            sink(
                AgentProtocolEvent(
                    type="ignored_late_response",
                    request=request,
                    response=cached,
                    details={"deduped": True},
                )
            )
            return cached

        request_envelope = InternalA2AEnvelope(
            message_type="request",
            run_id=request.run_id,
            thread_id=request.thread_id,
            conversation_id=request.conversation_id,
            correlation_id=request.correlation_id,
            idempotency_key=request.idempotency_key,
            source_agent=request.source_agent,
            target_agent=request.target_agent,
            payload=request.payload,
            metadata=request.metadata,
            causation_id=request.causation_id,
        )

        sink(
            AgentProtocolEvent(
                type="queued",
                request=request,
                details={"message_type": request_envelope.message_type},
            )
        )
        sink(
            AgentProtocolEvent(
                type="dispatched",
                request=request,
                details={"protocol": self.protocol_id},
            )
        )
        async with self._lock:
            self._tasks[request.correlation_id] = {
                "status": "running",
                "run_id": request.run_id,
                "thread_id": request.thread_id,
                "target_agent": request.target_agent,
                "idempotency_key": request.idempotency_key,
            }

        try:
            response = await self._dispatch(request)
        except asyncio.CancelledError:
            async with self._lock:
                self._tasks[request.correlation_id] = {
                    **self._tasks.get(request.correlation_id, {}),
                    "status": "cancelled",
                }
            sink(
                AgentProtocolEvent(
                    type="cancelled",
                    request=request,
                    details={"reason": "cancelled"},
                )
            )
            raise
        except Exception as exc:  # pragma: no cover - defensive branch
            async with self._lock:
                self._tasks[request.correlation_id] = {
                    **self._tasks.get(request.correlation_id, {}),
                    "status": "failed",
                    "error": str(exc),
                }
            sink(
                AgentProtocolEvent(
                    type="nacked",
                    request=request,
                    details={"error": str(exc)},
                )
            )
            raise

        response_envelope = InternalA2AEnvelope(
            message_type="response",
            run_id=response.run_id,
            thread_id=response.thread_id,
            conversation_id=response.conversation_id,
            correlation_id=response.correlation_id,
            idempotency_key=response.idempotency_key,
            source_agent=response.source_agent,
            target_agent=response.target_agent,
            payload={
                "success": response.success,
                "output": response.output,
                "error": response.error,
            },
            metadata=response.metadata,
            causation_id=request.correlation_id,
        )

        if response.success:
            await self._delivery_store.record_success(
                request.idempotency_key,
                response,
            )
            async with self._lock:
                self._tasks[request.correlation_id] = {
                    **self._tasks.get(request.correlation_id, {}),
                    "status": "completed",
                    "success": True,
                }
            sink(
                AgentProtocolEvent(
                    type="acked",
                    request=request,
                    response=response,
                    details={"message_type": response_envelope.message_type},
                )
            )
            sink(
                AgentProtocolEvent(type="completed", request=request, response=response)
            )
        else:
            async with self._lock:
                self._tasks[request.correlation_id] = {
                    **self._tasks.get(request.correlation_id, {}),
                    "status": "failed",
                    "success": False,
                    "error": response.error or "unknown",
                }
            sink(
                AgentProtocolEvent(
                    type="nacked",
                    request=request,
                    response=response,
                    details={"error": response.error or "unknown"},
                )
            )
            sink(AgentProtocolEvent(type="failed", request=request, response=response))

        return response

    async def record_dead_letter(
        self,
        request: AgentInvocationRequest,
        *,
        error: str,
        attempts: int,
    ) -> None:
        """Record exhausted retries as a dead-letter event."""
        dead_letter = AgentDeadLetter(request=request, error=error, attempts=attempts)
        await self._delivery_store.record_dead_letter(dead_letter)
        event = AgentProtocolEvent(
            type="dead_letter",
            request=request,
            details={"error": error, "attempts": attempts},
        )
        async with self._lock:
            self._dead_letters.append(dead_letter)
            self._event_log.append(event)

    async def get_task(self, task_id: str) -> dict[str, JSONValue]:
        """Fetch tracked task metadata by task/correlation id."""
        async with self._lock:
            payload = self._tasks.get(task_id)
            if payload is None:
                raise KeyError(f"Unknown task_id '{task_id}'")
            return dict(payload)

    async def cancel_task(self, task_id: str) -> dict[str, JSONValue]:
        """Mark a tracked task as cancellation requested."""
        async with self._lock:
            payload = self._tasks.get(task_id)
            if payload is None:
                raise KeyError(f"Unknown task_id '{task_id}'")
            current_status = payload.get("status")
            if current_status in {"completed", "failed", "cancelled"}:
                return dict(payload)
            payload = {**payload, "status": "cancel_requested"}
            self._tasks[task_id] = payload
            return dict(payload)
