"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Agent-level streaming types and handle.

Provides real-time token-by-token text deltas, tool lifecycle events, and
status updates during agent execution via ``Runner.run_stream()``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

from ..agents.types import AgentResult, AgentRunEvent, AgentState
from ..llms.types import JSONValue

# ---------------------------------------------------------------------------
# Stream event types
# ---------------------------------------------------------------------------

AgentStreamEventType = Literal[
    "text_delta",
    "tool_started",
    "tool_completed",
    "tool_deferred",
    "tool_background_resolved",
    "tool_background_failed",
    "step_started",
    "step_completed",
    "status_update",
    "run_event",
    "completed",
    "error",
]


@dataclass(frozen=True, slots=True)
class AgentStreamEvent:
    """
    A single event emitted during streamed agent execution.

    Attributes:
        type: Event category.
        text_delta: Incremental text chunk (for ``text_delta`` events).
        tool_name: Tool name (for ``tool_started``/``tool_completed``).
        tool_call_id: Tool call identifier.
        tool_success: Whether tool succeeded (for ``tool_completed``).
        tool_output: Tool output payload (for ``tool_completed``).
        tool_error: Tool error message (for ``tool_completed``).
        step: Current step index.
        state: Current agent state.
        run_event: Full ``AgentRunEvent`` (for ``run_event`` type).
        result: Terminal ``AgentResult`` (for ``completed`` events).
        error: Error message (for ``error`` events).
        data: Additional JSON-safe payload.
    """

    type: AgentStreamEventType
    text_delta: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_success: bool | None = None
    tool_output: JSONValue | None = None
    tool_error: str | None = None
    tool_ticket_id: str | None = None
    step: int | None = None
    state: AgentState | None = None
    run_event: AgentRunEvent | None = None
    result: AgentResult | None = None
    error: str | None = None
    data: dict[str, JSONValue] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def text_delta(delta: str, *, step: int | None = None) -> AgentStreamEvent:
    """Create a text delta stream event."""
    return AgentStreamEvent(type="text_delta", text_delta=delta, step=step)


def tool_started(
    tool_name: str,
    tool_call_id: str | None = None,
    *,
    step: int | None = None,
) -> AgentStreamEvent:
    """Create a tool-started stream event."""
    return AgentStreamEvent(
        type="tool_started",
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        step=step,
    )


def tool_completed(
    tool_name: str,
    tool_call_id: str | None = None,
    *,
    success: bool = True,
    output: JSONValue | None = None,
    error: str | None = None,
    step: int | None = None,
) -> AgentStreamEvent:
    """Create a tool-completed stream event."""
    return AgentStreamEvent(
        type="tool_completed",
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        tool_success=success,
        tool_output=output,
        tool_error=error,
        step=step,
    )


def tool_deferred(
    tool_name: str,
    tool_call_id: str | None = None,
    *,
    ticket_id: str | None = None,
    step: int | None = None,
    data: dict[str, JSONValue] | None = None,
) -> AgentStreamEvent:
    """Create a tool-deferred stream event."""
    return AgentStreamEvent(
        type="tool_deferred",
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        tool_ticket_id=ticket_id,
        step=step,
        data=data or {},
    )


def step_started(step: int, state: AgentState) -> AgentStreamEvent:
    """Create a step-started stream event."""
    return AgentStreamEvent(type="step_started", step=step, state=state)


def step_completed(step: int, state: AgentState) -> AgentStreamEvent:
    """Create a step-completed stream event."""
    return AgentStreamEvent(type="step_completed", step=step, state=state)


def status_update(
    state: AgentState,
    *,
    step: int | None = None,
    data: dict[str, JSONValue] | None = None,
) -> AgentStreamEvent:
    """Create a status update stream event."""
    return AgentStreamEvent(
        type="status_update",
        state=state,
        step=step,
        data=data or {},
    )


def stream_completed(result: AgentResult) -> AgentStreamEvent:
    """Create a stream-completed event with the terminal result."""
    return AgentStreamEvent(type="completed", result=result)


def stream_error(error: str) -> AgentStreamEvent:
    """Create a stream error event."""
    return AgentStreamEvent(type="error", error=error)


# ---------------------------------------------------------------------------
# Stream handle
# ---------------------------------------------------------------------------

_STREAM_END = object()


class AgentStreamHandle:
    """
    Async handle for consuming agent stream events.

    Usage::

        handle = await runner.run_stream(agent, user_message="Hello")
        async for event in handle:
            if event.type == "text_delta":
                print(event.text_delta, end="", flush=True)
        result = handle.result  # terminal AgentResult
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._result: AgentResult | None = None
        self._error: str | None = None
        self._done = False

    async def emit(self, event: AgentStreamEvent) -> None:
        """Push a stream event (called by the runner internally)."""
        if event.type == "completed" and event.result is not None:
            self._result = event.result
        elif event.type == "error":
            self._error = event.error
        await self._queue.put(event)

    async def close(self) -> None:
        """Signal end of stream."""
        self._done = True
        await self._queue.put(_STREAM_END)

    @property
    def result(self) -> AgentResult | None:
        """Terminal result, available after stream is consumed."""
        return self._result

    @property
    def done(self) -> bool:
        """Whether the stream has ended."""
        return self._done

    def __aiter__(self) -> AsyncIterator[AgentStreamEvent]:
        return self._iter_events()

    async def _iter_events(self) -> AsyncIterator[AgentStreamEvent]:
        """Yield events until stream end marker."""
        while True:
            item = await self._queue.get()
            if item is _STREAM_END:
                break
            yield item  # type: ignore[misc]

    async def collect_text(self) -> str:
        """
        Consume all events and return concatenated text output.

        Convenience for callers that only want final text.
        """
        parts: list[str] = []
        async for event in self:
            if event.type == "text_delta" and event.text_delta:
                parts.append(event.text_delta)
        return "".join(parts)
