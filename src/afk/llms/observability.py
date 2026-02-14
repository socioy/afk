from __future__ import annotations

"""
Typed observability primitives for LLM lifecycle events.
"""

from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Protocol

from .types import Usage


LLMLifecycleEventType = Literal[
    "request_start",
    "retry",
    "request_success",
    "request_error",
    "stream_event",
    "cancel",
    "interrupt",
]


@dataclass(frozen=True, slots=True)
class LLMLifecycleEvent:
    """
    One normalized lifecycle event emitted by the base LLM runtime.

    Observer callbacks are best-effort only; failures are swallowed by design.
    """

    event_type: LLMLifecycleEventType
    request_id: str
    provider_id: str
    model: str | None = None
    attempt: int | None = None
    latency_ms: float | None = None
    usage: Usage | None = None
    error_class: str | None = None
    error_message: str | None = None


class LLMObserver(Protocol):
    """Observer callback protocol used by the base LLM runtime."""

    def __call__(self, event: LLMLifecycleEvent) -> None | Awaitable[None]:
        ...


LLMObserverCallback = Callable[[LLMLifecycleEvent], None | Awaitable[None]]

