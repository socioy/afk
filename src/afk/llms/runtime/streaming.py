"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Module: runtime/streaming.py.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import cast

from ..errors import (
    LLMCancelledError,
    LLMCapabilityError,
    LLMInterruptedError,
    LLMInvalidResponseError,
)
from ..types import LLMResponse, LLMStreamEvent, LLMStreamHandle, StreamCompletedEvent

_STREAM_END = object()


class RuntimeStreamHandle(LLMStreamHandle):
    """Single-consumer stream control handle with cancel/interrupt support."""

    def __init__(
        self,
        *,
        source: AsyncIterator[LLMStreamEvent],
        interrupt_callback: Callable[[], Awaitable[None]] | None,
        cancel_callback: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._source = source
        self._interrupt_callback = interrupt_callback
        self._cancel_callback = cancel_callback
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._done = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._result: LLMResponse | None = None
        self._error: Exception | None = None
        self._cancelled = False
        self._interrupted = False
        self._consumed = False

    def _ensure_started(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        completed_count = 0
        try:
            async for event in self._source:
                if isinstance(event, StreamCompletedEvent):
                    completed_count += 1
                    if completed_count > 1:
                        raise LLMInvalidResponseError(
                            "Stream emitted more than one completion event"
                        )
                    self._result = event.response
                await self._queue.put(event)
            if completed_count != 1:
                raise LLMInvalidResponseError(
                    "Stream ended without exactly one completion event"
                )
        except asyncio.CancelledError:
            if self._interrupted:
                self._error = LLMInterruptedError("Stream interrupted")
            else:
                self._error = LLMCancelledError("Stream cancelled")
        except Exception as error:
            self._error = error
        finally:
            if self._error is not None:
                await self._queue.put(self._error)
            await self._queue.put(_STREAM_END)
            self._done.set()

    async def _iter_events(self) -> AsyncIterator[LLMStreamEvent]:
        self._ensure_started()
        while True:
            item = await self._queue.get()
            if item is _STREAM_END:
                break
            if isinstance(item, Exception):
                raise item
            yield cast(LLMStreamEvent, item)

    @property
    def events(self) -> AsyncIterator[LLMStreamEvent]:
        if self._consumed:
            raise RuntimeError("Stream handle supports only one events consumer")
        self._consumed = True
        return self._iter_events()

    async def cancel(self) -> None:
        self._ensure_started()
        self._cancelled = True
        if self._cancel_callback is not None:
            await self._cancel_callback()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def interrupt(self) -> None:
        self._ensure_started()
        self._interrupted = True
        if self._interrupt_callback is None:
            raise LLMCapabilityError("Stream interrupt is not supported")
        await self._interrupt_callback()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def await_result(self) -> LLMResponse | None:
        self._ensure_started()
        await self._done.wait()
        if self._error is not None:
            if isinstance(self._error, (LLMCancelledError, LLMInterruptedError)):
                return None
            raise self._error
        return self._result
