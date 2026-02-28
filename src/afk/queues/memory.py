"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

In-memory task queue implementation.
"""

from __future__ import annotations

import asyncio

from .base import BaseTaskQueue
from .types import TaskItem, TaskStatus


class InMemoryTaskQueue(BaseTaskQueue):
    """
    In-process task queue using ``asyncio.Queue`` and dict-based tracking.

    Suitable for single-process systems and testing. Tasks are lost on
    process restart.
    """

    def __init__(
        self,
        *,
        retry_backoff_base_s: float = 0.0,
        retry_backoff_max_s: float = 30.0,
        retry_backoff_jitter_s: float = 0.0,
    ) -> None:
        super().__init__(
            retry_backoff_base_s=retry_backoff_base_s,
            retry_backoff_max_s=retry_backoff_max_s,
            retry_backoff_jitter_s=retry_backoff_jitter_s,
        )
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: dict[str, TaskItem] = {}

    async def _save_task(self, task: TaskItem) -> None:
        """Persist one task in process-local memory."""
        self._tasks[task.id] = task

    async def _load_task(self, task_id: str) -> TaskItem | None:
        """Load one task by id from process-local memory."""
        return self._tasks.get(task_id)

    async def _delete_task(self, task_id: str) -> None:
        """Delete one task by id from process-local memory."""
        self._tasks.pop(task_id, None)

    async def _push_pending_id(self, task_id: str) -> None:
        """Append one task id to the in-memory pending FIFO."""
        await self._queue.put(task_id)

    async def _pop_pending_id(self, *, timeout: float | None = None) -> str | None:
        """Pop one pending task id with optional timeout."""
        if timeout is None:
            return await self._queue.get()
        if timeout <= 0:
            return None
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except TimeoutError:
            return None

    async def list_tasks(
        self,
        *,
        status: TaskStatus | None = None,
        limit: int = 100,
    ) -> list[TaskItem]:
        """List tasks with optional status filter."""
        items = list(self._tasks.values())
        if status is not None:
            items = [t for t in items if t.status == status]
        return items[:limit]

    @property
    def pending_count(self) -> int:
        """Number of tasks waiting in queue."""
        return self._queue.qsize()

    @property
    def total_count(self) -> int:
        """Total number of tracked tasks."""
        return len(self._tasks)
