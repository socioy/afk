"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Redis-backed persistent task queue.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import asdict
from typing import Any

from ..llms.types import JSONValue
from .base import BaseTaskQueue
from .types import RetryPolicy, TaskItem, TaskStatus

logger = logging.getLogger("afk.queues.redis")

_RELEASE_LOCK_IF_OWNER_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


class RedisTaskQueue(BaseTaskQueue):
    """
    Persistent task queue using Redis for durability across restarts.

    Uses:
    - Redis list (``{prefix}:pending``) for the FIFO queue
    - Redis list (``{prefix}:inflight``) for crash-safe in-flight tracking
    - Redis hash (``{prefix}:tasks``) for task state tracking

    Requires ``redis.asyncio`` (``pip install redis``).

    Args:
        redis: An ``redis.asyncio.Redis`` client instance.
        prefix: Key prefix for namespacing.
    """

    def __init__(
        self,
        redis: Any,
        *,
        prefix: str = "afk:queue",
        retry_backoff_base_s: float = 0.0,
        retry_backoff_max_s: float = 30.0,
        retry_backoff_jitter_s: float = 0.0,
    ) -> None:
        super().__init__(
            retry_backoff_base_s=retry_backoff_base_s,
            retry_backoff_max_s=retry_backoff_max_s,
            retry_backoff_jitter_s=retry_backoff_jitter_s,
        )
        self._redis = redis
        self._prefix = prefix

    def _pending_key(self) -> str:
        """Redis list key storing pending task ids."""
        return f"{self._prefix}:pending"

    def _tasks_key(self) -> str:
        """Redis hash key storing serialized task records."""
        return f"{self._prefix}:tasks"

    def _inflight_key(self) -> str:
        """Redis list key storing in-flight task ids."""
        return f"{self._prefix}:inflight"

    def _workers_key(self) -> str:
        """Redis sorted-set key storing active worker ids with expiry scores."""
        return f"{self._prefix}:workers"

    def _recovery_lock_key(self) -> str:
        """Redis key for startup recovery lock coordination."""
        return f"{self._prefix}:recovery-lock"

    def _serialize(self, task: TaskItem) -> str:
        """Serialize one task item to JSON for Redis hash storage."""
        data = asdict(task)
        return json.dumps(data, default=str)

    def _deserialize(self, raw: str | bytes) -> TaskItem:
        """Deserialize one task item from JSON stored in Redis."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        return TaskItem(**data)

    async def _save_task(self, task: TaskItem) -> None:
        """Persist one task record in Redis hash storage."""
        await self._redis.hset(self._tasks_key(), task.id, self._serialize(task))

    async def _load_task(self, task_id: str) -> TaskItem | None:
        """Load one task record by id from Redis hash storage."""
        raw = await self._redis.hget(self._tasks_key(), task_id)
        if raw is None:
            return None
        return self._deserialize(raw)

    async def _delete_task(self, task_id: str) -> None:
        """Delete one task record by id from Redis hash storage."""
        await self._redis.hdel(self._tasks_key(), task_id)

    async def _push_pending_id(self, task_id: str) -> None:
        """Append one task id to the Redis pending list."""
        await self._redis.rpush(self._pending_key(), task_id)

    async def _pop_pending_id(self, *, timeout: float | None = None) -> str | None:
        """
        Pop one task id from Redis pending list into in-flight list.

        For sub-second timeouts, wraps ``BRPOPLPUSH`` with ``asyncio.wait_for`` so
        caller timeout semantics stay precise.
        """
        if timeout is None:
            result = await self._redis.brpoplpush(
                self._pending_key(),
                self._inflight_key(),
                timeout=0,
            )
        else:
            if timeout <= 0:
                return None
            redis_timeout = max(1, math.ceil(timeout))
            try:
                result = await asyncio.wait_for(
                    self._redis.brpoplpush(
                        self._pending_key(),
                        self._inflight_key(),
                        timeout=redis_timeout,
                    ),
                    timeout=timeout,
                )
            except TimeoutError:
                return None

        if result is None:
            return None

        if isinstance(result, bytes):
            return result.decode("utf-8")
        return str(result)

    async def _ack_inflight_id(self, task_id: str) -> None:
        """Remove one task id from in-flight tracking once lifecycle advances."""
        await self._redis.lrem(self._inflight_key(), 1, task_id)

    async def _cleanup_stale_workers(self) -> None:
        """Remove stale worker entries whose expiry is in the past."""
        now = time.time()
        await self._redis.zremrangebyscore(self._workers_key(), "-inf", now)

    async def register_worker(self, worker_id: str, *, ttl_s: float) -> None:
        """
        Register an active worker with TTL-backed presence.

        Args:
            worker_id: Unique worker identifier.
            ttl_s: Presence TTL in seconds; refreshed periodically by worker.
        """
        if ttl_s <= 0:
            raise ValueError("ttl_s must be > 0")
        expiry = time.time() + ttl_s
        await self._redis.zadd(self._workers_key(), {worker_id: expiry})
        await self._cleanup_stale_workers()

    async def refresh_worker(self, worker_id: str, *, ttl_s: float) -> None:
        """Refresh one worker presence TTL."""
        await self.register_worker(worker_id, ttl_s=ttl_s)

    async def unregister_worker(self, worker_id: str) -> None:
        """Remove one worker from presence tracking."""
        await self._redis.zrem(self._workers_key(), worker_id)

    async def active_worker_count(self) -> int:
        """
        Return count of currently active workers after stale-entry cleanup.
        """
        await self._cleanup_stale_workers()
        count = await self._redis.zcard(self._workers_key())
        return int(count)

    async def _acquire_recovery_lock(
        self,
        *,
        owner: str,
        ttl_s: float,
    ) -> str | None:
        """Try to acquire startup recovery lock; return lock token on success."""
        if ttl_s <= 0:
            raise ValueError("ttl_s must be > 0")
        token = f"{owner}:{time.time_ns()}"
        lock_ttl = max(1, math.ceil(ttl_s))
        acquired = await self._redis.set(
            self._recovery_lock_key(),
            token,
            nx=True,
            ex=lock_ttl,
        )
        if not acquired:
            return None
        return token

    async def _release_recovery_lock(self, *, token: str) -> None:
        """
        Best-effort release for startup recovery lock.

        Uses atomic compare-and-delete Lua script when available.
        """
        key = self._recovery_lock_key()
        eval_fn = getattr(self._redis, "eval", None)
        if callable(eval_fn):
            await eval_fn(_RELEASE_LOCK_IF_OWNER_SCRIPT, 1, key, token)
            return

        # Fallback for limited clients that do not expose EVAL.
        logger.warning(
            "Redis client has no eval(); recovery lock release is non-atomic"
        )
        current = await self._redis.get(key)
        if isinstance(current, bytes):
            current = current.decode("utf-8")
        if current == token:
            await self._redis.delete(key)

    async def recover_inflight_if_idle(
        self,
        *,
        active_worker_id: str,
        limit: int | None = None,
        lock_ttl_s: float = 15.0,
    ) -> int:
        """
        Requeue in-flight tasks only when this is the sole active worker.

        A short Redis lock coordinates startup recovery so only one worker
        attempts the recovery check and move operation at a time.
        """
        token = await self._acquire_recovery_lock(
            owner=active_worker_id,
            ttl_s=lock_ttl_s,
        )
        if token is None:
            return 0
        try:
            active_count = await self.active_worker_count()
            if active_count != 1:
                return 0
            moved = await self.requeue_inflight(limit=limit)
            if moved > 0:
                logger.info(
                    "Recovered %d in-flight task(s) on startup (worker_id=%s)",
                    moved,
                    active_worker_id[:8],
                )
            return moved
        finally:
            await self._release_recovery_lock(token=token)

    async def complete(self, task_id: str, *, result: JSONValue | None = None) -> None:
        """
        Complete a task and acknowledge it from in-flight tracking.

        The ack happens in ``finally`` to avoid leaking stale in-flight ids.
        """
        try:
            await super().complete(task_id, result=result)
        finally:
            await self._ack_inflight_id(task_id)

    async def fail(
        self,
        task_id: str,
        *,
        error: str,
        retryable: bool = True,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        """
        Fail/retry a task and acknowledge it from in-flight tracking.

        Retryable failures are re-enqueued by ``BaseTaskQueue.fail`` before ack.
        """
        try:
            await super().fail(
                task_id,
                error=error,
                retryable=retryable,
                retry_policy=retry_policy,
            )
        finally:
            await self._ack_inflight_id(task_id)

    async def cancel(self, task_id: str) -> None:
        """
        Cancel a task and acknowledge it from in-flight tracking.

        This keeps cancellation idempotent for tasks cancelled while running.
        """
        try:
            await super().cancel(task_id)
        finally:
            await self._ack_inflight_id(task_id)

    async def requeue_inflight(self, *, limit: int | None = None) -> int:
        """
        Move in-flight task ids back to pending queue for crash recovery.

        Args:
            limit: Optional maximum ids to move in one call.

        Returns:
            Number of moved task ids.
        """
        moved = 0
        while limit is None or moved < limit:
            task_id = await self._redis.rpoplpush(
                self._inflight_key(),
                self._pending_key(),
            )
            if task_id is None:
                break
            moved += 1
        return moved

    async def list_tasks(
        self,
        *,
        status: TaskStatus | None = None,
        limit: int = 100,
    ) -> list[TaskItem]:
        """List all tasks from Redis hash with optional status filter."""
        all_raw = await self._redis.hvals(self._tasks_key())
        tasks = [self._deserialize(r) for r in all_raw]
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        return tasks[:limit]
