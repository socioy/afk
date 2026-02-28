from __future__ import annotations

import asyncio
import os
import time
import uuid

import pytest

from afk.queues import JOB_DISPATCH_CONTRACT, RedisTaskQueue


def _redis_url() -> str | None:
    return os.getenv("AFK_TEST_REDIS_URL")


@pytest.mark.skipif(_redis_url() is None, reason="AFK_TEST_REDIS_URL is not set")
def test_redis_recovery_lock_and_idle_guard_with_real_redis():
    async def _scenario() -> None:
        redis = pytest.importorskip("redis.asyncio")
        client = redis.Redis.from_url(_redis_url())
        prefix = f"itest:queue:{uuid.uuid4().hex}"
        queue = RedisTaskQueue(client, prefix=prefix)

        # Seed in-flight task.
        task = await queue.enqueue_contract(
            JOB_DISPATCH_CONTRACT,
            payload={"job_type": "noop"},
            agent_name=None,
        )
        running = await queue.dequeue(timeout=0.5)
        assert running is not None
        assert running.id == task.id

        # Two active workers => no recovery.
        await queue.register_worker("w1", ttl_s=30)
        await queue.register_worker("w2", ttl_s=30)
        moved = await queue.recover_inflight_if_idle(active_worker_id="w1")
        assert moved == 0

        # One active worker => recovery allowed.
        await queue.unregister_worker("w2")
        moved2 = await queue.recover_inflight_if_idle(active_worker_id="w1")
        assert moved2 == 1

        # Recovering puts task back to pending.
        replay = await queue.dequeue(timeout=0.5)
        assert replay is not None
        assert replay.id == task.id

        await queue.fail(task.id, error="done", retryable=False)
        await queue.unregister_worker("w1")
        await client.aclose()

    asyncio.run(_scenario())


@pytest.mark.skipif(_redis_url() is None, reason="AFK_TEST_REDIS_URL is not set")
def test_redis_retry_backoff_with_real_redis():
    async def _scenario() -> None:
        redis = pytest.importorskip("redis.asyncio")
        client = redis.Redis.from_url(_redis_url())
        prefix = f"itest:queue:{uuid.uuid4().hex}"
        queue = RedisTaskQueue(
            client,
            prefix=prefix,
            retry_backoff_base_s=0.2,
            retry_backoff_max_s=0.2,
            retry_backoff_jitter_s=0.0,
        )

        task = await queue.enqueue_contract(
            JOB_DISPATCH_CONTRACT,
            payload={"job_type": "noop"},
            agent_name=None,
            max_retries=1,
        )
        running = await queue.dequeue(timeout=0.5)
        assert running is not None
        await queue.fail(task.id, error="boom", retryable=True)

        # Immediate dequeue should typically timeout due to backoff.
        start = time.time()
        blocked = await queue.dequeue(timeout=0.05)
        assert blocked is None
        assert time.time() - start >= 0.04

        # Becomes available after backoff delay.
        replay = await queue.dequeue(timeout=0.5)
        assert replay is not None
        assert replay.id == task.id

        await queue.fail(task.id, error="done", retryable=False)
        await client.aclose()

    asyncio.run(_scenario())
