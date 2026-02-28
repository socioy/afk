from __future__ import annotations

import asyncio

import pytest

from afk.queues import (
    DEAD_LETTER_REASON_KEY,
    EXECUTION_CONTRACT_KEY,
    JOB_DISPATCH_CONTRACT,
    NEXT_ATTEMPT_AT_KEY,
    RUNNER_CHAT_CONTRACT,
    ExecutionContractContext,
    InMemoryTaskQueue,
    RedisTaskQueue,
    RetryPolicy,
    TaskItem,
    TaskWorker,
    TaskWorkerConfig,
    create_task_queue_from_env,
)


def run_async(coro):
    return asyncio.run(coro)


def test_in_memory_retry_allows_max_retries():
    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        task = await queue.enqueue_contract(
            RUNNER_CHAT_CONTRACT,
            payload={"user_message": "hello", "context": {}},
            agent_name="demo",
            max_retries=2,
        )

        first = await queue.dequeue(timeout=0.1)
        assert first is not None
        await queue.fail(task.id, error="boom-1")
        current = await queue.get(task.id)
        assert current is not None
        assert current.status == "retrying"
        assert current.retry_count == 1

        second = await queue.dequeue(timeout=0.1)
        assert second is not None
        await queue.fail(task.id, error="boom-2")
        current = await queue.get(task.id)
        assert current is not None
        assert current.status == "retrying"
        assert current.retry_count == 2

        third = await queue.dequeue(timeout=0.1)
        assert third is not None
        await queue.fail(task.id, error="boom-3")
        current = await queue.get(task.id)
        assert current is not None
        assert current.status == "failed"
        assert current.retry_count == 3

    run_async(scenario())


def test_enqueue_contract_writes_explicit_contract_metadata():
    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        task = await queue.enqueue_contract(
            JOB_DISPATCH_CONTRACT,
            payload={"job_type": "noop"},
            agent_name=None,
            metadata={"source": "tests"},
        )
        assert task.metadata[EXECUTION_CONTRACT_KEY] == JOB_DISPATCH_CONTRACT
        assert task.execution_contract == JOB_DISPATCH_CONTRACT
        assert task.agent_name is None

    run_async(scenario())


def test_enqueue_contract_rejects_empty_contract():
    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        with pytest.raises(ValueError, match="non-empty"):
            await queue.enqueue_contract(
                "   ",
                payload={},
                agent_name=None,
            )

    run_async(scenario())


def test_queue_factory_defaults_to_in_memory(monkeypatch):
    monkeypatch.delenv("AFK_QUEUE_BACKEND", raising=False)
    queue = create_task_queue_from_env()
    assert isinstance(queue, InMemoryTaskQueue)


def test_queue_factory_redis_with_injected_client(monkeypatch):
    monkeypatch.setenv("AFK_QUEUE_BACKEND", "redis")
    monkeypatch.setenv("AFK_QUEUE_REDIS_PREFIX", "tests:queue")
    injected = object()

    queue = create_task_queue_from_env(redis_client=injected)

    assert isinstance(queue, RedisTaskQueue)
    assert queue._redis is injected  # noqa: SLF001
    assert queue._prefix == "tests:queue"  # noqa: SLF001


def test_queue_factory_invalid_backend_raises(monkeypatch):
    monkeypatch.setenv("AFK_QUEUE_BACKEND", "bad-backend")
    with pytest.raises(ValueError, match="Unknown AFK_QUEUE_BACKEND"):
        create_task_queue_from_env()


def test_redis_queue_subsecond_timeout_does_not_become_infinite():
    class _FakeRedis:
        def __init__(self) -> None:
            self.timeouts: list[int] = []

        async def brpoplpush(self, source: str, destination: str, timeout: int):
            _ = source
            _ = destination
            self.timeouts.append(timeout)
            return None

    async def scenario() -> None:
        fake = _FakeRedis()
        queue = RedisTaskQueue(fake)
        out = await queue.dequeue(timeout=0.25)
        assert out is None
        assert fake.timeouts == [1]

    run_async(scenario())


def test_terminal_state_is_immutable_for_complete_and_fail():
    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        task = await queue.enqueue_contract(
            RUNNER_CHAT_CONTRACT,
            payload={"user_message": "hello", "context": {}},
            agent_name="demo",
        )

        running = await queue.dequeue(timeout=0.1)
        assert running is not None
        await queue.complete(task.id, result={"ok": True})

        # Terminal completed task should ignore subsequent fail().
        await queue.fail(task.id, error="late failure", retryable=False)
        current = await queue.get(task.id)
        assert current is not None
        assert current.status == "completed"
        assert current.retry_count == 0
        assert current.result == {"ok": True}
        assert current.error is None

        task2 = await queue.enqueue_contract(
            RUNNER_CHAT_CONTRACT,
            payload={"user_message": "hello", "context": {}},
            agent_name="demo",
        )
        running2 = await queue.dequeue(timeout=0.1)
        assert running2 is not None
        await queue.cancel(task2.id)

        # Terminal cancelled task should ignore subsequent complete().
        await queue.complete(task2.id, result={"should": "not-apply"})
        current2 = await queue.get(task2.id)
        assert current2 is not None
        assert current2.status == "cancelled"
        assert current2.result is None

    run_async(scenario())


def test_retryable_false_skips_requeue_even_when_budget_remains():
    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        task = await queue.enqueue_contract(
            RUNNER_CHAT_CONTRACT,
            payload={"user_message": "hello", "context": {}},
            agent_name="demo",
            max_retries=5,
        )
        running = await queue.dequeue(timeout=0.1)
        assert running is not None

        await queue.fail(task.id, error="schema error", retryable=False)
        current = await queue.get(task.id)
        assert current is not None
        assert current.status == "failed"
        assert current.retry_count == 1
        assert queue.pending_count == 0

    run_async(scenario())


def test_retry_backoff_defers_requeued_task_until_next_attempt():
    async def scenario() -> None:
        queue = InMemoryTaskQueue(
            retry_backoff_base_s=0.2,
            retry_backoff_max_s=1.0,
            retry_backoff_jitter_s=0.0,
        )
        task = await queue.enqueue_contract(
            RUNNER_CHAT_CONTRACT,
            payload={"user_message": "hello", "context": {}},
            agent_name="demo",
            max_retries=2,
        )
        running = await queue.dequeue(timeout=0.1)
        assert running is not None
        await queue.fail(task.id, error="boom")

        item = await queue.get(task.id)
        assert item is not None
        assert item.status == "retrying"
        assert item.metadata.get(NEXT_ATTEMPT_AT_KEY) is not None

        # Not ready yet; short dequeue should time out.
        blocked = await queue.dequeue(timeout=0.05)
        assert blocked is None

        # Ready after backoff period.
        await asyncio.sleep(0.2)
        next_item = await queue.dequeue(timeout=0.1)
        assert next_item is not None
        assert next_item.id == task.id
        assert next_item.metadata.get(NEXT_ATTEMPT_AT_KEY) is None

    run_async(scenario())


def test_failed_tasks_include_dead_letter_reason_metadata():
    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        task_a = await queue.enqueue_contract(
            RUNNER_CHAT_CONTRACT,
            payload={"user_message": "x", "context": {}},
            agent_name="demo",
            max_retries=0,
        )
        task_b = await queue.enqueue_contract(
            RUNNER_CHAT_CONTRACT,
            payload={"user_message": "y", "context": {}},
            agent_name="demo",
            max_retries=1,
        )

        running_a = await queue.dequeue(timeout=0.1)
        assert running_a is not None
        await queue.fail(task_a.id, error="schema", retryable=False)

        running_b = await queue.dequeue(timeout=0.1)
        assert running_b is not None
        await queue.fail(task_b.id, error="boom-1", retryable=True)
        running_b2 = await queue.dequeue(timeout=0.1)
        assert running_b2 is not None
        await queue.fail(task_b.id, error="boom-2", retryable=True)

        item_a = await queue.get(task_a.id)
        item_b = await queue.get(task_b.id)
        assert item_a is not None and item_b is not None
        assert item_a.metadata.get(DEAD_LETTER_REASON_KEY) == "non_retryable_error"
        assert item_b.metadata.get(DEAD_LETTER_REASON_KEY) == "retry_budget_exhausted"

        dlq = await queue.list_dead_letters(limit=10)
        assert {t.id for t in dlq} >= {task_a.id, task_b.id}

    run_async(scenario())


def test_dead_letter_redrive_and_purge_operations():
    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        t1 = await queue.enqueue_contract(
            RUNNER_CHAT_CONTRACT,
            payload={"user_message": "a", "context": {}},
            agent_name="demo",
            max_retries=0,
        )
        t2 = await queue.enqueue_contract(
            RUNNER_CHAT_CONTRACT,
            payload={"user_message": "b", "context": {}},
            agent_name="demo",
            max_retries=0,
        )
        r1 = await queue.dequeue(timeout=0.1)
        r2 = await queue.dequeue(timeout=0.1)
        assert r1 is not None and r2 is not None
        await queue.fail(t1.id, error="x", retryable=False)
        await queue.fail(t2.id, error="y", retryable=False)

        moved = await queue.redrive_dead_letters(limit=10, reason="non_retryable_error")
        assert moved == 2
        d1 = await queue.dequeue(timeout=0.1)
        d2 = await queue.dequeue(timeout=0.1)
        assert d1 is not None and d2 is not None
        assert {d1.id, d2.id} == {t1.id, t2.id}

        await queue.fail(t1.id, error="x2", retryable=False)
        await queue.fail(t2.id, error="y2", retryable=False)
        purged = await queue.purge_dead_letters(limit=10, reason="non_retryable_error")
        assert purged == 2
        assert await queue.get(t1.id) is None
        assert await queue.get(t2.id) is None

    run_async(scenario())


def test_per_task_retry_policy_override_is_persisted_and_used():
    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        policy = RetryPolicy(
            backoff_base_s=0.3, backoff_max_s=1.0, backoff_jitter_s=0.0
        )
        task = await queue.enqueue_contract(
            RUNNER_CHAT_CONTRACT,
            payload={"user_message": "hello", "context": {}},
            agent_name="demo",
            max_retries=1,
            retry_policy=policy,
        )
        running = await queue.dequeue(timeout=0.1)
        assert running is not None
        await queue.fail(task.id, error="boom", retryable=True)
        item = await queue.get(task.id)
        assert item is not None
        assert item.next_attempt_at is not None
        # Delay should be at least about base backoff.
        assert item.next_attempt_at - item.created_at >= 0.25

    run_async(scenario())


def test_queue_backends_preserve_optional_agent_and_contract_metadata():
    class _FakeRedis:
        def __init__(self) -> None:
            self._hashes: dict[str, dict[str, str]] = {}
            self._lists: dict[str, list[str]] = {}

        async def hset(self, key: str, field: str, value: str):
            self._hashes.setdefault(key, {})[field] = value

        async def hget(self, key: str, field: str):
            return self._hashes.get(key, {}).get(field)

        async def hvals(self, key: str):
            return list(self._hashes.get(key, {}).values())

        async def rpush(self, key: str, value: str):
            self._lists.setdefault(key, []).append(value)

        async def blpop(self, key: str, timeout: int):
            _ = timeout
            bucket = self._lists.get(key, [])
            if not bucket:
                return None
            value = bucket.pop(0)
            return (key, value)

    async def scenario() -> None:
        mem = InMemoryTaskQueue()
        mem_task = await mem.enqueue_contract(
            JOB_DISPATCH_CONTRACT,
            payload={"job_type": "noop"},
            agent_name=None,
            metadata={"scope": "mem"},
        )
        mem_item = await mem.get(mem_task.id)
        assert mem_item is not None
        assert mem_item.agent_name is None
        assert mem_item.execution_contract == JOB_DISPATCH_CONTRACT

        redis = RedisTaskQueue(_FakeRedis())
        redis_task = await redis.enqueue_contract(
            JOB_DISPATCH_CONTRACT,
            payload={"job_type": "noop"},
            agent_name=None,
            metadata={"scope": "redis"},
        )
        redis_item = await redis.get(redis_task.id)
        assert redis_item is not None
        assert redis_item.agent_name is None
        assert redis_item.execution_contract == JOB_DISPATCH_CONTRACT

    run_async(scenario())


def test_redis_queue_uses_inflight_and_acks_on_lifecycle_progress():
    class _FakeRedis:
        def __init__(self) -> None:
            self._hashes: dict[str, dict[str, str]] = {}
            self._lists: dict[str, list[str]] = {}

        async def hset(self, key: str, field: str, value: str):
            self._hashes.setdefault(key, {})[field] = value

        async def hget(self, key: str, field: str):
            return self._hashes.get(key, {}).get(field)

        async def hvals(self, key: str):
            return list(self._hashes.get(key, {}).values())

        async def rpush(self, key: str, value: str):
            self._lists.setdefault(key, []).append(value)

        async def brpoplpush(self, source: str, destination: str, timeout: int):
            _ = timeout
            src = self._lists.setdefault(source, [])
            if not src:
                return None
            value = src.pop()
            self._lists.setdefault(destination, []).insert(0, value)
            return value

        async def lrem(self, key: str, count: int, value: str):
            bucket = self._lists.setdefault(key, [])
            removed = 0
            i = 0
            while i < len(bucket):
                if bucket[i] == value and (count == 0 or removed < count):
                    bucket.pop(i)
                    removed += 1
                    continue
                i += 1
            return removed

        async def rpoplpush(self, source: str, destination: str):
            src = self._lists.setdefault(source, [])
            if not src:
                return None
            value = src.pop()
            self._lists.setdefault(destination, []).insert(0, value)
            return value

    async def scenario() -> None:
        fake = _FakeRedis()
        queue = RedisTaskQueue(fake)
        task = await queue.enqueue_contract(
            JOB_DISPATCH_CONTRACT,
            payload={"job_type": "noop"},
            agent_name=None,
        )

        pending_key = queue._pending_key()  # noqa: SLF001
        inflight_key = queue._inflight_key()  # noqa: SLF001
        assert fake._lists[pending_key] == [task.id]
        assert fake._lists.get(inflight_key, []) == []

        running = await queue.dequeue(timeout=0.1)
        assert running is not None
        assert fake._lists[pending_key] == []
        assert fake._lists[inflight_key] == [task.id]

        await queue.complete(task.id, result={"ok": True})
        assert fake._lists[inflight_key] == []

    run_async(scenario())


def test_redis_queue_can_requeue_inflight_for_recovery():
    class _FakeRedis:
        def __init__(self) -> None:
            self._hashes: dict[str, dict[str, str]] = {}
            self._lists: dict[str, list[str]] = {}

        async def hset(self, key: str, field: str, value: str):
            self._hashes.setdefault(key, {})[field] = value

        async def hget(self, key: str, field: str):
            return self._hashes.get(key, {}).get(field)

        async def hvals(self, key: str):
            return list(self._hashes.get(key, {}).values())

        async def rpush(self, key: str, value: str):
            self._lists.setdefault(key, []).append(value)

        async def brpoplpush(self, source: str, destination: str, timeout: int):
            _ = timeout
            src = self._lists.setdefault(source, [])
            if not src:
                return None
            value = src.pop()
            self._lists.setdefault(destination, []).insert(0, value)
            return value

        async def lrem(self, key: str, count: int, value: str):
            bucket = self._lists.setdefault(key, [])
            removed = 0
            i = 0
            while i < len(bucket):
                if bucket[i] == value and (count == 0 or removed < count):
                    bucket.pop(i)
                    removed += 1
                    continue
                i += 1
            return removed

        async def rpoplpush(self, source: str, destination: str):
            src = self._lists.setdefault(source, [])
            if not src:
                return None
            value = src.pop()
            self._lists.setdefault(destination, []).insert(0, value)
            return value

    async def scenario() -> None:
        fake = _FakeRedis()
        queue = RedisTaskQueue(fake)
        task = await queue.enqueue_contract(
            JOB_DISPATCH_CONTRACT,
            payload={"job_type": "noop"},
            agent_name=None,
        )

        pending_key = queue._pending_key()  # noqa: SLF001
        inflight_key = queue._inflight_key()  # noqa: SLF001
        running = await queue.dequeue(timeout=0.1)
        assert running is not None
        assert fake._lists[pending_key] == []
        assert fake._lists[inflight_key] == [task.id]

        moved = await queue.requeue_inflight()
        assert moved == 1
        assert fake._lists[inflight_key] == []
        assert fake._lists[pending_key] == [task.id]

    run_async(scenario())


def test_redis_release_recovery_lock_is_atomic_compare_delete():
    class _FakeRedis:
        def __init__(self) -> None:
            self._kv: dict[str, str] = {}

        async def eval(self, script: str, numkeys: int, key: str, token: str):
            _ = script
            assert numkeys == 1
            current = self._kv.get(key)
            if current == token:
                self._kv.pop(key, None)
                return 1
            return 0

    async def scenario() -> None:
        fake = _FakeRedis()
        queue = RedisTaskQueue(fake)
        lock_key = queue._recovery_lock_key()  # noqa: SLF001

        fake._kv[lock_key] = "token-new"
        await queue._release_recovery_lock(token="token-old")  # noqa: SLF001
        assert fake._kv[lock_key] == "token-new"

        fake._kv[lock_key] = "token-own"
        await queue._release_recovery_lock(token="token-own")  # noqa: SLF001
        assert lock_key not in fake._kv

    run_async(scenario())


def test_redis_recover_inflight_if_idle_requires_single_active_worker():
    class _FakeRedis:
        def __init__(self) -> None:
            self._hashes: dict[str, dict[str, str]] = {}
            self._lists: dict[str, list[str]] = {}
            self._zsets: dict[str, dict[str, float]] = {}
            self._kv: dict[str, str] = {}

        async def hset(self, key: str, field: str, value: str):
            self._hashes.setdefault(key, {})[field] = value

        async def hget(self, key: str, field: str):
            return self._hashes.get(key, {}).get(field)

        async def hvals(self, key: str):
            return list(self._hashes.get(key, {}).values())

        async def rpush(self, key: str, value: str):
            self._lists.setdefault(key, []).append(value)

        async def brpoplpush(self, source: str, destination: str, timeout: int):
            _ = timeout
            src = self._lists.setdefault(source, [])
            if not src:
                return None
            value = src.pop()
            self._lists.setdefault(destination, []).insert(0, value)
            return value

        async def lrem(self, key: str, count: int, value: str):
            bucket = self._lists.setdefault(key, [])
            removed = 0
            i = 0
            while i < len(bucket):
                if bucket[i] == value and (count == 0 or removed < count):
                    bucket.pop(i)
                    removed += 1
                    continue
                i += 1
            return removed

        async def rpoplpush(self, source: str, destination: str):
            src = self._lists.setdefault(source, [])
            if not src:
                return None
            value = src.pop()
            self._lists.setdefault(destination, []).insert(0, value)
            return value

        async def zadd(self, key: str, mapping: dict[str, float]):
            zset = self._zsets.setdefault(key, {})
            zset.update(mapping)

        async def zremrangebyscore(self, key: str, min_score, max_score):
            _ = min_score
            zset = self._zsets.setdefault(key, {})
            to_remove = [k for k, v in zset.items() if v <= float(max_score)]
            for member in to_remove:
                zset.pop(member, None)
            return len(to_remove)

        async def zrem(self, key: str, member: str):
            zset = self._zsets.setdefault(key, {})
            existed = member in zset
            zset.pop(member, None)
            return 1 if existed else 0

        async def zcard(self, key: str):
            return len(self._zsets.setdefault(key, {}))

        async def set(
            self,
            key: str,
            value: str,
            *,
            nx: bool = False,
            ex: int | None = None,
        ):
            _ = ex
            if nx and key in self._kv:
                return False
            self._kv[key] = value
            return True

        async def get(self, key: str):
            return self._kv.get(key)

        async def delete(self, key: str):
            existed = key in self._kv
            self._kv.pop(key, None)
            return 1 if existed else 0

    async def scenario() -> None:
        fake = _FakeRedis()
        queue = RedisTaskQueue(fake)

        # Seed one in-flight id.
        await queue.enqueue_contract(
            JOB_DISPATCH_CONTRACT,
            payload={"job_type": "noop"},
            agent_name=None,
        )
        task = await queue.dequeue(timeout=0.1)
        assert task is not None
        pending_key = queue._pending_key()  # noqa: SLF001
        inflight_key = queue._inflight_key()  # noqa: SLF001
        assert fake._lists[pending_key] == []
        assert len(fake._lists[inflight_key]) == 1

        await queue.register_worker("w1", ttl_s=30)
        await queue.register_worker("w2", ttl_s=30)
        moved = await queue.recover_inflight_if_idle(active_worker_id="w1")
        assert moved == 0
        assert len(fake._lists[inflight_key]) == 1

        await queue.unregister_worker("w2")
        moved2 = await queue.recover_inflight_if_idle(active_worker_id="w1")
        assert moved2 == 1
        assert fake._lists[inflight_key] == []
        assert len(fake._lists[pending_key]) == 1

    run_async(scenario())


def test_task_worker_startup_recovery_runs_for_supported_queue():
    class _RecoveryQueue(InMemoryTaskQueue):
        def __init__(self) -> None:
            super().__init__()
            self.registered: list[str] = []
            self.unregistered: list[str] = []
            self.refresh_calls = 0
            self.recovery_calls = 0

        async def register_worker(self, worker_id: str, *, ttl_s: float) -> None:
            _ = ttl_s
            self.registered.append(worker_id)

        async def refresh_worker(self, worker_id: str, *, ttl_s: float) -> None:
            _ = worker_id
            _ = ttl_s
            self.refresh_calls += 1

        async def unregister_worker(self, worker_id: str) -> None:
            self.unregistered.append(worker_id)

        async def recover_inflight_if_idle(self, *, active_worker_id: str) -> int:
            _ = active_worker_id
            self.recovery_calls += 1
            return 0

    async def scenario() -> None:
        queue = _RecoveryQueue()
        worker = TaskWorker(
            queue,
            agents={},
            config=TaskWorkerConfig(
                poll_interval_s=0.01,
                worker_presence_refresh_s=0.01,
                worker_presence_ttl_s=1.0,
            ),
        )
        await worker.start()
        deadline = asyncio.get_running_loop().time() + 0.5
        while queue.refresh_calls == 0 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        await worker.shutdown()

        assert len(queue.registered) == 1
        assert queue.recovery_calls == 1
        assert len(queue.unregistered) == 1
        assert queue.refresh_calls >= 1

    run_async(scenario())


def test_task_worker_presence_refresh_must_be_less_than_ttl():
    class _PresenceQueue(InMemoryTaskQueue):
        async def register_worker(self, worker_id: str, *, ttl_s: float) -> None:
            _ = worker_id
            _ = ttl_s

        async def refresh_worker(self, worker_id: str, *, ttl_s: float) -> None:
            _ = worker_id
            _ = ttl_s

        async def unregister_worker(self, worker_id: str) -> None:
            _ = worker_id

    async def scenario() -> None:
        queue = _PresenceQueue()
        worker = TaskWorker(
            queue,
            agents={},
            config=TaskWorkerConfig(
                worker_presence_ttl_s=1.0,
                worker_presence_refresh_s=1.0,
            ),
        )
        with pytest.raises(ValueError, match="less than"):
            await worker.start()

    run_async(scenario())


def test_task_worker_rejects_contract_id_mismatch():
    queue = InMemoryTaskQueue()

    class _MismatchContract:
        contract_id = "other.v1"
        requires_agent = False

        async def execute(
            self,
            task_item: TaskItem,
            *,
            agent,
            worker_context: ExecutionContractContext,
        ):
            _ = task_item
            _ = agent
            _ = worker_context
            return {"ok": True}

    with pytest.raises(ValueError, match="Contract id mismatch"):
        TaskWorker(
            queue,
            agents={"demo": object()},
            execution_contracts={"mismatch.v1": _MismatchContract()},
        )


def test_task_worker_shutdown_does_not_leak_semaphore_permits():
    class _BlockingContract:
        contract_id = "blocking.v1"
        requires_agent = False

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def execute(
            self,
            task_item: TaskItem,
            *,
            agent,
            worker_context: ExecutionContractContext,
        ):
            _ = task_item
            _ = agent
            _ = worker_context
            self.started.set()
            await self.release.wait()
            return {"ok": True}

    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        contract = _BlockingContract()
        worker = TaskWorker(
            queue,
            agents={"demo": object()},
            execution_contracts={"blocking.v1": contract},
            config=TaskWorkerConfig(
                max_concurrent_tasks=1,
                poll_interval_s=0.01,
                shutdown_timeout_s=1.0,
            ),
        )

        await worker.start()
        await queue.enqueue_contract(
            "blocking.v1",
            payload={},
            agent_name=None,
        )
        await asyncio.wait_for(contract.started.wait(), timeout=1.0)

        # Give the loop a chance to hit semaphore acquisition for the next cycle.
        await asyncio.sleep(0.05)

        shutdown_task = asyncio.create_task(worker.shutdown())
        await asyncio.sleep(0.05)
        contract.release.set()
        await shutdown_task

        assert worker.active_task_count == 0
        assert worker._semaphore._value == 1  # noqa: SLF001

    run_async(scenario())
