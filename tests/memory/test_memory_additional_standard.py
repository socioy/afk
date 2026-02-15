from __future__ import annotations

import asyncio

import pytest

import afk.memory as memory_api
import afk.memory.store as memory_store_api
from afk.memory.models import LongTermMemory, MemoryEvent
from afk.memory.store import InMemoryMemoryStore, MemoryStore, SQLiteMemoryStore


def run_async(coro):
    return asyncio.run(coro)


def test_memory_module_getattr_errors_for_unknown_attr():
    with pytest.raises(AttributeError):
        getattr(memory_api, "DoesNotExist")

    with pytest.raises(AttributeError):
        getattr(memory_store_api, "DoesNotExist")


def test_memory_store_capabilities_are_exposed():
    in_mem = InMemoryMemoryStore()
    sqlite = SQLiteMemoryStore(":memory:")

    assert in_mem.capabilities.text_search is True
    assert in_mem.capabilities.vector_search is True
    assert sqlite.capabilities.atomic_upsert is True


def test_memory_store_base_context_manager_with_minimal_impl():
    class MinimalStore(MemoryStore):
        async def append_event(self, event: MemoryEvent) -> None:
            self._ensure_setup()

        async def get_recent_events(self, thread_id: str, limit: int = 50):
            self._ensure_setup()
            return []

        async def get_events_since(
            self, thread_id: str, since_ms: int, limit: int = 500
        ):
            self._ensure_setup()
            return []

        async def put_state(self, thread_id: str, key: str, value):
            self._ensure_setup()

        async def get_state(self, thread_id: str, key: str):
            self._ensure_setup()
            return None

        async def list_state(self, thread_id: str, prefix: str | None = None):
            self._ensure_setup()
            return {}

        async def upsert_long_term_memory(
            self, memory: LongTermMemory, *, embedding=None
        ):
            self._ensure_setup()

        async def delete_long_term_memory(self, user_id: str | None, memory_id: str):
            self._ensure_setup()

        async def list_long_term_memories(
            self, user_id: str | None, *, scope=None, limit=100
        ):
            self._ensure_setup()
            return []

        async def search_long_term_memory_text(
            self, user_id: str | None, query: str, *, scope=None, limit=20
        ):
            self._ensure_setup()
            return []

        async def search_long_term_memory_vector(
            self,
            user_id: str | None,
            query_embedding,
            *,
            scope=None,
            limit=20,
            min_score=None,
        ):
            self._ensure_setup()
            return []

    store = MinimalStore()

    with pytest.raises(RuntimeError, match="not initialized"):
        run_async(store.get_recent_events("thread"))

    async def scenario():
        async with store:
            assert await store.get_recent_events("thread") == []
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.get_recent_events("thread")

    run_async(scenario())
