from __future__ import annotations

import asyncio

import pytest

from afk.memory import LongTermMemory, MemoryEvent
from afk.memory.store import InMemoryMemoryStore


def run_async(coro):
    return asyncio.run(coro)


def make_event(event_id: str, thread_id: str, ts: int, text: str) -> MemoryEvent:
    return MemoryEvent(
        id=event_id,
        thread_id=thread_id,
        user_id="u1",
        type="message",
        timestamp=ts,
        payload={"text": text},
        tags=["chat"],
    )


def make_memory(
    mem_id: str,
    user_id: str | None,
    scope: str,
    text: str,
    updated_at: int,
    *,
    tags: list[str] | None = None,
) -> LongTermMemory:
    return LongTermMemory(
        id=mem_id,
        user_id=user_id,
        scope=scope,
        data={"text": text, "kind": "note"},
        text=text,
        tags=tags or [],
        metadata={"source": "test"},
        created_at=updated_at - 100,
        updated_at=updated_at,
    )


def test_requires_setup_before_use():
    store = InMemoryMemoryStore()
    with pytest.raises(RuntimeError, match="not initialized"):
        run_async(store.get_recent_events("thread-1"))


def test_context_manager_runs_setup_and_close():
    store = InMemoryMemoryStore()

    async def scenario():
        async with store:
            await store.put_state("thread-1", "k", "v")
            assert await store.get_state("thread-1", "k") == "v"

    run_async(scenario())

    with pytest.raises(RuntimeError, match="not initialized"):
        run_async(store.get_state("thread-1", "k"))


def test_event_append_recent_and_since_with_limits():
    store = InMemoryMemoryStore()

    async def scenario():
        await store.setup()
        await store.append_event(make_event("e1", "t1", 1000, "one"))
        await store.append_event(make_event("e2", "t1", 2000, "two"))
        await store.append_event(make_event("e3", "t1", 3000, "three"))
        await store.append_event(make_event("e4", "t2", 4000, "other-thread"))

        recent = await store.get_recent_events("t1", limit=2)
        assert [event.id for event in recent] == ["e2", "e3"]

        since = await store.get_events_since("t1", since_ms=2000, limit=1)
        assert [event.id for event in since] == ["e2"]

        await store.close()

    run_async(scenario())


def test_state_put_get_and_prefix_listing_sorted():
    store = InMemoryMemoryStore()

    async def scenario():
        await store.setup()
        await store.put_state("thread-1", "z_last", 3)
        await store.put_state("thread-1", "a_first", 1)
        await store.put_state("thread-1", "a_second", 2)
        await store.put_state("thread-2", "a_first", "other")

        assert await store.get_state("thread-1", "a_first") == 1
        assert await store.get_state("thread-1", "missing") is None

        listed = await store.list_state("thread-1")
        assert list(listed.keys()) == ["a_first", "a_second", "z_last"]

        filtered = await store.list_state("thread-1", prefix="a_")
        assert filtered == {"a_first": 1, "a_second": 2}

        await store.close()

    run_async(scenario())


def test_long_term_memory_list_text_search_vector_search_and_delete():
    store = InMemoryMemoryStore()

    async def scenario():
        await store.setup()

        mem1 = make_memory(
            "m1", "u1", "global", "Python unit testing", 1000, tags=["tests"]
        )
        mem2 = make_memory(
            "m2", "u1", "project:a", "Fast sqlite storage", 2000, tags=["db"]
        )
        mem3 = make_memory(
            "m3", "u2", "global", "Different user memory", 3000, tags=["user"]
        )

        await store.upsert_long_term_memory(mem1, embedding=[1.0, 0.0])
        await store.upsert_long_term_memory(mem2, embedding=[0.7, 0.7])
        await store.upsert_long_term_memory(mem3, embedding=[0.0, 1.0])

        listed_all = await store.list_long_term_memories("u1")
        assert [m.id for m in listed_all] == ["m2", "m1"]

        listed_scope = await store.list_long_term_memories("u1", scope="project:a")
        assert [m.id for m in listed_scope] == ["m2"]

        text_hits = await store.search_long_term_memory_text("u1", "sqlite")
        assert [m.id for m in text_hits] == ["m2"]
        assert await store.search_long_term_memory_text("u1", "   ") == []

        vec_hits = await store.search_long_term_memory_vector(
            "u1", [1.0, 0.0], min_score=0.8
        )
        assert [m.id for m, _ in vec_hits] == ["m1"]

        # Deleting with mismatched user id must not remove the record.
        await store.delete_long_term_memory("u2", "m1")
        still_there = await store.list_long_term_memories("u1")
        assert {m.id for m in still_there} == {"m1", "m2"}

        await store.delete_long_term_memory("u1", "m1")
        after_delete = await store.list_long_term_memories("u1")
        assert [m.id for m in after_delete] == ["m2"]

        await store.close()

    run_async(scenario())


def test_upsert_without_embedding_preserves_existing_embedding():
    store = InMemoryMemoryStore()

    async def scenario():
        await store.setup()
        original = make_memory("m1", "u1", "global", "Original", 1000)
        await store.upsert_long_term_memory(original, embedding=[1.0, 0.0])

        updated = make_memory("m1", "u1", "global", "Updated", 2000)
        await store.upsert_long_term_memory(updated, embedding=None)

        vec_hits = await store.search_long_term_memory_vector("u1", [1.0, 0.0], limit=5)
        assert [m.id for m, _ in vec_hits] == ["m1"]
        assert vec_hits[0][0].text == "Updated"

        await store.close()

    run_async(scenario())


def test_vector_search_ignores_memories_without_embedding_and_applies_scope():
    store = InMemoryMemoryStore()

    async def scenario():
        await store.setup()
        mem1 = make_memory("m1", "u1", "scope:a", "has embedding", 1000)
        mem2 = make_memory("m2", "u1", "scope:b", "no embedding", 2000)
        await store.upsert_long_term_memory(mem1, embedding=[0.5, 0.5])
        await store.upsert_long_term_memory(mem2)

        scoped = await store.search_long_term_memory_vector(
            "u1", [0.5, 0.5], scope="scope:a"
        )
        assert [m.id for m, _ in scoped] == ["m1"]

        out_of_scope = await store.search_long_term_memory_vector(
            "u1", [0.5, 0.5], scope="scope:b"
        )
        assert out_of_scope == []

        await store.close()

    run_async(scenario())
