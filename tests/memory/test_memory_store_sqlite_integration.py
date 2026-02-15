from __future__ import annotations

import asyncio

import pytest

from afk.memory import LongTermMemory, MemoryEvent
from afk.memory.store import SQLiteMemoryStore


def run_async(coro):
    return asyncio.run(coro)


def make_event(event_id: str, thread_id: str, ts: int, kind: str) -> MemoryEvent:
    return MemoryEvent(
        id=event_id,
        thread_id=thread_id,
        user_id="u1",
        type="message",
        timestamp=ts,
        payload={"kind": kind},
        tags=["t"],
    )


def make_memory(
    mem_id: str,
    user_id: str | None,
    scope: str,
    text: str,
    updated_at: int,
    *,
    created_at: int | None = None,
) -> LongTermMemory:
    return LongTermMemory(
        id=mem_id,
        user_id=user_id,
        scope=scope,
        data={"body": text},
        text=text,
        tags=["tag"],
        metadata={"source": "sqlite-test"},
        created_at=created_at if created_at is not None else updated_at - 10,
        updated_at=updated_at,
    )


def test_sqlite_store_requires_setup(tmp_path):
    store = SQLiteMemoryStore(path=str(tmp_path / "mem.sqlite3"))
    with pytest.raises(RuntimeError, match="not initialized"):
        run_async(store.get_recent_events("thread"))


def test_sqlite_store_end_to_end_crud_and_search(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    store = SQLiteMemoryStore(path=str(db_path))

    async def scenario():
        await store.setup()

        await store.append_event(make_event("e1", "th1", 1000, "a"))
        await store.append_event(make_event("e2", "th1", 2000, "b"))
        await store.append_event(make_event("e3", "th1", 3000, "c"))
        await store.append_event(make_event("e4", "th2", 4000, "other"))

        recent = await store.get_recent_events("th1", limit=2)
        assert [event.id for event in recent] == ["e2", "e3"]

        since = await store.get_events_since("th1", since_ms=1500, limit=3)
        assert [event.id for event in since] == ["e2", "e3"]

        await store.put_state("th1", "z_last", {"v": 3})
        await store.put_state("th1", "a_first", {"v": 1})
        await store.put_state("th1", "a_next", {"v": 2})
        await store.put_state("th2", "a_first", {"v": 99})

        assert await store.get_state("th1", "a_first") == {"v": 1}
        assert await store.get_state("th1", "missing") is None
        assert list((await store.list_state("th1")).keys()) == [
            "a_first",
            "a_next",
            "z_last",
        ]
        assert await store.list_state("th1", prefix="a_") == {
            "a_first": {"v": 1},
            "a_next": {"v": 2},
        }

        mem1 = make_memory("m1", "u1", "global", "python sqlite testing", 1000)
        mem2 = make_memory("m2", "u1", "project:x", "vector memory search", 2000)
        mem3 = make_memory("m3", None, "global", "system memory", 3000)

        await store.upsert_long_term_memory(mem1, embedding=[1.0, 0.0])
        await store.upsert_long_term_memory(mem2, embedding=[0.0, 1.0])
        await store.upsert_long_term_memory(mem3, embedding=[0.6, 0.6])

        listed_user = await store.list_long_term_memories("u1")
        assert [m.id for m in listed_user] == ["m2", "m1"]

        listed_scope = await store.list_long_term_memories("u1", scope="project:x")
        assert [m.id for m in listed_scope] == ["m2"]

        text_hits = await store.search_long_term_memory_text("u1", "sqlite")
        assert [m.id for m in text_hits] == ["m1"]
        assert await store.search_long_term_memory_text("u1", "   ") == []

        vector_hits = await store.search_long_term_memory_vector(
            "u1", [1.0, 0.0], limit=5
        )
        assert [m.id for m, _ in vector_hits] == ["m1", "m2"]

        filtered_hits = await store.search_long_term_memory_vector(
            "u1", [1.0, 0.0], min_score=0.9
        )
        assert [m.id for m, _ in filtered_hits] == ["m1"]

        await store.delete_long_term_memory("u2", "m1")
        assert {m.id for m in await store.list_long_term_memories("u1")} == {"m1", "m2"}

        await store.delete_long_term_memory("u1", "m1")
        assert [m.id for m in await store.list_long_term_memories("u1")] == ["m2"]

        await store.close()

    run_async(scenario())


def test_sqlite_upsert_keeps_created_at_and_embedding_when_not_provided(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    store = SQLiteMemoryStore(path=str(db_path))

    async def scenario():
        await store.setup()

        first = make_memory("m1", "u1", "global", "first", 1000, created_at=111)
        await store.upsert_long_term_memory(first, embedding=[1.0, 0.0])

        second = make_memory("m1", "u1", "global", "second", 2000, created_at=999)
        await store.upsert_long_term_memory(second, embedding=None)

        listed = await store.list_long_term_memories("u1")
        assert len(listed) == 1
        assert listed[0].text == "second"
        assert listed[0].created_at == 111
        assert listed[0].updated_at == 2000

        hits = await store.search_long_term_memory_vector("u1", [1.0, 0.0], limit=5)
        assert [m.id for m, _ in hits] == ["m1"]

        await store.close()

    run_async(scenario())


def test_sqlite_store_persists_data_across_reopen(tmp_path):
    db_path = tmp_path / "persistent.sqlite3"
    initial = SQLiteMemoryStore(path=str(db_path))

    async def write_phase():
        await initial.setup()
        await initial.append_event(make_event("e1", "thread", 1000, "first"))
        await initial.put_state("thread", "session", {"id": 1})
        await initial.upsert_long_term_memory(
            make_memory("m1", "u1", "global", "remember this", 1234),
            embedding=[0.3, 0.7],
        )
        await initial.close()

    run_async(write_phase())

    reopened = SQLiteMemoryStore(path=str(db_path))

    async def read_phase():
        await reopened.setup()
        assert [e.id for e in await reopened.get_recent_events("thread")] == ["e1"]
        assert await reopened.get_state("thread", "session") == {"id": 1}
        assert [m.id for m in await reopened.list_long_term_memories("u1")] == ["m1"]
        assert [
            m.id
            for m, _ in await reopened.search_long_term_memory_vector("u1", [0.3, 0.7])
        ] == ["m1"]
        await reopened.close()

    run_async(read_phase())
