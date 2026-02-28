from __future__ import annotations

import importlib.util
import os
import time
from contextlib import contextmanager

import pytest

from afk.memory import (
    InMemoryMemoryStore,
    SQLiteMemoryStore,
    cosine_similarity,
    create_memory_store_from_env,
    new_id,
    now_ms,
)
from afk.memory.factory import _env_bool
from afk.memory.utils import json_dumps, json_loads
from afk.memory.vector import format_pgvector


@contextmanager
def preserved_env():
    snapshot = os.environ.copy()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


def test_now_ms_and_new_id_helpers():
    before = int(time.time() * 1000)
    value = now_ms()
    after = int(time.time() * 1000)
    assert before <= value <= after

    id_1 = new_id()
    id_2 = new_id("thread")
    assert id_1.startswith("mem_")
    assert id_2.startswith("thread_")
    assert id_1 != id_2


def test_json_helpers_round_trip_with_nested_values():
    payload = {
        "name": "afk",
        "count": 1,
        "nested": {"ok": True, "icon": ":)"},
        "items": [1, "two", None],
    }
    serialized = json_dumps(payload)
    deserialized = json_loads(serialized)

    assert '"name":"afk"' in serialized
    assert deserialized == payload


def test_cosine_similarity_and_vector_formatting():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 2]) == pytest.approx(0.0)
    assert format_pgvector([1, 2.5, 3.125]) == "[1,2.5,3.125]"

    with pytest.raises(ValueError, match="Embedding dim mismatch"):
        cosine_similarity([1, 2], [1])

    with pytest.raises(ValueError, match="1D vectors"):
        cosine_similarity([[1, 2]], [1, 2])  # type: ignore[arg-type]


def test_env_bool_parsing():
    with preserved_env():
        os.environ["FLAG"] = "TRUE"
        assert _env_bool("FLAG", default=False) is True
        os.environ["FLAG"] = "off"
        assert _env_bool("FLAG", default=True) is False
        del os.environ["FLAG"]
        assert _env_bool("FLAG", default=True) is True


def test_memory_factory_selects_backends_and_env_options(tmp_path):
    with preserved_env():
        os.environ["AFK_MEMORY_BACKEND"] = "in_memory"
        store = create_memory_store_from_env()
        assert isinstance(store, InMemoryMemoryStore)

    with preserved_env():
        db_path = str(tmp_path / "memory.sqlite3")
        os.environ["AFK_MEMORY_BACKEND"] = "sqlite"
        os.environ["AFK_SQLITE_PATH"] = db_path
        store = create_memory_store_from_env()
        assert isinstance(store, SQLiteMemoryStore)
        assert store.path == db_path

    with preserved_env():
        os.environ["AFK_MEMORY_BACKEND"] = "redis"
        os.environ["AFK_REDIS_HOST"] = "redis-host"
        os.environ["AFK_REDIS_PORT"] = "6380"
        os.environ["AFK_REDIS_DB"] = "9"
        os.environ["AFK_REDIS_PASSWORD"] = "secret"
        os.environ["AFK_REDIS_EVENTS_MAX"] = "123"
        if importlib.util.find_spec("redis") is None:
            with pytest.raises(ModuleNotFoundError):
                create_memory_store_from_env()
        else:
            store = create_memory_store_from_env()
            assert store.__class__.__name__ == "RedisMemoryStore"
            assert store.url == "redis://:secret@redis-host:6380/9"
            assert store.events_max_per_thread == 123

    with preserved_env():
        os.environ["AFK_MEMORY_BACKEND"] = "postgres"
        os.environ["AFK_PG_HOST"] = "pg-host"
        os.environ["AFK_PG_PORT"] = "5544"
        os.environ["AFK_PG_USER"] = "afk"
        os.environ["AFK_PG_PASSWORD"] = "pw"
        os.environ["AFK_PG_DB"] = "afkdb"
        os.environ["AFK_PG_SSL"] = "yes"
        os.environ["AFK_PG_POOL_MIN"] = "2"
        os.environ["AFK_PG_POOL_MAX"] = "11"
        os.environ["AFK_VECTOR_DIM"] = "1536"
        if importlib.util.find_spec("asyncpg") is None:
            with pytest.raises(ModuleNotFoundError):
                create_memory_store_from_env()
        else:
            store = create_memory_store_from_env()
            assert store.__class__.__name__ == "PostgresMemoryStore"
            assert store.dsn == "postgresql://afk:pw@pg-host:5544/afkdb"
            assert store.ssl is True
            assert store.pool_min == 2
            assert store.pool_max == 11
            assert store.vector_dim == 1536


def test_memory_factory_postgres_requires_vector_dim():
    with preserved_env():
        os.environ["AFK_MEMORY_BACKEND"] = "postgres"
        with pytest.raises(ValueError, match="AFK_VECTOR_DIM is required"):
            create_memory_store_from_env()


def test_memory_factory_rejects_unknown_backend():
    with preserved_env():
        os.environ["AFK_MEMORY_BACKEND"] = "unknown-backend"
        with pytest.raises(ValueError, match="Unknown AFK_MEMORY_BACKEND"):
            create_memory_store_from_env()
