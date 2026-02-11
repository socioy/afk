from __future__ import annotations

"""Public API for AFK memory models, stores, and factory helpers."""

from .models import JsonObject, JsonValue, LongTermMemory, MemoryEvent, now_ms, new_id
from .store import InMemoryMemoryStore, MemoryCapabilities, MemoryStore, SQLiteMemoryStore
from .vector import cosine_similarity
from .factory import create_memory_store_from_env


def __getattr__(name: str):
    if name == "RedisMemoryStore":
        from .store.redis import RedisMemoryStore

        return RedisMemoryStore
    if name == "PostgresMemoryStore":
        from .store.postgres import PostgresMemoryStore

        return PostgresMemoryStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "JsonValue",
    "JsonObject",
    "MemoryEvent",
    "LongTermMemory",
    "now_ms",
    "new_id",
    "MemoryStore",
    "MemoryCapabilities",
    "cosine_similarity",
    "InMemoryMemoryStore",
    "SQLiteMemoryStore",
    "RedisMemoryStore",
    "PostgresMemoryStore",
    "create_memory_store_from_env",
]
