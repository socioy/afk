from __future__ import annotations

"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.

This module provides the public API for the AFK memory subsystem, including models, stores, and utilities.
"""

from .models import JsonObject, JsonValue, LongTermMemory, MemoryEvent, now_ms, new_id
from .store import (
    InMemoryMemoryStore,
    MemoryCapabilities,
    MemoryStore,
    SQLiteMemoryStore,
)
from .vector import cosine_similarity
from .factory import create_memory_store_from_env
from .lifecycle import (
    MemoryCompactionResult,
    RetentionPolicy,
    StateRetentionPolicy,
    apply_event_retention,
    apply_state_retention,
    compact_thread_memory,
)


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
    "RetentionPolicy",
    "StateRetentionPolicy",
    "MemoryCompactionResult",
    "apply_event_retention",
    "apply_state_retention",
    "compact_thread_memory",
]
