"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

This module provides memory store adapters for the AFK memory subsystem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .in_memory import InMemoryMemoryStore
from .sqlite import SQLiteMemoryStore

RedisMemoryStore = None  # type: ignore[assignment]
PostgresMemoryStore = None  # type: ignore[assignment]


__all__ = [
    "InMemoryMemoryStore",
    "SQLiteMemoryStore",
    "RedisMemoryStore",
    "PostgresMemoryStore",
]

try:
    from .redis import RedisMemoryStore as _RedisMemoryStore
except ModuleNotFoundError:  # optional dependency: redis
    pass
else:
    RedisMemoryStore = _RedisMemoryStore

try:
    from .postgres import PostgresMemoryStore as _PostgresMemoryStore
except ModuleNotFoundError:  # optional dependency: asyncpg
    pass
else:
    PostgresMemoryStore = _PostgresMemoryStore

if TYPE_CHECKING:
    # For type checking, import all store classes directly
    from .postgres import PostgresMemoryStore
    from .redis import RedisMemoryStore
