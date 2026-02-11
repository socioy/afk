from __future__ import annotations

"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.

This module provides memory store implementations and base contracts for the AFK memory subsystem.
"""

"""Memory store implementations and base contracts."""

from .base import MemoryCapabilities, MemoryStore
from .in_memory import InMemoryMemoryStore
from .sqlite import SQLiteMemoryStore


def __getattr__(name: str):
    if name == "RedisMemoryStore":
        from .redis import RedisMemoryStore

        return RedisMemoryStore
    if name == "PostgresMemoryStore":
        from .postgres import PostgresMemoryStore

        return PostgresMemoryStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MemoryCapabilities",
    "MemoryStore",
    "InMemoryMemoryStore",
    "SQLiteMemoryStore",
    "RedisMemoryStore",
    "PostgresMemoryStore",
]
