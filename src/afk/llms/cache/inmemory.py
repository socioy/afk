"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Module: cache/inmemory.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..types import LLMResponse
from .base import CacheEntry, LLMCacheBackend


@dataclass(slots=True)
class InMemoryLLMCache(LLMCacheBackend):
    """Process-local cache backend suitable for development/test workloads."""

    backend_id: str = "inmemory"

    def __post_init__(self) -> None:
        self._rows: dict[str, CacheEntry] = {}

    async def get(self, key: str) -> LLMResponse | None:
        row = self._rows.get(key)
        if row is None:
            return None
        if row.expires_at_s < time.time():
            self._rows.pop(key, None)
            return None
        return row.value

    async def set(self, key: str, value: LLMResponse, *, ttl_s: float) -> None:
        self._rows[key] = CacheEntry(value=value, expires_at_s=time.time() + ttl_s)

    async def delete(self, key: str) -> None:
        self._rows.pop(key, None)
