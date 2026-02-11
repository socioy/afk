from __future__ import annotations

"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.

This module defines abstract interfaces and shared capabilities for memory store backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

from ..models import JsonValue, LongTermMemory, MemoryEvent


@dataclass(frozen=True, slots=True)
class MemoryCapabilities:
    """Describes optional backend features."""

    text_search: bool = True
    vector_search: bool = True
    atomic_upsert: bool = True
    ttl: bool = False


class MemoryStore(ABC):
    """
    Base contract for all memory backends.

    Vector behavior:
    - `upsert_long_term_memory(..., embedding=...)` stores embeddings when provided.
    - `search_long_term_memory_vector(...)` returns `(memory, similarity)` pairs where
      similarity is cosine similarity in [-1, 1], higher is better.
    """

    capabilities: MemoryCapabilities = MemoryCapabilities()

    def __init__(self) -> None:
        self._is_setup = False

    async def setup(self) -> None:
        """Initialize backend resources."""
        self._is_setup = True

    async def close(self) -> None:
        """Release backend resources."""
        self._is_setup = False

    async def __aenter__(self) -> "MemoryStore":
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def _ensure_setup(self) -> None:
        if not self._is_setup:
            raise RuntimeError(
                "MemoryStore is not initialized. Call setup() or use `async with`."
            )

    @abstractmethod
    async def append_event(self, event: MemoryEvent) -> None:
        """Append one event for a thread."""

    @abstractmethod
    async def get_recent_events(
        self, thread_id: str, limit: int = 50
    ) -> list[MemoryEvent]:
        """Return recent events for a thread in chronological order."""

    @abstractmethod
    async def get_events_since(
        self, thread_id: str, since_ms: int, limit: int = 500
    ) -> list[MemoryEvent]:
        """Return events newer than `since_ms` in chronological order."""

    @abstractmethod
    async def put_state(self, thread_id: str, key: str, value: JsonValue) -> None:
        """Set a state value for a thread-scoped key."""

    @abstractmethod
    async def get_state(self, thread_id: str, key: str) -> Optional[JsonValue]:
        """Return state value for a thread-scoped key."""

    @abstractmethod
    async def list_state(
        self, thread_id: str, prefix: str | None = None
    ) -> dict[str, JsonValue]:
        """List thread-scoped state, optionally by key prefix."""

    @abstractmethod
    async def upsert_long_term_memory(
        self,
        memory: LongTermMemory,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> None:
        """Insert or update one long-term memory record."""

    @abstractmethod
    async def delete_long_term_memory(
        self, user_id: Optional[str], memory_id: str
    ) -> None:
        """Delete one long-term memory record."""

    @abstractmethod
    async def list_long_term_memories(
        self,
        user_id: Optional[str],
        *,
        scope: str | None = None,
        limit: int = 100,
    ) -> list[LongTermMemory]:
        """List long-term memories for a user and optional scope."""

    @abstractmethod
    async def search_long_term_memory_text(
        self,
        user_id: Optional[str],
        query: str,
        *,
        scope: str | None = None,
        limit: int = 20,
    ) -> list[LongTermMemory]:
        """Text search over long-term memories."""

    @abstractmethod
    async def search_long_term_memory_vector(
        self,
        user_id: Optional[str],
        query_embedding: Sequence[float],
        *,
        scope: str | None = None,
        limit: int = 20,
        min_score: float | None = None,
    ) -> list[tuple[LongTermMemory, float]]:
        """Vector similarity search over long-term memories."""
