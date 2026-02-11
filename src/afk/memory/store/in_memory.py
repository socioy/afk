from __future__ import annotations

"""In-process memory store implementation for local development and tests."""

import asyncio
from typing import Optional, Sequence

from ..models import JsonValue, LongTermMemory, MemoryEvent, json_dumps
from ..vector import cosine_similarity
from .base import MemoryCapabilities, MemoryStore


class InMemoryMemoryStore(MemoryStore):
    """Fast, process-local memory backend with full text/vector retrieval support."""

    capabilities = MemoryCapabilities(text_search=True, vector_search=True, atomic_upsert=True, ttl=False)

    def __init__(self) -> None:
        super().__init__()
        self._lock = asyncio.Lock()
        self._events_by_thread: dict[str, list[MemoryEvent]] = {}
        self._state_by_thread_key: dict[tuple[str, str], JsonValue] = {}
        self._memory_by_id: dict[str, LongTermMemory] = {}
        self._embedding_by_memory_id: dict[str, list[float]] = {}

    async def append_event(self, event: MemoryEvent) -> None:
        self._ensure_setup()
        async with self._lock:
            self._events_by_thread.setdefault(event.thread_id, []).append(event)

    async def get_recent_events(self, thread_id: str, limit: int = 50) -> list[MemoryEvent]:
        self._ensure_setup()
        async with self._lock:
            return list(self._events_by_thread.get(thread_id, [])[-limit:])

    async def get_events_since(self, thread_id: str, since_ms: int, limit: int = 500) -> list[MemoryEvent]:
        self._ensure_setup()
        async with self._lock:
            matches = [event for event in self._events_by_thread.get(thread_id, []) if event.timestamp >= since_ms]
            return matches[:limit]

    async def put_state(self, thread_id: str, key: str, value: JsonValue) -> None:
        self._ensure_setup()
        async with self._lock:
            self._state_by_thread_key[(thread_id, key)] = value

    async def get_state(self, thread_id: str, key: str) -> JsonValue | None:
        self._ensure_setup()
        async with self._lock:
            return self._state_by_thread_key.get((thread_id, key))

    async def list_state(self, thread_id: str, prefix: str | None = None) -> dict[str, JsonValue]:
        self._ensure_setup()
        async with self._lock:
            filtered_state: dict[str, JsonValue] = {}
            for (candidate_thread_id, state_key), state_value in self._state_by_thread_key.items():
                if candidate_thread_id != thread_id:
                    continue
                if prefix is not None and not state_key.startswith(prefix):
                    continue
                filtered_state[state_key] = state_value
            return dict(sorted(filtered_state.items(), key=lambda item: item[0]))

    async def upsert_long_term_memory(
        self,
        memory: LongTermMemory,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> None:
        self._ensure_setup()
        async with self._lock:
            self._memory_by_id[memory.id] = memory
            if embedding is not None:
                self._embedding_by_memory_id[memory.id] = [float(value) for value in embedding]

    async def delete_long_term_memory(self, user_id: str | None, memory_id: str) -> None:
        self._ensure_setup()
        async with self._lock:
            memory = self._memory_by_id.get(memory_id)
            if memory is None:
                return
            if user_id is not None and memory.user_id != user_id:
                return
            del self._memory_by_id[memory_id]
            self._embedding_by_memory_id.pop(memory_id, None)

    async def list_long_term_memories(
        self,
        user_id: str | None,
        *,
        scope: str | None = None,
        limit: int = 100,
    ) -> list[LongTermMemory]:
        self._ensure_setup()
        async with self._lock:
            candidates = [memory for memory in self._memory_by_id.values() if memory.user_id == user_id]
            if scope is not None:
                candidates = [memory for memory in candidates if memory.scope == scope]
            candidates.sort(key=lambda memory: memory.updated_at, reverse=True)
            return candidates[:limit]

    async def search_long_term_memory_text(
        self,
        user_id: str | None,
        query: str,
        *,
        scope: str | None = None,
        limit: int = 20,
    ) -> list[LongTermMemory]:
        self._ensure_setup()
        normalized_query = query.strip().lower()
        if not normalized_query:
            return []

        async with self._lock:
            candidates = [memory for memory in self._memory_by_id.values() if memory.user_id == user_id]
            if scope is not None:
                candidates = [memory for memory in candidates if memory.scope == scope]

            ranked: list[tuple[int, int, LongTermMemory]] = []
            for memory in candidates:
                searchable_text = " ".join(
                    [
                        (memory.text or ""),
                        " ".join(memory.tags),
                        json_dumps(memory.data),
                    ]
                ).lower()
                if normalized_query not in searchable_text:
                    continue
                rank_position = searchable_text.find(normalized_query)
                ranked.append((rank_position, -memory.updated_at, memory))

            ranked.sort(key=lambda item: (item[0], item[1]))
            return [memory for _, _, memory in ranked[:limit]]

    async def search_long_term_memory_vector(
        self,
        user_id: str | None,
        query_embedding: Sequence[float],
        *,
        scope: str | None = None,
        limit: int = 20,
        min_score: float | None = None,
    ) -> list[tuple[LongTermMemory, float]]:
        self._ensure_setup()
        query_values = [float(value) for value in query_embedding]
        async with self._lock:
            candidates = [memory for memory in self._memory_by_id.values() if memory.user_id == user_id]
            if scope is not None:
                candidates = [memory for memory in candidates if memory.scope == scope]

            ranked: list[tuple[LongTermMemory, float]] = []
            for memory in candidates:
                embedding = self._embedding_by_memory_id.get(memory.id)
                if embedding is None:
                    continue
                similarity = cosine_similarity(query_values, embedding)
                if min_score is not None and similarity < min_score:
                    continue
                ranked.append((memory, similarity))

            ranked.sort(key=lambda item: item[1], reverse=True)
            return ranked[:limit]
