"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

This module defines core data models and JSON helpers for the AFK memory subsystem.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

EventType = Literal["tool_call", "tool_result", "message", "system", "trace"]
JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    """Represents an event in short-term memory for a specific conversation thread."""

    id: str
    thread_id: str
    user_id: str | None
    type: EventType
    timestamp: int
    payload: JsonObject
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class LongTermMemory:
    """Represents a durable memory record for retrieval and personalization."""

    id: str
    user_id: str | None
    scope: str  # e.g. "global", "org:123", "project:abc"
    data: JsonObject
    text: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: JsonObject = field(default_factory=dict)
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = field(default_factory=lambda: int(time.time() * 1000))
