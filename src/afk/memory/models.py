from __future__ import annotations

"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.

This module defines core data models and JSON helpers for the AFK memory subsystem.
"""

from dataclasses import dataclass, field
import json
import time
import uuid
from typing import TypeAlias, cast
from typing import Any, List, Literal, Optional

EventType = Literal["tool_call", "tool_result", "message", "system", "trace"]
JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    """Represents an event in short-term memory for a specific conversation thread."""

    id: str
    thread_id: str
    user_id: Optional[str]
    type: EventType
    timestamp: int
    payload: JsonObject
    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class LongTermMemory:
    """Represents a durable memory record for retrieval and personalization."""

    id: str
    user_id: Optional[str]
    scope: str  # e.g. "global", "org:123", "project:abc"
    data: JsonObject
    text: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: JsonObject = field(default_factory=dict)
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = field(default_factory=lambda: int(time.time() * 1000))


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id(prefix: str = "mem") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def json_dumps(obj: JsonValue | dict[str, Any] | list[Any] | Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def json_loads(s: str) -> JsonValue:
    return cast(JsonValue, json.loads(s))
