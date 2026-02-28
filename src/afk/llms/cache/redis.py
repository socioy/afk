"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Module: cache/redis.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..types import LLMResponse, ToolCall, Usage
from .base import LLMCacheBackend


@dataclass(slots=True)
class RedisLLMCache(LLMCacheBackend):
    """Redis-backed cache backend for multi-process deployments."""

    backend_id: str = "redis"

    def __init__(self, redis_client) -> None:
        self._redis = redis_client

    async def get(self, key: str) -> LLMResponse | None:
        blob = await self._redis.get(key)
        if blob is None:
            return None
        try:
            row = json.loads(blob)
        except Exception:
            return None

        tool_calls = [
            ToolCall(
                id=item.get("id") if isinstance(item, dict) else None,
                tool_name=item.get("tool_name", "") if isinstance(item, dict) else "",
                arguments=item.get("arguments", {}) if isinstance(item, dict) else {},
            )
            for item in (row.get("tool_calls") or [])
        ]

        usage_row = row.get("usage") if isinstance(row.get("usage"), dict) else {}
        return LLMResponse(
            text=row.get("text", ""),
            request_id=row.get("request_id"),
            provider_request_id=row.get("provider_request_id"),
            session_token=row.get("session_token"),
            checkpoint_token=row.get("checkpoint_token"),
            structured_response=row.get("structured_response"),
            tool_calls=tool_calls,
            finish_reason=row.get("finish_reason"),
            usage=Usage(
                input_tokens=usage_row.get("input_tokens"),
                output_tokens=usage_row.get("output_tokens"),
                total_tokens=usage_row.get("total_tokens"),
            ),
            raw=row.get("raw") if isinstance(row.get("raw"), dict) else {},
            model=row.get("model"),
        )

    async def set(self, key: str, value: LLMResponse, *, ttl_s: float) -> None:
        payload = {
            "text": value.text,
            "request_id": value.request_id,
            "provider_request_id": value.provider_request_id,
            "session_token": value.session_token,
            "checkpoint_token": value.checkpoint_token,
            "structured_response": value.structured_response,
            "tool_calls": [
                {"id": tc.id, "tool_name": tc.tool_name, "arguments": tc.arguments}
                for tc in value.tool_calls
            ],
            "finish_reason": value.finish_reason,
            "usage": {
                "input_tokens": value.usage.input_tokens,
                "output_tokens": value.usage.output_tokens,
                "total_tokens": value.usage.total_tokens,
            },
            "raw": value.raw,
            "model": value.model,
        }
        await self._redis.setex(
            key, int(max(1, ttl_s)), json.dumps(payload, ensure_ascii=True)
        )

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)
