"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

OpenAI-backed adapter built on top of the shared Responses adapter base.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from ...errors import LLMConfigurationError
from ...types import Message
from ..base.responses import ResponsesClientBase
from ..shared import (
    collect_headers,
    json_text,
    normalize_role,
    to_input_text_part,
    tool_result_label,
)


class OpenAIClient(ResponsesClientBase):
    """Concrete adapter using `openai.AsyncOpenAI` Responses API."""

    _OPENAI_SUPPORTED_THINKING_EFFORTS = {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    }

    @property
    def provider_id(self) -> str:
        return "openai"

    def _provider_supported_thinking_efforts(self) -> set[str] | None:
        """OpenAI Responses API supports official effort labels."""
        return set(self._OPENAI_SUPPORTED_THINKING_EFFORTS)

    def _provider_default_thinking_effort(self) -> str | None:
        """Default effort when `thinking=True` and no explicit effort is set."""
        return "medium"

    async def _responses_create(self, payload: dict[str, Any]) -> Any:
        """Dispatch chat/stream payload to OpenAI Responses API."""
        client = self._build_client()
        call_payload = self._with_transport_headers(payload)
        return await client.responses.create(**call_payload)

    async def _embedding_create(self, payload: dict[str, Any]) -> Any:
        """Dispatch embedding payload to OpenAI embeddings API."""
        client = self._build_client()
        call_payload = self._with_transport_headers(payload)
        return await client.embeddings.create(**call_payload)

    def _message_to_responses_input_items(
        self, message: Message
    ) -> list[dict[str, Any]]:
        """Convert one normalized message into OpenAI Responses input items."""
        role = normalize_role(message.role)

        if isinstance(message.content, str):
            content: str | list[dict[str, Any]] = message.content
            if message.role == "tool":
                label = tool_result_label(message.name)
                content = f"[tool_result:{label}] {message.content}"
            return [
                {
                    "type": "message",
                    "role": role,
                    "content": content,
                }
            ]

        message_parts: list[dict[str, Any]] = []
        side_items: list[dict[str, Any]] = []

        for part in message.content:
            if not isinstance(part, dict):
                message_parts.append(to_input_text_part(part))
                continue

            p_type = part.get("type")
            if p_type == "text":
                text = part.get("text")
                if isinstance(text, str):
                    message_parts.append({"type": "input_text", "text": text})
                continue

            if p_type == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict) and isinstance(
                    image_url.get("url"), str
                ):
                    message_parts.append(
                        {
                            "type": "input_image",
                            "image_url": image_url["url"],
                        }
                    )
                continue

            if p_type == "tool_use":
                tool_use_id = part.get("id")
                name = part.get("name")
                args = part.get("input")
                if isinstance(tool_use_id, str) and isinstance(name, str):
                    side_items.append(
                        {
                            "type": "function_call",
                            "call_id": tool_use_id,
                            "name": name,
                            "arguments": json.dumps(
                                args, ensure_ascii=True, default=str
                            ),
                        }
                    )
                else:
                    message_parts.append(to_input_text_part(json_text(part)))
                continue

            if p_type == "tool_result":
                tool_use_id = part.get("tool_use_id")
                content = part.get("content")
                if isinstance(tool_use_id, str) and isinstance(content, str):
                    side_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_use_id,
                            "output": content,
                        }
                    )
                else:
                    message_parts.append(to_input_text_part(json_text(part)))
                continue

            message_parts.append(to_input_text_part(json_text(part)))

        items: list[dict[str, Any]] = []
        if message_parts:
            items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": message_parts,
                }
            )

        items.extend(side_items)

        if not items:
            items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": [{"type": "input_text", "text": ""}],
                }
            )

        return items

    def _structured_output_payload(
        self,
        response_model: type[BaseModel],
    ) -> dict[str, Any]:
        """OpenAI structured output payload for strict JSON schema mode."""
        return {
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                    "strict": True,
                }
            }
        }

    def _build_client(self) -> Any:
        """Construct or return cached AsyncOpenAI client from shared config."""
        if hasattr(self, "_client") and self._client is not None:
            return self._client

        try:
            from openai import AsyncOpenAI
        except Exception as e:  # pragma: no cover - environment dependent
            raise LLMConfigurationError(
                "openai package is not installed. Install it with: pip install openai"
            ) from e

        kwargs: dict[str, Any] = {}
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.api_base_url:
            kwargs["base_url"] = self.config.api_base_url

        self._client = AsyncOpenAI(**kwargs)
        return self._client

    def _with_transport_headers(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Map AFK transport keys into OpenAI request options/headers."""
        out = dict(payload)
        headers = collect_headers(
            out.get("extra_headers"),
            idempotency_key=out.pop("idempotency_key", None),
            metadata=out.get("metadata"),
        )

        if headers:
            out["extra_headers"] = headers

        return out
