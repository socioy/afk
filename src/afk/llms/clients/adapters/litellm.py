from __future__ import annotations

"""
LiteLLM-backed adapter built on top of the shared Responses adapter base.
"""

import json
from typing import Any

from pydantic import BaseModel

from ..base.responses import ResponsesClientBase
from ...errors import LLMConfigurationError
from ...types import Message


class LiteLLMClient(ResponsesClientBase):
    """Concrete adapter using `litellm` Responses API wrappers."""

    @property
    def provider_id(self) -> str:
        return "litellm"

    async def _responses_create(self, payload: dict[str, Any]) -> Any:
        """Dispatch chat/stream payload to `litellm.aresponses`."""
        try:
            from litellm import aresponses
        except Exception as e:  # pragma: no cover - environment dependent
            raise LLMConfigurationError(
                "litellm is not installed. Install the dependency to use LiteLLMClient."
            ) from e

        return await aresponses(**self._with_transport_defaults(payload))

    async def _embedding_create(self, payload: dict[str, Any]) -> Any:
        """Dispatch embedding payload to `litellm.aembedding`."""
        try:
            from litellm import aembedding
        except Exception as e:  # pragma: no cover - environment dependent
            raise LLMConfigurationError(
                "litellm is not installed. Install the dependency to use LiteLLMClient."
            ) from e

        return await aembedding(**self._with_transport_defaults(payload))

    def _message_to_responses_input_items(self, message: Message) -> list[dict[str, Any]]:
        """Convert one normalized message into one LiteLLM/OpenAI-style input item."""
        role = message.role if message.role in ("user", "assistant", "system") else "user"

        if isinstance(message.content, str):
            content: str | list[dict[str, Any]] = message.content
            if message.role == "tool":
                label = message.name or "tool"
                content = f"[tool_result:{label}] {message.content}"
            return [
                {
                    "type": "message",
                    "role": role,
                    "content": content,
                }
            ]

        parts: list[dict[str, Any]] = []
        for part in message.content:
            if not isinstance(part, dict):
                parts.append(
                    {
                        "type": "input_text",
                        "text": str(part),
                    }
                )
                continue

            p_type = part.get("type")
            if p_type == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append({"type": "input_text", "text": text})
                continue

            if p_type == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                    parts.append(
                        {
                            "type": "input_image",
                            "image_url": image_url["url"],
                        }
                    )
                continue

            if p_type == "tool_use":
                parts.append(
                    {
                        "type": "input_text",
                        "text": (
                            f"[tool_use:{part.get('name')}] "
                            f"{json.dumps(part.get('input'), ensure_ascii=True, default=str)}"
                        ),
                    }
                )
                continue

            if p_type == "tool_result":
                parts.append(
                    {
                        "type": "input_text",
                        "text": (
                            f"[tool_result:{part.get('tool_use_id')}] "
                            f"{part.get('content', '')}"
                        ),
                    }
                )
                continue

            parts.append(
                {
                    "type": "input_text",
                    "text": json.dumps(part, ensure_ascii=True, default=str),
                }
            )

        if message.role == "tool":
            label = message.name or "tool"
            prefix = {"type": "input_text", "text": f"[tool_result:{label}]"}
            parts = [prefix, *parts]

        if not parts:
            parts = [{"type": "input_text", "text": ""}]

        return [
            {
                "type": "message",
                "role": role,
                "content": parts,
            }
        ]

    def _structured_output_payload(
        self,
        response_model: type[BaseModel],
    ) -> dict[str, Any]:
        """LiteLLM accepts a model type directly for structured text format."""
        return {"text_format": response_model}

    def _with_transport_defaults(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply config-level transport defaults without overriding explicit extras."""
        out = dict(payload)

        idempotency_key = out.pop("idempotency_key", None)
        headers = out.get("headers")
        header_map: dict[str, str] = {}
        if isinstance(headers, dict):
            for key, value in headers.items():
                if isinstance(key, str) and isinstance(value, str):
                    header_map[key] = value

        if isinstance(idempotency_key, str) and idempotency_key:
            header_map.setdefault("Idempotency-Key", idempotency_key)

        metadata = out.get("metadata")
        if isinstance(metadata, dict):
            request_id = metadata.get("afk_request_id")
            if isinstance(request_id, str) and request_id:
                header_map.setdefault("X-Request-Id", request_id)

        if header_map:
            out["headers"] = header_map

        if self.config.api_base_url:
            out.setdefault("api_base", self.config.api_base_url)
        if self.config.api_key:
            out.setdefault("api_key", self.config.api_key)
        return out
