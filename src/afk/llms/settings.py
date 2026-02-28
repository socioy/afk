"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

LLM runtime settings and explicit config loading.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .config import LLMConfig


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Explicit settings used by LLM providers and runtime modules."""

    default_provider: str = "litellm"
    default_model: str = "gpt-4.1-mini"
    embedding_model: str | None = None
    api_base_url: str | None = None
    api_key: str | None = None

    timeout_s: float = 30.0
    max_retries: int = 3
    backoff_base_s: float = 0.5
    backoff_jitter_s: float = 0.15
    json_max_retries: int = 2
    max_input_chars: int = 200000

    stream_idle_timeout_s: float | None = 45.0

    @staticmethod
    def from_env() -> LLMSettings:
        """Load settings from environment variables."""
        return LLMSettings(
            default_provider=os.getenv("AFK_LLM_PROVIDER", "litellm"),
            default_model=os.getenv("AFK_LLM_MODEL", "gpt-4.1-mini"),
            embedding_model=os.getenv("AFK_EMBED_MODEL"),
            api_base_url=os.getenv("AFK_LLM_API_BASE_URL"),
            api_key=os.getenv("AFK_LLM_API_KEY"),
            timeout_s=float(os.getenv("AFK_LLM_TIMEOUT_S", "30")),
            max_retries=int(os.getenv("AFK_LLM_MAX_RETRIES", "3")),
            backoff_base_s=float(os.getenv("AFK_LLM_BACKOFF_BASE_S", "0.5")),
            backoff_jitter_s=float(os.getenv("AFK_LLM_BACKOFF_JITTER_S", "0.15")),
            json_max_retries=int(os.getenv("AFK_LLM_JSON_MAX_RETRIES", "2")),
            max_input_chars=int(os.getenv("AFK_LLM_MAX_INPUT_CHARS", "200000")),
            stream_idle_timeout_s=float(
                os.getenv("AFK_LLM_STREAM_IDLE_TIMEOUT_S", "45")
            ),
        )

    def to_legacy_config(self) -> LLMConfig:
        """Adapt settings into the adapter-level LLMConfig object."""
        return LLMConfig(
            default_model=self.default_model,
            embedding_model=self.embedding_model,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
            backoff_base_s=self.backoff_base_s,
            backoff_jitter_s=self.backoff_jitter_s,
            json_max_retries=self.json_max_retries,
            max_input_chars=self.max_input_chars,
            api_base_url=self.api_base_url,
            api_key=self.api_key,
        )
