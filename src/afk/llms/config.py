from __future__ import annotations
import os

"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.

"""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LLMConfig:
    # Models
    default_model: str
    embedding_model: str | None

    # Reliability
    timeout_s: float
    max_retries: int
    backoff_base_s: float
    backoff_jitter_s: float

    # Structured output behavior
    json_max_retries: int

    # Safety/limits
    max_input_chars: int
    api_base_url: str | None = None
    api_key: str | None = None

    @staticmethod
    def from_env() -> "LLMConfig":
        return LLMConfig(
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
        )
