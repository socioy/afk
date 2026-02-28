"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

LLM resolution helpers for agent runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ...llms import LLMClient, create_llm_client
from ...llms.providers import LLMTransport
from ..errors import AgentConfigurationError

ModelResolver = Callable[[str], LLMTransport]


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """Resolved model metadata plus instantiated LLM adapter."""

    requested_model: str
    normalized_model: str
    llm: LLMTransport
    adapter: str


def resolve_model_to_llm(
    model: str | LLMTransport,
    *,
    resolver: ModelResolver | None = None,
) -> ResolvedModel:
    """
    Resolve model input into a concrete LLM adapter + normalized model name.
    """
    if isinstance(model, LLMClient):
        normalized = model.settings.default_model
        return ResolvedModel(
            requested_model=normalized,
            normalized_model=normalized,
            llm=model,
            adapter=model.provider_id,
        )

    if hasattr(model, "provider_id") and hasattr(model, "chat"):
        normalized = "unknown"
        return ResolvedModel(
            requested_model=normalized,
            normalized_model=normalized,
            llm=model,
            adapter=model.provider_id,
        )

    if not isinstance(model, str) or not model.strip():
        raise AgentConfigurationError(
            "Agent.model must be either an LLM instance or a non-empty model string."
        )

    raw = model.strip()
    if resolver is not None:
        llm = resolver(raw)
        if not hasattr(llm, "provider_id") or not hasattr(llm, "chat"):
            raise AgentConfigurationError(
                "Custom model resolver must return an LLM transport."
            )
        return ResolvedModel(
            requested_model=raw,
            normalized_model=raw,
            llm=llm,
            adapter=llm.provider_id,
        )

    lowered = raw.lower()
    prefix, normalized = _split_prefix(raw)

    if prefix in {"openai"}:
        return ResolvedModel(
            requested_model=raw,
            normalized_model=normalized,
            llm=create_llm_client(provider="openai"),
            adapter="openai",
        )

    if prefix in {"anthropic", "claude"}:
        return ResolvedModel(
            requested_model=raw,
            normalized_model=normalized,
            llm=create_llm_client(provider="anthropic_agent"),
            adapter="anthropic_agent",
        )

    if prefix in {"litellm", "ollama", "ollama_chat"}:
        return ResolvedModel(
            requested_model=raw,
            normalized_model=raw,
            llm=create_llm_client(provider="litellm"),
            adapter="litellm",
        )

    if lowered.startswith("gpt"):
        return ResolvedModel(
            requested_model=raw,
            normalized_model=raw,
            llm=create_llm_client(provider="openai"),
            adapter="openai",
        )

    if lowered.startswith("claude"):
        return ResolvedModel(
            requested_model=raw,
            normalized_model=raw,
            llm=create_llm_client(provider="anthropic_agent"),
            adapter="anthropic_agent",
        )

    return ResolvedModel(
        requested_model=raw,
        normalized_model=raw,
        llm=create_llm_client(provider="litellm"),
        adapter="litellm",
    )


def _split_prefix(model_name: str) -> tuple[str | None, str]:
    """Split `provider/model` strings into provider prefix and model name."""
    parts = model_name.split("/", 1)
    if len(parts) == 1:
        return None, model_name
    return parts[0].strip().lower(), parts[1].strip()
