from __future__ import annotations

"""
Factory utilities for constructing concrete LLM adapters.
"""

import os
from typing import TYPE_CHECKING, Callable, Mapping

from .config import LLMConfig
from .errors import LLMConfigurationError
from .middleware import MiddlewareStack

if TYPE_CHECKING:
    from .llm import LLM


AdapterFactory = Callable[[LLMConfig, MiddlewareStack], "LLM"]
_BUILTIN_ADAPTERS = {"litellm", "anthropic_agent", "openai"}
_REGISTRY: dict[str, AdapterFactory] = {}


def register_llm_adapter(
    name: str,
    factory: AdapterFactory,
    *,
    overwrite: bool = False,
) -> None:
    """Register a custom adapter factory by name."""
    key = name.strip().lower()
    if not key:
        raise ValueError("Adapter name must be non-empty")

    if (not overwrite) and key in _REGISTRY:
        raise ValueError(f"Adapter already registered: {key}")

    _REGISTRY[key] = factory


def available_llm_adapters() -> list[str]:
    """Return built-in and runtime-registered adapter names."""
    return sorted(set(_BUILTIN_ADAPTERS) | set(_REGISTRY.keys()))


def create_llm(
    adapter: str,
    *,
    config: LLMConfig | None = None,
    middlewares: MiddlewareStack | None = None,
    thinking_effort_aliases: Mapping[str, str] | None = None,
    supported_thinking_efforts: set[str] | None = None,
    default_thinking_effort: str | None = None,
) -> "LLM":
    """Create an LLM client instance for a specific adapter key."""
    key = adapter.strip().lower()
    if not key:
        raise LLMConfigurationError("Adapter name must be non-empty")

    cfg = config or LLMConfig.from_env()
    mws = middlewares or MiddlewareStack()

    factory = _REGISTRY.get(key)
    if factory is None:
        factory = _builtin_factory(
            key,
            thinking_effort_aliases=thinking_effort_aliases,
            supported_thinking_efforts=supported_thinking_efforts,
            default_thinking_effort=default_thinking_effort,
        )
    elif any(
        value is not None
        for value in (
            thinking_effort_aliases,
            supported_thinking_efforts,
            default_thinking_effort,
        )
    ):
        raise LLMConfigurationError(
            "thinking-effort overrides are only supported by built-in adapters "
            "via `create_llm`. For custom adapters, instantiate the class directly."
        )

    return factory(cfg, mws)


def create_llm_from_env(
    *,
    config: LLMConfig | None = None,
    middlewares: MiddlewareStack | None = None,
    thinking_effort_aliases: Mapping[str, str] | None = None,
    supported_thinking_efforts: set[str] | None = None,
    default_thinking_effort: str | None = None,
) -> "LLM":
    """Create an LLM client using `AFK_LLM_ADAPTER` (defaults to `litellm`)."""
    adapter = os.getenv("AFK_LLM_ADAPTER", "litellm")
    return create_llm(
        adapter,
        config=config,
        middlewares=middlewares,
        thinking_effort_aliases=thinking_effort_aliases,
        supported_thinking_efforts=supported_thinking_efforts,
        default_thinking_effort=default_thinking_effort,
    )


def _builtin_factory(
    adapter: str,
    *,
    thinking_effort_aliases: Mapping[str, str] | None,
    supported_thinking_efforts: set[str] | None,
    default_thinking_effort: str | None,
) -> AdapterFactory:
    """Resolve built-in adapter factories lazily to avoid hard imports."""
    if adapter == "litellm":
        from .clients.adapters.litellm import LiteLLMClient

        return lambda cfg, mws: LiteLLMClient(
            config=cfg,
            middlewares=mws,
            thinking_effort_aliases=thinking_effort_aliases,
            supported_thinking_efforts=supported_thinking_efforts,
            default_thinking_effort=default_thinking_effort,
        )

    if adapter == "anthropic_agent":
        from .clients.adapters.anthropic_agent import AnthropicAgentClient

        return lambda cfg, mws: AnthropicAgentClient(
            config=cfg,
            middlewares=mws,
            thinking_effort_aliases=thinking_effort_aliases,
            supported_thinking_efforts=supported_thinking_efforts,
            default_thinking_effort=default_thinking_effort,
        )

    if adapter == "openai":
        from .clients.adapters.openai import OpenAIClient

        return lambda cfg, mws: OpenAIClient(
            config=cfg,
            middlewares=mws,
            thinking_effort_aliases=thinking_effort_aliases,
            supported_thinking_efforts=supported_thinking_efforts,
            default_thinking_effort=default_thinking_effort,
        )

    raise LLMConfigurationError(
        f"Unknown LLM adapter '{adapter}'. Available: {', '.join(available_llm_adapters())}"
    )
