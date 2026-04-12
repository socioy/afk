"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Module: builder.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from .middleware import MiddlewareStack
from .observability import LLMObserver
from .profiles import PROFILES
from .runtime.client import LLMClient
from .runtime.contracts import (
    CachePolicy,
    CircuitBreakerPolicy,
    CoalescingPolicy,
    HedgingPolicy,
    RateLimitPolicy,
    RetryPolicy,
    TimeoutPolicy,
)
from .settings import LLMSettings
from .types import JSONValue


class LLMBuilder:
    """Builder-first DX for creating production-ready llm clients."""

    def __init__(self) -> None:
        self._provider: str | None = None
        self._settings = LLMSettings.from_env()
        self._provider_settings: dict[str, Mapping[str, JSONValue]] = {}
        self._middlewares: MiddlewareStack | None = None
        self._observers: list[LLMObserver] | None = None
        self._cache_backend = None
        self._router = None

        self._retry_policy: RetryPolicy | None = None
        self._timeout_policy: TimeoutPolicy | None = None
        self._rate_limit_policy: RateLimitPolicy | None = None
        self._breaker_policy: CircuitBreakerPolicy | None = None
        self._hedging_policy: HedgingPolicy | None = None
        self._cache_policy: CachePolicy | None = None
        self._coalescing_policy: CoalescingPolicy | None = None

    def provider(self, provider: str) -> LLMBuilder:
        """Set the default provider id used when `build()` is called."""
        self._provider = provider.strip().lower()
        return self

    def model(self, model: str) -> LLMBuilder:
        """Override the default model in builder settings."""
        self._settings = replace(self._settings, default_model=model)
        return self

    def settings(self, settings: LLMSettings) -> LLMBuilder:
        """Replace builder settings with an explicit `LLMSettings` instance."""
        self._settings = settings
        return self

    def profile(self, name: str) -> LLMBuilder:
        """Apply one named runtime profile from `afk.llms.profiles.PROFILES`."""
        key = name.strip().lower()
        profile = PROFILES.get(key)
        if profile is None:
            raise ValueError(f"Unknown llm profile '{name}'")
        self._retry_policy = profile.retry
        self._timeout_policy = profile.timeout
        self._rate_limit_policy = profile.rate_limit
        self._breaker_policy = profile.breaker
        self._hedging_policy = profile.hedging
        self._cache_policy = profile.cache
        self._coalescing_policy = profile.coalescing
        return self

    def for_agent_runtime(self) -> LLMBuilder:
        """
        Apply a production-oriented baseline for complex agent orchestration.

        This keeps defaults explicit while providing a one-call DX path.
        """
        return self.profile("production")

    def with_provider_settings(
        self,
        provider: str,
        settings: Mapping[str, JSONValue],
    ) -> LLMBuilder:
        """Attach provider-specific settings passed to provider factory hooks."""
        self._provider_settings[provider.strip().lower()] = settings
        return self

    def with_middlewares(self, middlewares: MiddlewareStack) -> LLMBuilder:
        """Configure middleware stacks for chat/stream/embed transport paths."""
        self._middlewares = middlewares
        return self

    def with_observers(self, observers: list[LLMObserver]) -> LLMBuilder:
        """Configure lifecycle observers for best-effort telemetry callbacks."""
        self._observers = list(observers)
        return self

    def with_cache(self, cache_backend) -> LLMBuilder:
        """Select one cache backend instance or registered backend id."""
        self._cache_backend = cache_backend
        return self

    def with_router(self, router) -> LLMBuilder:
        """Select one router instance or registered router id."""
        self._router = router
        return self

    def build(self) -> LLMClient:
        """Materialize one configured `LLMClient` instance."""
        provider = self._provider or self._settings.default_provider
        return LLMClient(
            provider=provider,
            settings=self._settings,
            provider_settings=self._provider_settings,
            middlewares=self._middlewares,
            observers=self._observers,
            cache_backend=self._cache_backend,
            router=self._router,
            retry_policy=self._retry_policy,
            timeout_policy=self._timeout_policy,
            rate_limit_policy=self._rate_limit_policy,
            circuit_breaker_policy=self._breaker_policy,
            hedging_policy=self._hedging_policy,
            cache_policy=self._cache_policy,
            coalescing_policy=self._coalescing_policy,
        )
