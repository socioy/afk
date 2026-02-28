"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Module: routing/defaults.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..types import LLMRequest
from .base import LLMRouter


@dataclass(slots=True)
class OrderedFallbackRouter(LLMRouter):
    """Default router honoring explicit `RoutePolicy.provider_order` first."""

    router_id: str = "ordered_fallback"

    def route(
        self,
        req: LLMRequest,
        *,
        available_providers: list[str],
        default_provider: str,
    ) -> list[str]:
        """Return deterministic provider order for one request."""
        requested = []
        if req.route_policy is not None:
            requested = [
                name.strip().lower()
                for name in req.route_policy.provider_order
                if isinstance(name, str) and name.strip()
            ]

        order: list[str] = []
        if requested:
            seed = [*requested, default_provider]
        else:
            seed = [default_provider]
        for name in seed:
            if name in available_providers and name not in order:
                order.append(name)
        return order
