"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

A2A protocol/auth/hosting exports for agent communication.
"""

from .auth import (
    A2AAuthContext,
    A2AAuthError,
    A2AAuthorizationDecision,
    A2AAuthorizationError,
    A2AAuthProvider,
    A2APrincipal,
    AllowAllA2AAuthProvider,
    APIKeyA2AAuthProvider,
    JWTA2AAuthProvider,
)
from .delivery import A2ADeliveryStore, InMemoryA2ADeliveryStore, RedisA2ADeliveryStore
from .google_adapter import GoogleA2AAdapterError, GoogleA2AProtocolAdapter
from .internal_protocol import InternalA2AEnvelope, InternalA2AProtocol
from .server import A2AServiceHost, A2AServiceHostError

__all__ = [
    "A2AAuthContext",
    "A2APrincipal",
    "A2AAuthorizationDecision",
    "A2AAuthProvider",
    "A2AAuthError",
    "A2AAuthorizationError",
    "AllowAllA2AAuthProvider",
    "APIKeyA2AAuthProvider",
    "JWTA2AAuthProvider",
    "A2ADeliveryStore",
    "InMemoryA2ADeliveryStore",
    "RedisA2ADeliveryStore",
    "GoogleA2AProtocolAdapter",
    "GoogleA2AAdapterError",
    "A2AServiceHost",
    "A2AServiceHostError",
    "InternalA2AEnvelope",
    "InternalA2AProtocol",
]
