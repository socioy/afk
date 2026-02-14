"""LLM client package.

Structure:
- `adapters/`: provider-specific adapter implementations
- `base/`: reusable adapter base classes
- `shared/`: reusable normalization/mapping utilities
"""

from .adapters import AnthropicAgentClient, LiteLLMClient, OpenAIClient
from .base import ResponsesClientBase

__all__ = [
    "ResponsesClientBase",
    "LiteLLMClient",
    "AnthropicAgentClient",
    "OpenAIClient",
]
