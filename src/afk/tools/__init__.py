from __future__ import annotations

"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.

AFK Tools public API.

This package exposes:
- Core tool types (Tool, ToolSpec, ToolContext, ToolResult)
- Hooks & middleware types (PreHook, PostHook, Middleware)
- Decorators for authoring tools quickly (including registry-level middleware)
- ToolRegistry (+ RegistryMiddleware support)
- Export helpers for LiteLLM tool/function calling schemas
"""

from .base import (
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
    PreHook,
    PostHook,
    Middleware,
    ToolFn,
    as_async,
)

from .decorator import (
    tool,
    prehook,
    posthook,
    middleware,
    registry_middleware,
)

from .errors import (
    ToolAlreadyRegisteredError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPolicyError,
    ToolTimeoutError,
    ToolValidationError,
)

from .export import (
    export_tools,
    normalize_json_schema,
    to_litellm_tools,
    to_litellm_tools_from_specs,
    tool_to_litellm_tool,
    toolspec_to_litellm_tool,
)

from .registry import (
    ToolRegistry,
    RegistryMiddleware,
    RegistryMiddlewareFn,
    ToolCallRecord,
)

__all__ = [
    # core
    "Tool",
    "ToolSpec",
    "ToolContext",
    "ToolResult",
    "ToolFn",
    "as_async",
    # hooks/middleware
    "PreHook",
    "PostHook",
    "Middleware",
    "RegistryMiddleware",
    "RegistryMiddlewareFn",
    # decorators
    "tool",
    "prehook",
    "posthook",
    "middleware",
    "registry_middleware",
    # registry
    "ToolRegistry",
    "ToolCallRecord",
    # export (LiteLLM)
    "export_tools",
    "normalize_json_schema",
    "to_litellm_tools",
    "to_litellm_tools_from_specs",
    "tool_to_litellm_tool",
    "toolspec_to_litellm_tool",
    # errors
    "ToolAlreadyRegisteredError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolPolicyError",
    "ToolTimeoutError",
    "ToolValidationError",
]
