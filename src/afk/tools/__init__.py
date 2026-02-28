"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

AFK Tools public API.

This package exposes:
- Core tool types (Tool, ToolSpec, ToolContext, ToolResult)
- Hooks & middleware types (PreHook, PostHook, Middleware)
- Decorators for authoring tools quickly (including registry-level middleware)
- ToolRegistry (+ RegistryMiddleware support)
- Export helpers for LiteLLM tool/function calling schemas
"""

from __future__ import annotations

from .core import (
    Middleware,
    PostHook,
    PreHook,
    Tool,
    ToolAlreadyRegisteredError,
    ToolContext,
    ToolDeferredHandle,
    ToolExecutionError,
    ToolFn,
    ToolNotFoundError,
    ToolPolicyError,
    ToolResult,
    ToolSpec,
    ToolTimeoutError,
    ToolValidationError,
    as_async,
    middleware,
    posthook,
    prehook,
    registry_middleware,
    tool,
)
from .prebuilts import build_runtime_tools, build_skill_tools
from .registry import (
    RegistryMiddleware,
    RegistryMiddlewareFn,
    ToolCallRecord,
    ToolRegistry,
)
from .security import (
    SandboxProfile,
    SandboxProfileProvider,
    SecretScopeProvider,
    apply_tool_output_limits,
    build_registry_output_limit_middleware,
    build_registry_sandbox_policy,
    resolve_sandbox_profile,
    validate_tool_args_against_sandbox,
)

__all__ = [
    # core
    "Tool",
    "ToolSpec",
    "ToolContext",
    "ToolResult",
    "ToolDeferredHandle",
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
    "SandboxProfile",
    "SandboxProfileProvider",
    "SecretScopeProvider",
    "validate_tool_args_against_sandbox",
    "build_registry_sandbox_policy",
    "build_registry_output_limit_middleware",
    "resolve_sandbox_profile",
    "apply_tool_output_limits",
    "build_skill_tools",
    "build_runtime_tools",
    # errors
    "ToolAlreadyRegisteredError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolPolicyError",
    "ToolTimeoutError",
    "ToolValidationError",
]
