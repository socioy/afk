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

from .core import (
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

from .core import (
    tool,
    prehook,
    posthook,
    middleware,
    registry_middleware,
)

from .core import (
    ToolAlreadyRegisteredError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPolicyError,
    ToolTimeoutError,
    ToolValidationError,
)

from .core import (
    export_tools,
    normalize_json_schema,
    to_litellm_tools,
    to_litellm_tools_from_specs,
    tool_to_litellm_tool,
    toolspec_to_litellm_tool,
)

from .registery import (
    ToolRegistry,
    RegistryMiddleware,
    RegistryMiddlewareFn,
    ToolCallRecord,
)
from .security import (
    SandboxProfileProvider,
    SandboxProfile,
    SecretScopeProvider,
    apply_tool_output_limits,
    build_registry_output_limit_middleware,
    build_registry_sandbox_policy,
    resolve_sandbox_profile,
    validate_tool_args_against_sandbox,
)
from .prebuilts import build_runtime_tools, build_skill_tools

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
