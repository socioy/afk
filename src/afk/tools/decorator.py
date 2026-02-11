from __future__ import annotations

"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.

This module provides decorators for defining tools, prehooks, posthooks, and middlewares in a concise way.
It also supports registry-level middlewares via @registry_middleware.
"""

import inspect
from typing import Any, Callable, Optional, Type, TypeVar

from pydantic import BaseModel

from .base import (
    Middleware,
    PostHook,
    PreHook,
    Tool,
    ToolFn,
    ToolSpec,
)

# Registry-level middleware wrapper lives in registry.py
# (Avoid importing ToolRegistry here to prevent heavy imports.)
from .registry import RegistryMiddleware, RegistryMiddlewareFn  # noqa: E402


ArgsT = TypeVar("ArgsT", bound=BaseModel)
ReturnT = TypeVar("ReturnT")


def _default_description(fn: Callable[..., Any], fallback: str) -> str:
    doc = inspect.getdoc(fn) or ""
    first_line = doc.splitlines()[0].strip() if doc else ""
    return first_line or fallback


def tool(
    *,
    args_model: Type[ArgsT],
    name: str | None = None,
    description: str | None = None,
    timeout: float | None = None,
    prehooks: Optional[list[PreHook[Any, Any]]] = None,
    posthooks: Optional[list[PostHook]] = None,
    middlewares: Optional[list[Middleware[Any, Any]]] = None,
    raise_on_error: bool = False,
) -> Callable[[ToolFn], Tool[ArgsT, ReturnT]]:
    """
    Create an AFK Tool from a sync/async function and a Pydantic v2 args model.

    Tool function can be sync or async and should use one of:
      def/async def fn(args: ArgsModel) -> Any
      def/async def fn(args: ArgsModel, ctx: ToolContext) -> Any
      def/async def fn(ctx: ToolContext, args: ArgsModel) -> Any

    Hooks/middlewares are optional:
      - prehooks: list of PreHook that transform args (must return dict compatible with tool args_model)
      - posthooks: list of PostHook that can transform output
      - middlewares: list of Middleware that wrap execution

    raise_on_error:
      - False (default): Tool.call returns ToolResult(success=False) on failure
      - True: raises ToolExecutionError/ToolTimeoutError/ToolValidationError
    """

    def decorator(fn: ToolFn) -> Tool[ArgsT, ReturnT]:
        tool_name = name or getattr(fn, "__name__", "tool")
        tool_desc = description or _default_description(fn, tool_name)

        schema = args_model.model_json_schema()
        spec = ToolSpec(name=tool_name, description=tool_desc, parameters_schema=schema)

        return Tool(
            spec=spec,
            fn=fn,
            args_model=args_model,
            default_timeout=timeout,
            prehooks=prehooks,
            posthooks=posthooks,
            middlewares=middlewares,
            raise_on_error=raise_on_error,
        )

    return decorator


def prehook(
    *,
    args_model: Type[ArgsT],
    name: str | None = None,
    description: str | None = None,
    timeout: float | None = None,
    raise_on_error: bool = False,
) -> Callable[[ToolFn], PreHook[ArgsT, Any]]:
    """
    Create a PreHook from a sync/async function and a Pydantic v2 args model.

    PreHook function should return a dict of transformed args compatible with the MAIN tool args_model.
    """

    def decorator(fn: ToolFn) -> PreHook[ArgsT, Any]:
        hook_name = name or getattr(fn, "__name__", "prehook")
        hook_desc = description or _default_description(fn, hook_name)

        schema = args_model.model_json_schema()
        spec = ToolSpec(name=hook_name, description=hook_desc, parameters_schema=schema)

        return PreHook(
            spec=spec,
            fn=fn,
            args_model=args_model,
            default_timeout=timeout,
            raise_on_error=raise_on_error,
        )

    return decorator


def posthook(
    *,
    args_model: Type[ArgsT],
    name: str | None = None,
    description: str | None = None,
    timeout: float | None = None,
    raise_on_error: bool = False,
) -> Callable[[ToolFn], PostHook]:
    """
    Create a PostHook from a sync/async function and a Pydantic v2 args model.

    Recommended args model shape:
      class PostArgs(BaseModel):
          output: Any
          tool_name: str | None = None

    AFK passes posthooks a payload dict like:
      {"output": <tool_output>, "tool_name": "<tool_name>"}
    """

    def decorator(fn: ToolFn) -> PostHook:
        hook_name = name or getattr(fn, "__name__", "posthook")
        hook_desc = description or _default_description(fn, hook_name)

        schema = args_model.model_json_schema()
        spec = ToolSpec(name=hook_name, description=hook_desc, parameters_schema=schema)

        return PostHook(
            spec=spec,
            fn=fn,
            args_model=args_model,  # type: ignore[arg-type]
            default_timeout=timeout,
            raise_on_error=raise_on_error,
        )

    return decorator


def middleware(
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float | None = None,
) -> Callable[[ToolFn], Middleware[Any, Any]]:
    """
    Create a Middleware from a sync/async function (tool-level middleware).

    Supported signatures (sync or async):
      fn(call_next, args)
      fn(call_next, args, ctx)
      fn(args, ctx, call_next)
      fn(ctx, args, call_next)

    call_next is: async (args, ctx) -> output
    """

    def decorator(fn: ToolFn) -> Middleware[Any, Any]:
        mw_name = name or getattr(fn, "__name__", "middleware")
        mw_desc = description or _default_description(fn, mw_name)

        # Tool-level middleware doesn't have a strict args model; keep schema empty.
        spec = ToolSpec(name=mw_name, description=mw_desc, parameters_schema={})
        return Middleware(spec=spec, fn=fn, default_timeout=timeout)

    return decorator


def registry_middleware(
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[RegistryMiddlewareFn], RegistryMiddleware]:
    """
    Create a registry-level middleware wrapper.

    Supported signatures (sync OR async):

      (call_next, tool, raw_args, ctx)
      (call_next, tool, raw_args, ctx, timeout, tool_call_id)
      (tool, raw_args, ctx, call_next)
      (tool, raw_args, ctx, call_next, timeout, tool_call_id)

    call_next is: async (tool, raw_args, ctx, timeout, tool_call_id) -> ToolResult
    """

    def decorator(fn: RegistryMiddlewareFn) -> RegistryMiddleware:
        mw = RegistryMiddleware(fn, name=name or getattr(fn, "__name__", "registry_middleware"))
        # RegistryMiddleware currently doesn't store description; keep it attached for debugging/introspection
        if description:
            setattr(mw, "description", description)
        else:
            setattr(mw, "description", _default_description(fn, getattr(mw, "name", "registry_middleware")))
        return mw

    return decorator