from __future__ import annotations
"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.

This module implements the ToolRegistry for AFK.
It supports registering sync/async tools (tools are executed async via Tool/BaseTool),
optional entry-point plugin discovery, concurrency limiting, allow/deny policies,
registry-level middlewares (wrap ALL tools), and exporting tool specs to LLM tool-calling formats.
"""

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence

try:
    # Python 3.10+
    from importlib import metadata as importlib_metadata
except Exception:  # pragma: no cover
    import importlib_metadata  # type: ignore

from .base import Tool, ToolContext, ToolResult, ToolSpec, as_async 

from .errors import (
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
    ToolPolicyError,
    ToolTimeoutError,
)

ToolPolicy = Callable[[str, Dict[str, Any], ToolContext], None]


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    tool_name: str
    started_at_s: float
    ended_at_s: float
    ok: bool
    error: Optional[str] = None
    tool_call_id: Optional[str] = None


# ---------- Registry-level middleware types ----------

RegistryCallNext = Callable[
    [Tool[Any, Any], Dict[str, Any], ToolContext, Optional[float], Optional[str]],
    Awaitable[ToolResult[Any]],
]
RegistryMiddlewareFn = Callable[..., Any]  # sync or async; we wrap via as_async


def _infer_registry_middleware_style(fn: Callable[..., Any]) -> str:
    """
    Supported registry middleware signatures (sync OR async):

      (call_next, tool, raw_args, ctx)
      (call_next, tool, raw_args, ctx, timeout, tool_call_id)
      (tool, raw_args, ctx, call_next)
      (tool, raw_args, ctx, call_next, timeout, tool_call_id)

    We keep it strict (no *args/**kwargs) so it’s predictable and fast.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params):
        raise ValueError(
            f"Registry middleware '{getattr(fn, '__name__', 'unknown')}' cannot use *args/**kwargs."
        )

    if len(params) == 4:
        # try to detect whether call_next is first or last
        if params[0].name in ("call_next", "next") or "next" in params[0].name:
            return "next_tool_args_ctx"
        if params[3].name in ("call_next", "next") or "next" in params[3].name:
            return "tool_args_ctx_next"
        # fallback: assume first is next
        return "next_tool_args_ctx"

    if len(params) == 6:
        if params[0].name in ("call_next", "next") or "next" in params[0].name:
            return "next_tool_args_ctx_timeout_id"
        if params[3].name in ("call_next", "next") or "next" in params[3].name:
            return "tool_args_ctx_next_timeout_id"
        return "next_tool_args_ctx_timeout_id"

    raise ValueError(
        f"Registry middleware '{getattr(fn, '__name__', 'unknown')}' must have 4 or 6 parameters. Got {sig}"
    )


class RegistryMiddleware:
    """
    Wraps ALL tool calls executed through the registry.

    Use this for:
      - logging / tracing
      - global rate-limits / budgets
      - common retries
      - redaction / argument normalization
      - tenant-wide policies (in addition to policy hook)

    Supported fn signatures (sync OR async):

      (call_next, tool, raw_args, ctx)
      (call_next, tool, raw_args, ctx, timeout, tool_call_id)
      (tool, raw_args, ctx, call_next)
      (tool, raw_args, ctx, call_next, timeout, tool_call_id)

    call_next: async (tool, raw_args, ctx, timeout, tool_call_id) -> ToolResult
    """

    def __init__(self, fn: RegistryMiddlewareFn, *, name: str | None = None) -> None:
        self.name = name or getattr(fn, "__name__", "registry_middleware")
        self._original_fn = fn
        self.fn = as_async(fn)  # makes sync middleware work too
        self._style = _infer_registry_middleware_style(fn)

    async def __call__(
        self,
        call_next: RegistryCallNext,
        tool: Tool[Any, Any],
        raw_args: Dict[str, Any],
        ctx: ToolContext,
        timeout: float | None,
        tool_call_id: str | None,
    ) -> ToolResult[Any]:
        if self._style == "next_tool_args_ctx":
            return await self.fn(call_next, tool, raw_args, ctx)
        if self._style == "tool_args_ctx_next":
            return await self.fn(tool, raw_args, ctx, call_next)
        if self._style == "next_tool_args_ctx_timeout_id":
            return await self.fn(call_next, tool, raw_args, ctx, timeout, tool_call_id)
        # tool_args_ctx_next_timeout_id
        return await self.fn(tool, raw_args, ctx, call_next, timeout, tool_call_id)


# ---------- ToolRegistry ----------

class ToolRegistry:
    """
    Stores tools by name and provides safe async execution with:
      - concurrency limiting
      - registry-level default timeout
      - optional policy hook
      - optional plugin discovery via entry points
      - registry-level middlewares (wrap ALL calls)
      - tool spec export for LLM tool-calling
    """

    def __init__(
        self,
        *,
        max_concurrency: int = 32,
        default_timeout: float | None = None,
        policy: ToolPolicy | None = None,
        enable_plugins: bool = False,
        plugin_entry_point_group: str = "afk.tools",
        allow_overwrite_plugins: bool = False,
        middlewares: Optional[List[RegistryMiddleware | RegistryMiddlewareFn]] = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

        self._tools: Dict[str, Tool[Any, Any]] = {}
        self._sem = asyncio.Semaphore(max_concurrency)
        self._default_timeout = default_timeout
        self._policy = policy
        self._records: List[ToolCallRecord] = []
        self._middlewares: List[RegistryMiddleware] = []

        if middlewares:
            for mw in middlewares:
                self.add_middleware(mw)

        if enable_plugins:
            self.load_plugins(
                entry_point_group=plugin_entry_point_group,
                overwrite=allow_overwrite_plugins,
            )

    # ''''''''''''''''''''''''''''''''''''''
    # Registration / discovery
    # ''''''''''''''''''''''''''''''''''''''

    def register(self, tool: Tool[Any, Any], *, overwrite: bool = False) -> None:
        name = tool.spec.name
        if not overwrite and name in self._tools:
            raise ToolAlreadyRegisteredError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def register_many(self, tools: Iterable[Tool[Any, Any]], *, overwrite: bool = False) -> None:
        for t in tools:
            self.register(t, overwrite=overwrite)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool[Any, Any]:
        try:
            return self._tools[name]
        except KeyError as e:
            raise ToolNotFoundError(f"Unknown tool: {name}") from e

    def list(self) -> List[Tool[Any, Any]]:
        return list(self._tools.values())

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def has(self, name: str) -> bool:
        return name in self._tools

    def load_plugins(self, *, entry_point_group: str = "afk.tools", overwrite: bool = False) -> int:
        """
        Load Tool objects (or factories returning Tool) from Python entry points.

        Plugin pyproject.toml example:
          [project.entry-points."afk.tools"]
          my_tool = "my_pkg.tools:my_tool"  # Tool instance OR factory returning Tool

        Returns number of tools loaded.
        """
        eps = importlib_metadata.entry_points()
        group = eps.select(group=entry_point_group)

        loaded = 0
        for ep in group:
            obj = ep.load()

            tool_obj: Tool[Any, Any] | None = None
            if isinstance(obj, Tool):
                tool_obj = obj
            elif callable(obj):
                maybe = obj()
                if isinstance(maybe, Tool):
                    tool_obj = maybe

            if tool_obj is None:
                continue

            self.register(tool_obj, overwrite=overwrite)
            loaded += 1

        return loaded

    # ''''''''''''''''''''''''''''''''''''''
    # Registry middlewares
    # ''''''''''''''''''''''''''''''''''''''

    def add_middleware(self, mw: RegistryMiddleware | RegistryMiddlewareFn) -> None:
        """
        Add a registry-level middleware.
        You can pass a RegistryMiddleware instance OR a raw (sync/async) callable.
        """
        if isinstance(mw, RegistryMiddleware):
            self._middlewares.append(mw)
        else:
            self._middlewares.append(RegistryMiddleware(mw))

    def set_middlewares(self, mws: List[RegistryMiddleware | RegistryMiddlewareFn]) -> None:
        self._middlewares = []
        for mw in mws:
            self.add_middleware(mw)

    def clear_middlewares(self) -> None:
        self._middlewares = []

    def list_middlewares(self) -> List[str]:
        return [mw.name for mw in self._middlewares]

    # ''''''''''''''''''''''''''''''''''''''
    # Execution
    # ''''''''''''''''''''''''''''''''''''''

    async def call(
        self,
        name: str,
        raw_args: Dict[str, Any],
        *,
        ctx: ToolContext | None = None,
        timeout: float | None = None,
        tool_call_id: str | None = None,
    ) -> ToolResult[Any]:
        """
        Execute a registered tool by name.

        Timeout precedence:
          1) call(timeout=...)
          2) tool.default_timeout
          3) registry default_timeout
        """
        tool = self.get(name)
        ctx = ctx or ToolContext()

        # Policy hook (permissions/budget/allowlist)
        if self._policy is not None:
            try:
                self._policy(name, raw_args, ctx)
            except ToolPolicyError:
                raise
            except Exception as e:
                raise ToolPolicyError(str(e)) from e

        started = time.time()

        async with self._sem:
            effective_timeout = (
                timeout
                if timeout is not None
                else (tool.default_timeout if tool.default_timeout is not None else self._default_timeout)
            )

            async def _core_call(
                t: Tool[Any, Any],
                a: Dict[str, Any],
                c: ToolContext,
                to: float | None,
                tcid: str | None,
            ) -> ToolResult[Any]:
                # NOTE: we intentionally pass timeout=None into Tool.call here,
                # because we enforce registry-level timeout with asyncio.wait_for below
                # to cover the entire middleware + tool stack.
                if to is None:
                    return await t.call(a, ctx=c, timeout=None, tool_call_id=tcid)

                try:
                    return await asyncio.wait_for(
                        t.call(a, ctx=c, timeout=None, tool_call_id=tcid),
                        timeout=to,
                    )
                except asyncio.TimeoutError as e:
                    raise ToolTimeoutError(f"Tool '{t.spec.name}' timed out after {to} seconds.") from e

            # Wrap with registry-level middleware chain
            call_next: RegistryCallNext = _core_call
            for mw in reversed(self._middlewares):
                prev = call_next

                async def _wrapped(
                    t: Tool[Any, Any],
                    a: Dict[str, Any],
                    c: ToolContext,
                    to: float | None,
                    tcid: str | None,
                    _mw=mw,
                    _prev=prev,
                ) -> ToolResult[Any]:
                    return await _mw(_prev, t, a, c, to, tcid)

                call_next = _wrapped

            try:
                res = await call_next(tool, raw_args, ctx, effective_timeout, tool_call_id)

                self._records.append(
                    ToolCallRecord(
                        tool_name=name,
                        started_at_s=started,
                        ended_at_s=time.time(),
                        ok=res.success,
                        error=res.error_message,
                        tool_call_id=tool_call_id,
                    )
                )
                return res

            except Exception as e:
                self._records.append(
                    ToolCallRecord(
                        tool_name=name,
                        started_at_s=started,
                        ended_at_s=time.time(),
                        ok=False,
                        error=str(e),
                        tool_call_id=tool_call_id,
                    )
                )
                raise

    async def call_many(
        self,
        calls: Sequence[tuple[str, Dict[str, Any]]],
        *,
        ctx: ToolContext | None = None,
        timeout: float | None = None,
        tool_call_id_prefix: str | None = None,
        return_exceptions: bool = False,
    ) -> List[ToolResult[Any] | Exception]:
        """
        Execute multiple tool calls concurrently (bounded by registry semaphore).

        calls: list of (tool_name, raw_args)
        return_exceptions:
          - False: will raise on first exception
          - True: returns Exception objects in result list
        """
        ctx = ctx or ToolContext()

        async def _one(i: int, n: str, a: Dict[str, Any]) -> ToolResult[Any]:
            tcid = f"{tool_call_id_prefix}:{i}" if tool_call_id_prefix else None
            return await self.call(n, a, ctx=ctx, timeout=timeout, tool_call_id=tcid)

        tasks = [asyncio.create_task(_one(i, n, a)) for i, (n, a) in enumerate(calls)]
        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        return results  # type: ignore[return-value]

    # ''''''''''''''''''''''''''''''''''''''
    # Observability
    # ''''''''''''''''''''''''''''''''''''''

    def recent_calls(self, limit: int = 100) -> List[ToolCallRecord]:
        return self._records[-limit:]

    # ''''''''''''''''''''''''''''''''''''''
    # Export / specs
    # ''''''''''''''''''''''''''''''''''''''

    def specs(self) -> List[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def to_openai_function_tools(self) -> List[Dict[str, Any]]:
        """
        Export registry tools in OpenAI function-tool format:
        [
          {"type":"function","function":{"name":...,"description":...,"parameters":...}},
          ...
        ]
        """
        out: List[Dict[str, Any]] = []
        for t in self._tools.values():
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.spec.name,
                        "description": t.spec.description,
                        "parameters": t.spec.parameters_schema,
                    },
                }
            )
        return out

    def list_tool_summaries(self) -> List[Dict[str, Any]]:
        """
        Lightweight listing for UIs / debugging.
        """
        return [{"name": t.spec.name, "description": t.spec.description} for t in self._tools.values()]