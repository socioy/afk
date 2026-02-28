"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

External MCP server registry/store and AFK tool materialization helpers.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict

from afk.mcp.store.transport import MCPJsonRpcClient
from afk.mcp.store.types import (
    MCPRemoteCallError,
    MCPRemoteProtocolError,
    MCPRemoteTool,
    MCPServerRef,
    MCPServerResolutionError,
    MCPStoreError,
)
from afk.mcp.store.utils import (
    _extract_mcp_text,
    normalize_json_schema,
    normalize_remote_tools,
    resolve_server_ref,
)
from afk.tools import Tool, ToolContext, ToolSpec

__all__ = [
    "MCPServerRef",
    "MCPRemoteTool",
    "MCPStoreError",
    "MCPServerResolutionError",
    "MCPRemoteProtocolError",
    "MCPRemoteCallError",
    "MCPStore",
    "normalize_json_schema",
    "get_mcp_store",
    "reset_mcp_store",
]


class _MCPArgs(BaseModel):
    """Permissive args model used for dynamic remote MCP tools."""

    model_config = ConfigDict(extra="allow")


class MCPStore:
    """
    Process-wide registry for external MCP servers.

    The store resolves server references, caches remote tool metadata and can
    materialize those tools as AFK `Tool` instances so existing runtime
    orchestration (policy/sandbox/replay/fail-safe) continues to apply.
    """

    def __init__(self, *, client: MCPJsonRpcClient | None = None) -> None:
        self._lock = threading.RLock()
        self._servers: dict[str, MCPServerRef] = {}
        self._tool_cache: dict[str, list[MCPRemoteTool]] = {}
        self._client = client or MCPJsonRpcClient()

    def register_server(self, ref: MCPServerRef) -> None:
        with self._lock:
            existing = self._servers.get(ref.name)
            self._servers[ref.name] = ref
            if existing is None or existing != ref:
                self._tool_cache.pop(ref.name, None)

    def unregister_server(self, name: str) -> None:
        with self._lock:
            self._servers.pop(name, None)
            self._tool_cache.pop(name, None)

    def clear(self) -> None:
        with self._lock:
            self._servers.clear()
            self._tool_cache.clear()

    def resolve_server(self, ref: str | dict[str, Any] | MCPServerRef) -> MCPServerRef:
        """Resolve and register a server reference."""
        resolved = resolve_server_ref(ref)
        self.register_server(resolved)
        return resolved

    async def list_tools(
        self,
        ref: str | dict[str, Any] | MCPServerRef,
        *,
        refresh: bool = False,
    ) -> list[MCPRemoteTool]:
        """List remote MCP tools for one server, using cache by default."""
        server = self.resolve_server(ref)
        with self._lock:
            if not refresh and server.name in self._tool_cache:
                return list(self._tool_cache[server.name])

        response = await self._client.call(
            server,
            method="tools/list",
            params={},
        )
        normalized = normalize_remote_tools(server, response.get("tools"))

        with self._lock:
            self._tool_cache[server.name] = list(normalized)
        return normalized

    async def call_tool(
        self,
        ref: str | dict[str, Any] | MCPServerRef,
        *,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call one remote MCP tool and return raw MCP result payload."""
        server = self.resolve_server(ref)
        response = await self._client.call(
            server,
            method="tools/call",
            params={"name": tool_name, "arguments": arguments},
        )
        if not isinstance(response, dict):
            raise MCPRemoteProtocolError(
                f"Invalid tools/call response from '{server.name}' for '{tool_name}'"
            )

        if bool(response.get("isError")):
            message = (
                _extract_mcp_text(response.get("content")) or "Remote MCP tool error"
            )
            raise MCPRemoteCallError(f"{server.name}:{tool_name}: {message}")

        return response

    async def tools_from_servers(
        self,
        refs: Iterable[str | dict[str, Any] | MCPServerRef],
    ) -> list[Tool[Any, Any]]:
        """Materialize AFK tools from one or more external MCP servers."""
        tools: list[Tool[Any, Any]] = []
        for ref in refs:
            server = self.resolve_server(ref)
            remote_tools = await self.list_tools(server)
            for remote in remote_tools:
                invoke = self._make_remote_tool_fn(server=server, remote=remote)
                spec = ToolSpec(
                    name=remote.qualified_name,
                    description=remote.description,
                    parameters_schema=normalize_json_schema(remote.input_schema),
                )
                tools.append(
                    Tool(
                        spec=spec,
                        fn=invoke,
                        args_model=_MCPArgs,
                    )
                )
        return tools

    def _make_remote_tool_fn(
        self,
        *,
        server: MCPServerRef,
        remote: MCPRemoteTool,
    ):
        async def _invoke(args: _MCPArgs, ctx: ToolContext) -> Any:
            _ = ctx
            raw = args.model_dump(mode="python")
            result = await self.call_tool(
                server,
                tool_name=remote.name,
                arguments=raw,
            )
            text = _extract_mcp_text(result.get("content"))
            if text is not None:
                return text
            return result

        return _invoke


_MCP_STORE: MCPStore | None = None
_MCP_STORE_LOCK = threading.Lock()


def get_mcp_store() -> MCPStore:
    """Return process-wide MCP store singleton."""
    global _MCP_STORE
    if _MCP_STORE is not None:
        return _MCP_STORE
    with _MCP_STORE_LOCK:
        if _MCP_STORE is None:
            _MCP_STORE = MCPStore()
    return _MCP_STORE


def reset_mcp_store() -> None:
    """Reset MCP store singleton (for tests)."""
    global _MCP_STORE
    with _MCP_STORE_LOCK:
        _MCP_STORE = None
