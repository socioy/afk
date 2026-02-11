from __future__ import annotations

"""
MIT License
Copyright (c) 2026 socioy
See LICENSE file for full license text.

Tool export utilities for LiteLLM.

LiteLLM expects OpenAI-compatible tool definitions for tool/function calling.
This module exports AFK Tool/ToolSpec into LiteLLM-ready payloads.
"""

from typing import Any, Dict, Iterable, List

from .base import Tool, ToolSpec


def normalize_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pydantic v2's model_json_schema() is generally usable as-is.
    We ensure it is at least an object schema with 'properties' to avoid edge cases.
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    out = dict(schema)
    out.setdefault("type", "object")
    out.setdefault("properties", {})
    return out


def toolspec_to_litellm_tool(spec: ToolSpec) -> Dict[str, Any]:
    """
    Convert a ToolSpec into a LiteLLM tool definition.

    LiteLLM tool schema is OpenAI-compatible:
      {
        "type": "function",
        "function": {
          "name": "...",
          "description": "...",
          "parameters": { ...JSON Schema... }
        }
      }
    """
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": normalize_json_schema(spec.parameters_schema),
        },
    }


def tool_to_litellm_tool(tool: Tool[Any, Any]) -> Dict[str, Any]:
    return toolspec_to_litellm_tool(tool.spec)


def to_litellm_tools(tools: Iterable[Tool[Any, Any]]) -> List[Dict[str, Any]]:
    """
    Export Tool objects to a list of LiteLLM tool definitions.
    Pass this list as `tools=...` to `litellm.completion(...)`.
    """
    return [tool_to_litellm_tool(t) for t in tools]


def to_litellm_tools_from_specs(specs: Iterable[ToolSpec]) -> List[Dict[str, Any]]:
    """
    Export ToolSpec objects to a list of LiteLLM tool definitions.
    """
    return [toolspec_to_litellm_tool(s) for s in specs]


def export_tools(
    tools: Iterable[Tool[Any, Any]],
    *,
    format: str = "litellm",
) -> List[Dict[str, Any]]:
    """
    Generic export entrypoint.

    Supported formats:
      - "litellm" (default): OpenAI-compatible function tools for LiteLLM
      - "openai": alias of litellm (since the schema is identical)
    """
    fmt = format.lower().strip()
    if fmt in ("litellm", "openai", "function", "openai_function"):
        return to_litellm_tools(tools)
    raise ValueError(f"Unknown export format: {format}")
