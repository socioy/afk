"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Provider-facing tool export utilities.

This module converts AFK tool/tool-spec objects into provider-compatible
function tool definitions used by OpenAI-compatible transports (OpenAI,
LiteLLM, etc.).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def normalize_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure schema is a safe object-parameter schema.

    Coerces invalid/malformed fields to predictable defaults so providers
    always receive a well-formed function-parameters schema.
    """
    if not isinstance(schema, dict):
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    out = dict(schema)

    # Tool parameters should always be object-shaped.
    out["type"] = "object"

    properties_raw = out.get("properties")
    if not isinstance(properties_raw, dict):
        properties: dict[str, Any] = {}
    else:
        properties = {
            str(key): (value if isinstance(value, dict) else {})
            for key, value in properties_raw.items()
        }
    out["properties"] = properties

    required_raw = out.get("required")
    if isinstance(required_raw, list):
        required = [
            str(name)
            for name in required_raw
            if isinstance(name, str) and name in properties
        ]
    else:
        required = []
    out["required"] = required

    additional_properties = out.get("additionalProperties")
    if not isinstance(additional_properties, (bool, dict)):
        out["additionalProperties"] = False

    return out


def toolspec_to_openai_tool(spec: Any) -> dict[str, Any]:
    """
    Convert a tool spec-like object into an OpenAI-compatible function tool.

    Required attributes on `spec`:
      - name: str
      - description: str
      - parameters_schema: dict
    """
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": normalize_json_schema(spec.parameters_schema),
        },
    }


def tool_to_openai_tool(tool: Any) -> dict[str, Any]:
    """Convert a tool-like object with `.spec` into a function tool."""
    return toolspec_to_openai_tool(tool.spec)


def to_openai_tools(tools: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert tool-like objects into OpenAI-compatible function tools."""
    return [tool_to_openai_tool(tool) for tool in tools]


def to_openai_tools_from_specs(specs: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert tool-spec-like objects into OpenAI-compatible function tools."""
    return [toolspec_to_openai_tool(spec) for spec in specs]


def export_tools_for_provider(
    tools: Iterable[Any],
    *,
    format: str = "openai",
) -> list[dict[str, Any]]:
    """
    Generic export entrypoint for OpenAI-compatible function tool payloads.

    Supported formats:
      - "openai" (default)
      - "litellm" (same schema)
      - "function"
      - "openai_function"
    """
    fmt = format.lower().strip()
    if fmt in ("openai", "litellm", "function", "openai_function"):
        return to_openai_tools(tools)
    raise ValueError(f"Unknown export format: {format}")
