from __future__ import annotations

"""
Shared client-side normalization helpers used across LLM adapters.
"""

from dataclasses import asdict, is_dataclass
from typing import Any

from ...types import ToolCall, Usage
from ...utils import safe_json_loads


def to_plain_dict(value: Any) -> dict[str, Any]:
    """Best-effort conversion of SDK/provider objects into plain dictionaries."""
    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    if hasattr(value, "to_dict"):
        try:
            dumped = value.to_dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    if is_dataclass(value):
        try:
            dumped = asdict(value)
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            dumped = dict(value.__dict__)
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    return {}


def to_jsonable(value: Any) -> Any:
    """Recursively coerce values into JSON-serializable primitives/containers."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, list):
        return [to_jsonable(v) for v in value]

    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]

    if is_dataclass(value):
        return to_jsonable(asdict(value))

    as_dict = to_plain_dict(value)
    if as_dict:
        return to_jsonable(as_dict)

    return repr(value)


def extract_text_from_content(content: Any) -> str:
    """Extract plain text from common OpenAI/LiteLLM content shapes."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        out: list[str] = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
                continue

            if not isinstance(item, dict):
                continue

            text = item.get("text")
            if isinstance(text, str):
                out.append(text)
                continue

            if item.get("type") == "output_text":
                maybe = item.get("text")
                if isinstance(maybe, str):
                    out.append(maybe)
        return "".join(out)

    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text

    return ""


def extract_usage(raw_dict: dict[str, Any]) -> Usage:
    """Normalize usage token counters from provider payloads."""
    usage = raw_dict.get("usage")
    if not isinstance(usage, dict):
        return Usage()

    input_tokens = usage.get("prompt_tokens")
    if input_tokens is None:
        input_tokens = usage.get("input_tokens")

    output_tokens = usage.get("completion_tokens")
    if output_tokens is None:
        output_tokens = usage.get("output_tokens")

    total_tokens = usage.get("total_tokens")
    return Usage(
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        total_tokens=total_tokens if isinstance(total_tokens, int) else None,
    )


def extract_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
    """Extract normalized tool calls from chat completion payloads."""
    if not isinstance(raw_tool_calls, list):
        return []

    out: list[ToolCall] = []
    for item in raw_tool_calls:
        tc = to_plain_dict(item)
        function = tc.get("function")
        if not isinstance(function, dict):
            function = {}

        name = function.get("name")
        if not isinstance(name, str):
            name = ""

        args_obj: dict[str, Any] = {}
        raw_args = function.get("arguments")
        if isinstance(raw_args, dict):
            args_obj = raw_args
        elif isinstance(raw_args, str):
            parsed = safe_json_loads(raw_args)
            if isinstance(parsed, dict):
                args_obj = parsed

        call_id = tc.get("id") if isinstance(tc.get("id"), str) else None
        out.append(ToolCall(id=call_id, tool_name=name, arguments=args_obj))

    return out


def finalize_stream_tool_calls(tool_buffers: dict[int, dict[str, Any]]) -> list[ToolCall]:
    """Build final normalized tool calls from accumulated stream deltas."""
    out: list[ToolCall] = []
    for idx in sorted(tool_buffers.keys()):
        buf = tool_buffers[idx]
        args_str = "".join(buf.get("args_parts", []))
        parsed_args = safe_json_loads(args_str) if args_str else None
        out.append(
            ToolCall(
                id=buf.get("id") if isinstance(buf.get("id"), str) else None,
                tool_name=buf.get("name") if isinstance(buf.get("name"), str) else "",
                arguments=parsed_args if isinstance(parsed_args, dict) else {},
            )
        )
    return out


def get_attr(obj: Any, name: str) -> Any:
    """Safe getattr helper."""
    return getattr(obj, name, None)


def get_attr_str(obj: Any, name: str) -> str | None:
    """Safe getattr helper returning strings only."""
    value = getattr(obj, name, None)
    return value if isinstance(value, str) else None
