"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Shared client helper utilities.
"""

from .content import (
    json_text,
    normalize_role,
    to_input_text_part,
    tool_result_label,
)
from .normalization import (
    extract_text_from_content,
    extract_tool_calls,
    extract_usage,
    finalize_stream_tool_calls,
    get_attr,
    get_attr_str,
    to_jsonable,
    to_plain_dict,
)
from .transport import collect_headers

__all__ = [
    "to_plain_dict",
    "to_jsonable",
    "extract_text_from_content",
    "extract_usage",
    "extract_tool_calls",
    "finalize_stream_tool_calls",
    "get_attr",
    "get_attr_str",
    "json_text",
    "normalize_role",
    "tool_result_label",
    "to_input_text_part",
    "collect_headers",
]
