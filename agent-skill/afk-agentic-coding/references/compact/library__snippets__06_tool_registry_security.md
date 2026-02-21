# 06_tool_registry_security

Compact reference for 06_tool_registry_security.

Source: `docs/library/snippets/06_tool_registry_security.mdx`

````python 06_tool_registry_security.py
"""
Example 06: Tool registry security with sandbox policy.

Run:
    python 06_tool_registry_security.py
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from afk.tools import (
    SandboxProfile,
    ToolContext,
    ToolPolicyError,
    ToolRegistry,
    build_registry_output_limit_middleware,
    build_registry_sandbox_policy,
    tool,
)

class ReadArgs(BaseModel):
    path: str = Field(min_length=1)
    max_chars: int = Field(default=2000, ge=1, le=20_000)

@tool(
    args_model=ReadArgs,
    name="safe_read_file",
    description="Read a UTF-8 text file from an allowlisted path with max size bounds.",
)
def safe_read_file(args: ReadArgs, ctx: ToolContext) -> dict[str, str | bool]:
    _ = ctx
    target = Path(args.path).resolve()
    content = target.read_text(encoding="utf-8")
    truncated = len(content) > args.max_chars
    if truncated:
````

> Code block truncated to 40 lines. Source: `docs/library/snippets/06_tool_registry_security.mdx`
