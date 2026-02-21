# 07_tool_hooks_and_middleware

Compact reference for 07_tool_hooks_and_middleware.

Source: `docs/library/snippets/07_tool_hooks_and_middleware.mdx`

````python 07_tool_hooks_and_middleware.py
"""
Example 07: Tool hooks and tool-level middleware.

Run:
    python 07_tool_hooks_and_middleware.py
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from afk.tools import ToolContext, middleware, posthook, prehook, tool

class EchoArgs(BaseModel):
    text: str = Field(min_length=1, max_length=200)

class PostArgs(BaseModel):
    output: Any
    tool_name: str | None = None

@prehook(
    args_model=EchoArgs,
    name="trim_input",
    description="Normalize input text by removing leading and trailing whitespace.",
)
def trim_input(args: EchoArgs) -> dict[str, str]:
    return {"text": args.text.strip()}

@middleware(
    name="append_suffix",
    description="Append punctuation to string outputs after tool execution.",
)
async def append_suffix(call_next, args: EchoArgs, ctx: ToolContext):
    _ = ctx
````

> Code block truncated to 40 lines. Source: `docs/library/snippets/07_tool_hooks_and_middleware.mdx`
