"""
Example 07: Tool hooks and tool-level middleware.

Run:
    uv run python docs/library/examples/07_tool_hooks_and_middleware.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from afk.tools import ToolContext, middleware, posthook, prehook, tool


class EchoArgs(BaseModel):
    text: str = Field(min_length=1, max_length=200)


class PostArgs(BaseModel):
    output: Any
    tool_name: str | None = None


@prehook(args_model=EchoArgs, name="trim_input")
def trim_input(args: EchoArgs) -> dict[str, str]:
    return {"text": args.text.strip()}


@middleware(name="append_suffix")
async def append_suffix(call_next, args: EchoArgs, ctx: ToolContext):
    _ = ctx
    out = await call_next(args, ctx)
    if isinstance(out, str):
        return out + "!"
    return out


@posthook(args_model=PostArgs, name="wrap_output")
def wrap_output(args: PostArgs) -> dict[str, Any]:
    return {
        "tool": args.tool_name,
        "value": args.output,
    }


@tool(
    args_model=EchoArgs,
    name="echo_clean",
    prehooks=[trim_input],
    middlewares=[append_suffix],
    posthooks=[wrap_output],
)
def echo_clean(args: EchoArgs) -> str:
    return args.text.upper()


async def main() -> None:
    result = await echo_clean.call({"text": "   hello tooling   "})
    print("success:", result.success)
    print("tool:", result.tool_name)
    print("output:", result.output)


if __name__ == "__main__":
    asyncio.run(main())
