"""
Example 06: Tool registry security with sandbox policy.

Run:
    uv run python docs/library/examples/06_tool_registry_security.py
"""

from __future__ import annotations

import asyncio
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


@tool(args_model=ReadArgs, name="safe_read_file")
def safe_read_file(args: ReadArgs, ctx: ToolContext) -> dict[str, str | bool]:
    _ = ctx
    target = Path(args.path).resolve()
    content = target.read_text(encoding="utf-8")
    truncated = len(content) > args.max_chars
    if truncated:
        content = content[: args.max_chars]
    return {"path": str(target), "content": content, "truncated": truncated}


async def main() -> None:
    cwd = Path.cwd()
    profile = SandboxProfile(
        profile_id="docs-only",
        allow_network=False,
        allow_command_execution=False,
        allowed_paths=[str(cwd / "docs")],
        max_output_chars=4000,
    )

    registry = ToolRegistry(
        policy=build_registry_sandbox_policy(profile=profile, cwd=cwd),
        middlewares=[build_registry_output_limit_middleware(profile=profile)],
    )
    registry.register(safe_read_file)

    allowed = await registry.call(
        "safe_read_file",
        {"path": str(cwd / "docs" / "library" / "index.md")},
    )
    print("profile_id:", profile.profile_id)
    print("allowed_success:", allowed.success)

    try:
        await registry.call("safe_read_file", {"path": str(cwd / "README.md")})
    except ToolPolicyError as exc:
        print("blocked_by_policy:", str(exc))


if __name__ == "__main__":
    asyncio.run(main())
