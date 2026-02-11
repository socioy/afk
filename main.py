import asyncio
from typing import Any

from pydantic import BaseModel, Field

# ---- Import fallback: use whichever matches your project structure ----
try:
    # If you really exposed as src.afk.tools (uncommon)
    from src.afk.tools import (
        ToolRegistry,
        ToolContext,
        ToolResult,
        tool,
        prehook,
        posthook,
        middleware,
        registry_middleware,
        to_litellm_tools,
        ToolPolicyError,
        ToolTimeoutError,
    )
except ModuleNotFoundError:
    # Typical Python packaging: src/afk/... => import afk.tools
    from afk.tools import (
        ToolRegistry,
        ToolContext,
        ToolResult,
        tool,
        prehook,
        posthook,
        middleware,
        registry_middleware,
        to_litellm_tools,
        ToolPolicyError,
        ToolTimeoutError,
    )


# -----------------------
# Models
# -----------------------


class AddArgs(BaseModel):
    a: int = Field(..., description="First integer")
    b: int = Field(0, description="Second integer")


class PostArgs(BaseModel):
    output: Any
    tool_name: str | None = None


# -----------------------
# Tool-level hooks/middleware
# -----------------------


@prehook(args_model=AddArgs, name="clamp_args")
def clamp_args(args: AddArgs) -> dict:
    # Ensure args are non-negative
    return {"a": max(0, args.a), "b": max(0, args.b)}


@posthook(args_model=PostArgs, name="double_output")
def double_output(args: PostArgs) -> Any:
    # If numeric, double it; else return as-is
    if isinstance(args.output, (int, float)):
        return args.output * 2
    return args.output


@middleware(name="tool_logger")
async def tool_logger(call_next, args: AddArgs, ctx: ToolContext):
    print(f"[tool-mw] before args={args.model_dump()} ctx.request_id={ctx.request_id}")
    out = await call_next(args, ctx)
    print(f"[tool-mw] after out={out}")
    return out


# -----------------------
# Tools
# -----------------------


@tool(
    args_model=AddArgs,
    name="add_sync",
    description="Adds two integers (sync tool).",
    prehooks=[clamp_args],
    posthooks=[double_output],
    middlewares=[tool_logger],
)
def add_sync(args: AddArgs) -> int:
    # Sync tool: should still work (auto-wrapped to async internally)
    return args.a + args.b


@tool(
    args_model=AddArgs,
    name="add_async",
    description="Adds two integers (async tool).",
    prehooks=[clamp_args],
    posthooks=[double_output],
    middlewares=[tool_logger],
)
async def add_async(args: AddArgs) -> int:
    await asyncio.sleep(0.01)
    return args.a + args.b


@tool(
    args_model=AddArgs,
    name="slow_tool",
    description="Sleeps then returns a+b (used to test timeouts).",
    timeout=0.05,  # tool-level default timeout
)
async def slow_tool(args: AddArgs) -> int:
    await asyncio.sleep(0.2)
    return args.a + args.b


# -----------------------
# Registry-level middleware
# -----------------------


@registry_middleware(name="registry_logger")
async def registry_logger(call_next, tool_obj, raw_args, ctx, timeout, tool_call_id):
    print(
        f"[reg-mw] -> {tool_obj.spec.name} raw_args={raw_args} timeout={timeout} tcid={tool_call_id}"
    )
    res = await call_next(tool_obj, raw_args, ctx, timeout, tool_call_id)
    print(
        f"[reg-mw] <- {tool_obj.spec.name} success={res.success} err={res.error_message}"
    )
    return res


@registry_middleware(name="block_negative_a")
def block_negative_a(call_next, tool_obj, raw_args, ctx, timeout, tool_call_id):
    # Demonstrates SYNC registry middleware (auto-wrapped)
    if isinstance(raw_args, dict) and raw_args.get("a", 0) < 0:
        return ToolResult(
            success=False,
            error_message="Blocked by registry middleware: a < 0",
            tool_name=tool_obj.spec.name,
        )
    return call_next(tool_obj, raw_args, ctx, timeout, tool_call_id)


# -----------------------
# Registry policy
# -----------------------


def policy(tool_name: str, raw_args: dict, ctx: ToolContext) -> None:
    # Example policy: block a specific tool for a specific user
    if ctx.user_id == "banned_user" and tool_name == "add_sync":
        raise ToolPolicyError("User is not allowed to call add_sync")


# -----------------------
# Main test runner
# -----------------------


async def main():
    reg = ToolRegistry(
        max_concurrency=8,
        default_timeout=1.0,
        policy=policy,
        middlewares=[registry_logger, block_negative_a],  # registry-level middlewares
    )

    # register tools
    reg.register(add_sync)
    reg.register(add_async)
    reg.register(slow_tool)

    ctx = ToolContext(request_id="req_001", user_id="u_123")

    print(
        "\n--- 1) Sync tool call (auto-wrapped) + pre/post hooks + tool middleware ---"
    )
    res1 = await reg.call("add_sync", {"a": -2, "b": 5}, ctx=ctx, tool_call_id="tc_1")
    print("result:", res1)

    print("\n--- 2) Async tool call + hooks/middleware ---")
    res2 = await reg.call("add_async", {"a": 3, "b": 4}, ctx=ctx, tool_call_id="tc_2")
    print("result:", res2)

    print(
        "\n--- 3) Registry middleware block test (a < 0 blocked before tool runs) ---"
    )
    res3 = await reg.call("add_async", {"a": -1, "b": 2}, ctx=ctx, tool_call_id="tc_3")
    print("result:", res3)

    print("\n--- 4) Policy block test (banned user) ---")
    banned_ctx = ToolContext(request_id="req_002", user_id="banned_user")
    try:
        await reg.call(
            "add_sync", {"a": 1, "b": 1}, ctx=banned_ctx, tool_call_id="tc_4"
        )
    except ToolPolicyError as e:
        print("policy blocked as expected:", e)

    print("\n--- 5) Tool timeout test (slow_tool has tool-level timeout=0.05) ---")
    try:
        await reg.call("slow_tool", {"a": 1, "b": 2}, ctx=ctx, tool_call_id="tc_5")
    except ToolTimeoutError as e:
        print("timeout raised as expected:", e)

    print("\n--- 6) call_many test (concurrent execution) ---")
    batch = [
        ("add_sync", {"a": 1, "b": 2}),
        ("add_async", {"a": 10, "b": 5}),
        ("add_async", {"a": -3, "b": 9}),  # will be blocked by registry middleware
        ("add_sync", {"a": 7, "b": 8}),
    ]
    results = await reg.call_many(
        batch, ctx=ctx, tool_call_id_prefix="batch", return_exceptions=True
    )
    for i, r in enumerate(results):
        print(f"batch[{i}] =>", r)

    print("\n--- 7) LiteLLM export payload test ---")
    litellm_tools_payload = to_litellm_tools(reg.list())
    # print a compact preview
    print("litellm tools payload (first item):", litellm_tools_payload[0])
    print("total tools exported:", len(litellm_tools_payload))

    print("\n--- 8) Recent calls record ---")
    for rec in reg.recent_calls(limit=20):
        print(rec)


if __name__ == "__main__":
    asyncio.run(main())
