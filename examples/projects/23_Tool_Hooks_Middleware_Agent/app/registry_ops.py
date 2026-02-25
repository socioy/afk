"""ToolRegistry setup with registry-level middleware instrumentation."""

from afk.tools import ToolContext, ToolRegistry, registry_middleware

from .tools import draft_support_reply


@registry_middleware(name="audit_registry_calls")
async def audit_registry_calls(call_next, tool_obj, raw_args, ctx, timeout, tool_call_id):
    result = await call_next(tool_obj, raw_args, ctx, timeout, tool_call_id)
    result.metadata["audited"] = True
    result.metadata["tool_call_id"] = tool_call_id
    return result


def build_registry() -> ToolRegistry:
    registry = ToolRegistry(middlewares=[audit_registry_calls])
    registry.register(draft_support_reply)
    return registry


async def run_once(channel: str, message: str):
    registry = build_registry()
    result = await registry.call(
        "draft_support_reply",
        {"channel": channel, "message": message},
        ctx=ToolContext(request_id="req-23"),
        tool_call_id="tc-23-1",
    )
    return result, registry.recent_calls(limit=1)
