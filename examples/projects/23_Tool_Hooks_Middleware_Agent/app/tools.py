"""Tool definitions demonstrating full hook and middleware stack."""

from time import perf_counter

from afk.tools import ToolContext, middleware, posthook, prehook, tool

from .models import DraftReplyArgs, PostPayload


@prehook(args_model=DraftReplyArgs, name="normalize_input")
def normalize_input(args: DraftReplyArgs) -> dict[str, str]:
    return {
        "channel": args.channel.strip().lower(),
        "message": args.message.strip(),
    }


@middleware(name="latency_wrapper")
async def latency_wrapper(call_next, args: DraftReplyArgs, ctx: ToolContext):
    started = perf_counter()
    output = await call_next(args, ctx)
    elapsed_ms = round((perf_counter() - started) * 1000.0, 2)
    return {
        "payload": output,
        "latency_ms": elapsed_ms,
        "request_id": ctx.request_id,
    }


@posthook(args_model=PostPayload, name="attach_contract")
def attach_contract(args: PostPayload) -> dict[str, object]:
    return {
        "tool": args.tool_name,
        "contract": "support.reply.v2",
        "result": args.output,
    }


@tool(
    args_model=DraftReplyArgs,
    name="draft_support_reply",
    prehooks=[normalize_input],
    middlewares=[latency_wrapper],
    posthooks=[attach_contract],
)
def draft_support_reply(args: DraftReplyArgs) -> str:
    return f"[{args.channel}] Acknowledged: {args.message}"
