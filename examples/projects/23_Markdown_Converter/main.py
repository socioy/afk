"""
---
name: Markdown Converter
description: A markdown converter agent that uses tool-level @middleware for logging, timing, and error wrapping around every tool call.
tags: [agent, runner, tools, middleware, registry-middleware]
---
---
This example demonstrates two levels of middleware in the AFK tools system:
1. **Tool-level @middleware** — wraps a specific tool with logic that runs before and after it.
2. **@registry_middleware** — wraps ALL tools in a ToolRegistry, applying cross-cutting concerns globally.

Middlewares are ideal for logging, timing, rate limiting, caching, error wrapping, and other
concerns that shouldn't live in the tool's core logic. The markdown converter agent transforms
text between formats, with middleware that times every call and logs inputs/outputs.
---
"""

import time  # <- For timing tool execution in middleware.
from pydantic import BaseModel, Field  # <- Pydantic for typed argument schemas.
from afk.core import Runner  # <- Runner executes agents.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.
from afk.tools import tool, middleware, registry_middleware, ToolRegistry, ToolContext  # <- @middleware for tool-level wrapping, @registry_middleware for registry-level wrapping.


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class ConvertArgs(BaseModel):  # <- Schema for text conversion tools.
    text: str = Field(description="The text to convert")


class TableArgs(BaseModel):  # <- Schema for markdown table generation.
    headers: list[str] = Field(description="Column headers for the table")
    rows: list[list[str]] = Field(description="Rows of data, each row is a list of cell values")


class EmptyArgs(BaseModel):
    pass


# ===========================================================================
# Logging list — middleware will write here for observability
# ===========================================================================

call_log: list[dict] = []  # <- A simple in-memory log that middleware writes to. In production you'd use a proper logger or telemetry system.


# ===========================================================================
# Tool-level middleware (wraps a specific tool)
# ===========================================================================

@middleware(name="timing_middleware", description="Measures execution time of the wrapped tool")
async def timing_middleware(call_next, args, ctx: ToolContext):  # <- A tool-level middleware. It receives call_next (the next handler in the chain), the tool's args, and an optional ToolContext. You MUST call call_next(args) to execute the actual tool — skipping it short-circuits the chain.
    start = time.monotonic()  # <- Record start time before calling the tool.
    result = await call_next(args)  # <- call_next(args) invokes the next middleware in the chain, or the tool itself if this is the last middleware. You can modify args before passing them, or modify the result after.
    elapsed = time.monotonic() - start
    call_log.append({  # <- Log the timing data.
        "tool": "timed_tool",
        "elapsed_ms": round(elapsed * 1000, 2),
        "success": result.success if hasattr(result, "success") else True,
    })
    print(f"  [middleware] tool executed in {elapsed * 1000:.1f}ms")  # <- Print timing info inline for demo visibility.
    return result  # <- Return the result unchanged. You could also modify it here (e.g., add metadata).


@middleware(name="input_logger", description="Logs tool input arguments before execution")
async def input_logger(call_next, args):  # <- Another middleware — this one logs inputs. Middlewares are composable: you can attach multiple to a single tool and they chain in order.
    print(f"  [middleware] input: {args}")  # <- Show what the tool received.
    return await call_next(args)  # <- Pass through to the next handler.


# ===========================================================================
# Registry-level middleware (wraps ALL tools in a registry)
# ===========================================================================

@registry_middleware(name="global_call_counter", description="Counts total tool calls across all tools in the registry")
async def global_call_counter(call_next, tool_obj, raw_args, ctx):  # <- A registry middleware receives: call_next, the Tool object, raw_args dict, and ToolContext. It wraps EVERY tool registered in the registry.
    tool_name = tool_obj.spec.name  # <- Access the tool's name from its spec.
    print(f"  [registry-mw] calling tool: {tool_name}")
    call_log.append({"event": "call_start", "tool": tool_name})  # <- Log every call for the session summary.
    result = await call_next(tool_obj, raw_args, ctx)  # <- Call the next handler. For registry middleware, pass through all original arguments.
    call_log.append({"event": "call_end", "tool": tool_name})
    return result


# ===========================================================================
# Tool definitions
# ===========================================================================

@tool(
    args_model=ConvertArgs,
    name="text_to_markdown_heading",
    description="Convert plain text into a markdown heading (H1-H3 based on length)",
    middlewares=[timing_middleware, input_logger],  # <- Attach tool-level middlewares. They run in order: timing_middleware wraps input_logger wraps the tool. So timing captures the full execution including input_logger.
)
def text_to_markdown_heading(args: ConvertArgs) -> str:  # <- The tool's core logic is clean — no logging or timing clutter.
    text = args.text.strip()
    if len(text) < 20:
        return f"# {text}"  # <- Short text gets H1.
    elif len(text) < 50:
        return f"## {text}"  # <- Medium text gets H2.
    return f"### {text}"  # <- Long text gets H3.


@tool(
    args_model=ConvertArgs,
    name="text_to_bullet_list",
    description="Convert newline-separated text into a markdown bullet list",
    middlewares=[timing_middleware],
)
def text_to_bullet_list(args: ConvertArgs) -> str:
    lines = [line.strip() for line in args.text.strip().split("\n") if line.strip()]
    if not lines:
        return "No items to convert."
    return "\n".join(f"- {line}" for line in lines)


@tool(
    args_model=ConvertArgs,
    name="text_to_code_block",
    description="Wrap text in a markdown code block with optional language detection",
    middlewares=[timing_middleware],
)
def text_to_code_block(args: ConvertArgs) -> str:
    text = args.text.strip()
    # --- Simple language detection ---
    lang = ""
    if "def " in text or "import " in text:
        lang = "python"
    elif "function " in text or "const " in text or "=>" in text:
        lang = "javascript"
    elif "SELECT " in text.upper() or "FROM " in text.upper():
        lang = "sql"
    return f"```{lang}\n{text}\n```"


@tool(
    args_model=TableArgs,
    name="create_markdown_table",
    description="Create a formatted markdown table from headers and row data",
    middlewares=[timing_middleware],
)
def create_markdown_table(args: TableArgs) -> str:
    if not args.headers:
        return "No headers provided."
    header_row = "| " + " | ".join(args.headers) + " |"
    separator = "| " + " | ".join("---" for _ in args.headers) + " |"
    data_rows = []
    for row in args.rows:
        padded = row + [""] * (len(args.headers) - len(row))  # <- Pad short rows with empty cells.
        data_rows.append("| " + " | ".join(padded[:len(args.headers)]) + " |")
    return "\n".join([header_row, separator] + data_rows)


@tool(args_model=ConvertArgs, name="text_to_blockquote", description="Convert text into a markdown blockquote")
def text_to_blockquote(args: ConvertArgs) -> str:
    lines = args.text.strip().split("\n")
    return "\n".join(f"> {line}" for line in lines)


# ===========================================================================
# ToolRegistry with global middleware
# ===========================================================================

registry = ToolRegistry(  # <- Create a registry and attach the global middleware. This middleware wraps ALL tools registered here.
    middlewares=[global_call_counter],  # <- Registry-level middleware applies to every tool call through this registry. Combine with tool-level middleware for layered cross-cutting concerns.
)

registry.register(text_to_markdown_heading)  # <- Register all tools in the registry.
registry.register(text_to_bullet_list)
registry.register(text_to_code_block)
registry.register(create_markdown_table)
registry.register(text_to_blockquote)


# ===========================================================================
# Agent and runner setup
# ===========================================================================

markdown_agent = Agent(
    name="markdown-converter",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a markdown formatting assistant. You help users convert plain text into
    well-formatted markdown.

    Available conversions:
    - Headings: Convert text to H1/H2/H3 based on length
    - Bullet lists: Convert newline-separated items to bullets
    - Code blocks: Wrap code with language detection
    - Tables: Create tables from structured data
    - Blockquotes: Convert text to blockquotes

    When the user provides text, choose the most appropriate conversion. If unclear, ask
    what format they want. Show the raw markdown output so they can copy it.

    **NOTE**: Always show the converted markdown clearly!
    """,
    tools=registry.list(),  # <- Pull tools from the registry.
)

runner = Runner()


if __name__ == "__main__":
    print("Markdown Converter Agent (type 'quit' to exit)")
    print("=" * 50)
    print("Convert text to markdown headings, lists, code blocks, tables, or blockquotes.\n")

    while True:
        user_input = input("[] > ")

        if user_input.strip().lower() in ("quit", "exit", "q"):
            # --- Show middleware call log ---
            if call_log:
                print(f"\n--- Session Stats ({len(call_log)} middleware events) ---")
                tool_calls = [e for e in call_log if e.get("event") == "call_start"]
                print(f"Total tool calls: {len(tool_calls)}")
                for tc in tool_calls:
                    print(f"  - {tc['tool']}")
            print("Goodbye!")
            break

        response = runner.run_sync(markdown_agent, user_message=user_input)
        print(f"[markdown-converter] > {response.final_text}\n")



"""
---
Tl;dr: This example creates a markdown converter agent with two levels of middleware: tool-level @middleware
(timing and input logging attached to specific tools) and @registry_middleware (global call counting applied
to all tools). Middlewares wrap tool execution with call_next patterns, enabling logging, timing, caching,
and other cross-cutting concerns without polluting core tool logic. Multiple middlewares chain in order.
---
---
What's next?
- Try adding a caching middleware that returns cached results for repeated inputs.
- Create a rate-limiting registry middleware that throttles tool calls.
- Experiment with middleware that modifies the tool's result (e.g. adding watermarks to all output).
- Chain multiple tool-level middlewares on a single tool and observe the execution order.
- Combine prehooks, posthooks, AND middleware on the same tool to see the full execution lifecycle.
- Check out the other examples for PolicyEngine-based tool gating and approval workflows!
---
"""
