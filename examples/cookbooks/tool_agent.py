"""
tool_agent.py — Agent with custom tools.

Shows how to create tools using the @tool decorator and attach them to an agent.

Usage:
    export AFK_LLM_API_KEY=sk-...
    python examples/tool_agent.py
"""

from afk.quickstart import Agent, Runner, ToolResult, tool


@tool(name="add", description="Add two numbers together.")
def add(a: float, b: float) -> ToolResult[float]:
    """Return the sum of a and b."""
    return ToolResult(output=a + b)


@tool(name="multiply", description="Multiply two numbers.")
def multiply(a: float, b: float) -> ToolResult[float]:
    """Return the product of a and b."""
    return ToolResult(output=a * b)


async def main() -> None:
    agent = Agent(
        name="calculator",
        model="gpt-4.1-mini",
        instructions="You are a calculator assistant. Use your tools to compute answers.",
        tools=[add, multiply],
    )

    runner = Runner()
    result = await runner.run(
        agent,
        user_message="What is 17 * 23 + 42?",
    )
    print(f"Answer: {result.final_text}")
    print(f"Tools used: {[t.tool_name for t in result.tool_executions]}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
