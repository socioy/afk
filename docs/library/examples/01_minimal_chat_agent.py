"""
Example 01: Minimal chat agent with one typed tool.

Run:
    uv run python docs/library/examples/01_minimal_chat_agent.py
"""

from __future__ import annotations

import asyncio
import os

from pydantic import BaseModel, Field

from afk.agents import Agent
from afk.tools import tool


class SumArgs(BaseModel):
    numbers: list[float] = Field(min_length=1, max_length=50)


@tool(args_model=SumArgs, name="sum_numbers")
def sum_numbers(args: SumArgs) -> dict[str, float]:
    return {"sum": float(sum(args.numbers))}


async def main() -> None:
    model_name = os.getenv("AFK_LLM_MODEL", "gpt-4.1-mini")

    agent = Agent(
        name="MathTutor",
        model=model_name,
        instructions=(
            "You are a math tutor. Explain your work clearly. "
            "Use sum_numbers whenever arithmetic is needed."
        ),
        tools=[sum_numbers],
    )

    result = await agent.call(
        "Please add 2.5, 8, and -1. Then explain the answer in one sentence.",
        context={"user_id": "demo-user"},
    )

    print("model:", model_name)
    print("state:", result.state)
    print("run_id:", result.run_id)
    print("thread_id:", result.thread_id)
    print("final_text:", result.final_text)
    print("tool_calls:", len(result.tool_executions))


if __name__ == "__main__":
    asyncio.run(main())
