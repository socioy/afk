# 01_minimal_chat_agent

Compact reference for 01_minimal_chat_agent.

Source: `docs/library/snippets/01_minimal_chat_agent.mdx`

````python 01_minimal_chat_agent.py
"""
Example 01: Minimal chat agent with one typed tool.

Run:
    python 01_minimal_chat_agent.py
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from afk.agents import Agent
from afk.tools import tool

class SumArgs(BaseModel):
    numbers: list[float] = Field(min_length=1, max_length=50)

@tool(
    args_model=SumArgs,
    name="sum_numbers",
    description="Add a list of numbers and return the numeric sum.",
)
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
````

> Code block truncated to 40 lines. Source: `docs/library/snippets/01_minimal_chat_agent.mdx`
