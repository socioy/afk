"""
Example 03: Subagents with a deterministic router.

Run:
    uv run python docs/library/examples/03_subagents_with_router.py
"""

from __future__ import annotations

import asyncio
import os

from afk.agents import Agent
from afk.agents.types import RouterDecision, RouterInput


def route_subagents(data: RouterInput) -> RouterDecision:
    # Router receives normalized transcript/context state and picks subagent names.
    transcript = "\n".join(str(message.get("text", "")) for message in data.messages)
    if "draft" in transcript.lower():
        # Sequential two-stage workflow: draft first, then review.
        return RouterDecision(targets=["Writer", "Reviewer"], parallel=False)
    # Fallback flow: review-only path.
    return RouterDecision(targets=["Reviewer"], parallel=False)


async def main() -> None:
    model_name = os.getenv("AFK_LLM_MODEL", "gpt-4.1-mini")

    writer = Agent(
        name="Writer",
        model=model_name,
        instructions="Write a concise first draft from the user request.",
        inherit_context_keys=["topic"],
    )

    reviewer = Agent(
        name="Reviewer",
        model=model_name,
        instructions="Review the latest draft and produce actionable feedback.",
        inherit_context_keys=["topic"],
    )

    manager = Agent(
        name="Manager",
        model=model_name,
        instructions=(
            "Coordinate work between Writer and Reviewer. "
            "Use subagent outputs in your final response."
        ),
        # Runner uses this callback every step to select subagent execution targets.
        subagents=[writer, reviewer],
        subagent_router=route_subagents,
    )

    result = await manager.call(
        "Create a draft release note for our billing feature and then review it.",
        context={"topic": "billing"},
    )

    print("model:", model_name)
    print("state:", result.state)
    print("subagent_executions:", len(result.subagent_executions))
    print("final_text:\n", result.final_text)


if __name__ == "__main__":
    asyncio.run(main())
