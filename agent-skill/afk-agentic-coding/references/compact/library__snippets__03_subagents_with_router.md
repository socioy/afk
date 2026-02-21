# 03_subagents_with_router

Compact reference for 03_subagents_with_router.

Source: `docs/library/snippets/03_subagents_with_router.mdx`

````python 03_subagents_with_router.py
"""
Example 03: Subagents with a deterministic router.

Run:
    python 03_subagents_with_router.py
"""

from __future__ import annotations

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
````

> Code block truncated to 40 lines. Source: `docs/library/snippets/03_subagents_with_router.mdx`
