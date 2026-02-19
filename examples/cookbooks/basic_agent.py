"""
basic_agent.py — Minimal AFK agent example.

Demonstrates the simplest possible agent: one model, no tools, one message.

Usage:
    export AFK_LLM_API_KEY=sk-...
    python examples/basic_agent.py
"""

from afk.quickstart import Agent, Runner


async def main() -> None:
    agent = Agent(
        name="greeter",
        model="gpt-4.1-mini",
        instructions="You are a helpful assistant. Be concise.",
    )

    runner = Runner()
    result = await runner.run(agent, user_message="Hello! What can you do?")
    print(result.final_text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
