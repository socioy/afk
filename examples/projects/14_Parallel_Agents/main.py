"""
---
name: Parallel Agents
description: Run multiple agents concurrently using asyncio.gather to get different perspectives on the same input.
tags: [agent, runner, async, parallel]
---
---
This example demonstrates how to run multiple independent agents at the same time using asyncio.gather.
Three agents -- an optimist, a realist, and a critic -- each analyze the same user input from their own perspective.
Because they are independent, we can run them in parallel rather than one after another, which is significantly faster.
---
"""

import asyncio  # <- Python's built-in library for concurrent async execution. We use asyncio.gather to run multiple agents at the same time.
from afk.core import Runner  # <- Runner is responsible for executing agents and managing their state. Tl;dr: it's what you use to run your agents after you create them.
from afk.agents import Agent  # <- Agent is the main class for creating agents in AFK. It encapsulates the model, instructions, and other configurations that define the agent's behavior.

# --- Define three agents, each with a different analytical perspective ---

optimist = Agent(
    name="optimist",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use. See https://afk.arpan.sh/library/agents for more details.
    instructions="You analyze situations from an optimistic perspective. Find the silver lining, opportunities, and positive aspects in whatever the user describes. Be genuine, not dismissive.",  # <- Instructions that guide the agent's behavior. This optimist always looks for the upside.
)

realist = Agent(
    name="realist",  # <- A second agent with its own name and personality.
    model="ollama_chat/gpt-oss:20b",  # <- Same model, but different instructions produce very different outputs.
    instructions="You analyze situations pragmatically. State facts, identify risks and opportunities equally, and provide balanced, practical advice.",  # <- The realist gives a balanced, grounded perspective.
)

critic = Agent(
    name="critic",  # <- A third agent focused on constructive criticism.
    model="ollama_chat/gpt-oss:20b",  # <- All three agents can use the same model -- the instructions are what differentiate them.
    instructions="You are a constructive critic. Identify potential problems, risks, and weaknesses. Your goal is to help by pointing out what could go wrong so it can be addressed proactively.",  # <- The critic spots risks and weaknesses so they can be addressed early.
)

runner = Runner()  # <- A single Runner instance can execute multiple agents. You don't need a separate runner for each agent.


async def analyze_parallel(user_input: str):
    """Run all three agents concurrently and collect their perspectives."""
    results = await asyncio.gather(
        runner.run(optimist, user_message=user_input),
        runner.run(realist, user_message=user_input),
        runner.run(critic, user_message=user_input),
    )  # <- asyncio.gather runs all three agents at the same time. Much faster than running them sequentially.
    return results  # <- results is a list of three response objects, one per agent, in the same order they were passed to gather.


async def main():
    """Interactive loop: take user input, run three agents in parallel, print each perspective."""
    print("[parallel] > Three Perspectives Analyzer")
    print(
        "[parallel] > Describe a situation and get optimist, realist, and critic viewpoints."
    )
    print(
        "[parallel] > Type 'quit' to exit.\n"
    )  # <- Simple REPL-style loop so the user can try multiple inputs without restarting.

    while True:
        user_input = input(
            "[] > "
        )  # <- Take user input from the console to interact with the agents.

        if user_input.lower() in ("quit", "exit", "q"):
            break  # <- Let the user exit the loop gracefully.

        results = await analyze_parallel(
            user_input
        )  # <- Fan out to all three agents in parallel and wait for all of them to finish.

        for agent_name, result in zip(
            ["Optimist", "Realist", "Critic"], results
        ):  # <- Pair each label with its corresponding result for clean output.
            print(f"\n--- {agent_name} ---")
            print(
                f"{result.final_text}"
            )  # <- result.final_text contains the agent's text response. Each agent produces its own independent result.

        print()  # <- Blank line between rounds for readability.


if __name__ == "__main__":
    asyncio.run(
        main()
    )  # <- asyncio.run is the entry point for async programs. It starts the event loop and runs our main coroutine.


"""
---
Tl;dr: This example creates three agents (optimist, realist, critic) that each analyze the same user input from a different perspective. Using asyncio.gather, all three agents run concurrently -- so you get three viewpoints in roughly the time it takes to get one. This pattern is useful whenever you have independent tasks that don't depend on each other's output.
---
---
What's next?
- Try adding a fourth agent with a different perspective, like a "creative" or "devil's advocate" agent. Since they run in parallel, adding more agents has minimal impact on total response time.
- Experiment with different models for each agent. For example, you could give the critic a larger model for more thorough analysis while using a smaller, faster model for the optimist.
- Explore using the results from all three agents as input to a "summarizer" agent that synthesizes the perspectives into a single balanced recommendation -- this is a common multi-agent pattern.
- Check out the other examples in the library to see how to use tools, handoffs, and other agent capabilities!
---
"""
