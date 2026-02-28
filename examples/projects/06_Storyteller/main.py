"""
---
name: Storyteller
description: An interactive storyteller agent that creates and continues stories collaboratively using the async Runner API.
tags: [agent, runner, async]
---
---
This example demonstrates how to use the async Runner API (`runner.run()`) instead of the synchronous `runner.run_sync()`.
The agent is an interactive storyteller that crafts vivid stories based on user prompts and continues them collaboratively.
---
"""

import asyncio  # <- asyncio is Python's built-in library for writing asynchronous code. We use it here to run the async Runner API.
from afk.core import Runner  # <- Runner is responsible for executing agents and managing their state. Tl;dr: it's what you use to run your agents after you create them.
from afk.agents import Agent  # <- Agent is the main class for creating agents in AFK. It encapsulates the model, instructions, and other configurations that define the agent's behavior. Tl;dr: you create an Agent to define what your agent is and how it should behave, and then you use the Runner to execute it.

storyteller = Agent(
    name="storyteller",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use. See https://afk.arpan.sh/library/agents for more details.
    instructions="""
    You are a creative storyteller. You craft vivid, engaging short stories based on user prompts.

    ## Rules:
    - Keep each story segment to 2-3 paragraphs
    - Use vivid imagery and sensory details
    - Leave room for the user to guide the story direction
    - When continuing a story, build naturally on what came before
    - Include a mix of dialogue and narration

    ## Story Start:
    When the user gives you a theme or prompt, begin the story immediately.
    End each segment with a moment of suspense or a choice point, then ask
    the user what should happen next.

    ## Story Continuation:
    When the user provides direction, weave it naturally into the narrative.
    """,  # <- Instructions that guide the agent's behavior. This is where you can specify the agent's goals, constraints, and any other information that will help it perform its task.
)
runner = Runner()


async def main():
    """Main async entry point for the storyteller."""
    print("[storyteller] > Welcome to the Interactive Storyteller!")
    print(
        "[storyteller] > Give me a theme, setting, or characters and I'll craft a story."
    )
    print(
        "[storyteller] > You can guide the story by telling me what happens next."
    )
    print(
        "[storyteller] > Type 'quit' to exit.\n"
    )

    while True:  # <- A simple loop to keep the conversation going. The user can type 'quit' to exit.
        user_input = input(
            "[] > "
        )  # <- Take user input from the console to interact with the agent.

        if user_input.lower() in ("quit", "exit", "q"):
            print("[storyteller] > And so our story comes to an end... Farewell!")
            break

        response = await runner.run(
            storyteller, user_message=user_input
        )  # <- Run the agent asynchronously using the Runner. This is the async version of run_sync() — same API, but uses `await`. Use this when you need to run agents inside an async context, or when you want to run multiple agents concurrently.

        print(
            f"\n[storyteller] > {response.final_text}\n"
        )  # <- Print the agent's response to the console. Note: the response is an object that contains various information about the agent's execution, but we are only interested in the final text output for this example.


if __name__ == "__main__":
    asyncio.run(
        main()
    )  # <- asyncio.run() is the standard way to run async code from a synchronous entry point. It creates an event loop, runs the coroutine, and cleans up when done. This is the recommended pattern for scripts that need to use async/await.



"""
---
Tl;dr: This example creates an interactive storyteller agent that uses the "ollama_chat/gpt-oss:20b" model to craft vivid stories based on user prompts. Instead of using `runner.run_sync()`, it demonstrates the async API with `await runner.run()`. The user can guide the story by providing directions, and the agent continues the narrative collaboratively. The script uses `asyncio.run()` as the entry point — the standard pattern for running async code from a synchronous context.
---
---
What's next?
- Try running multiple agents concurrently with `asyncio.gather()` — for example, one agent generates the story and another critiques it. This is where async really shines over sync.
- Explore passing conversation history to the runner so the storyteller remembers previous story segments across calls.
- Check out the other examples in the library to see how to use tools, create more complex agents, and build multi-agent systems!
---
"""
