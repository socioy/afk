"""
---
name: Greeting Agent
description: A simple agent that greets a person by name.
tags: [agent, runner]
---
---
This example demonstrates how to create a simple agent that greets a person by name or as a friend if the name is not provided.
---
"""

from afk.core import Runner # <- Runner is reponnsible for executing agents and managing their state. Tl;dr: it's what you use to run your agents after you create them.
from afk.agents import Agent # <- Agent is the main class for creating agents in AFK. It encapsulates the model, instructions, and other configurations that define the agent's behavior. Tl;dr: you create an Agent to define what your agent is and how it should behave, and then you use the Runner to execute it.

greeter = Agent(
    name="greeter", # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b", # <- The llm model the agent will use. See https://afk.arpan.sh/library/agents for more details.
    instructions="""
    Greet the user by name. If you don't know their name, refer them as a friend.
    # example conversations:
    
    # example 1: user doesn't provide name
    user: "Hello!"
    agent: "Hello, friend! How is your day going?"

    # example 2: user provides name
    user: "Hi, I'm Arpan."
    agent: "Hi Arpan! Nice to meet you. How are you doing today?"

    **NOTE**: Be **creative** with your greetings! 

    """, # <- Instructions that guide the agent's behavior. This is where you can specify the agent's goals, constraints, and any other information that will help it perform its task.
)
runner = Runner()

if __name__ == "__main__":
    user_input = input(
        "[] > "
    )  # <-  Take user input from the console to interact with the agent.

    response = runner.run_sync(
        greeter, user_message=user_input
    )  # <- Run the agent synchronously using the Runner. We pass the user's input as a message to the agent.

    print(
        f"[greeter] > {response.final_text}"
    )  # <- Print the agent's response to the console. Note: the response is an object that contains various information about the agent's execution, but we are only interested in the final text output for this example.



"""
---
Tl;dr: This example creates a simple greeting agent that uses the "ollama_chat/gpt-oss:20b" model to greet users by name. The agent is guided by instructions that specify how it should greet users and handle cases where the user's name is not provided. The user can interact with the agent through the console, and the agent's response is printed back to the console.
---
---
What's next?
- Try changing the instructions to see how it affects the agent's behavior. For example, you could make the agent more formal, or have it ask follow-up questions after greeting the user.
- Explore the different fields in the response object to see what other information you can get about the agent's execution. For example, you could look at the reasoning trace to see how the agent arrived at its response.
- Check out the other examples in the library to see how to use tools, create more complex agents, and build multi-agent systems!
---
"""