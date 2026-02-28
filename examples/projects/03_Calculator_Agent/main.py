"""
---
name: Calculator Agent
description: A calculator agent with multiple math tools that can add, subtract, multiply, and divide.
tags: [agent, runner, tools]
---
---
This example demonstrates how to give an agent multiple tools so it can perform different operations. The agent is a calculator that can add, subtract, multiply, and divide. It shows how the LLM selects the right tool based on the user's intent, and how to use Pydantic models for structured tool arguments with multiple fields.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic is used to define structured argument models for tools. This lets you specify exactly what inputs each tool expects, with types, descriptions, and validation built in.
from afk.core import Runner  # <- Runner is responsible for executing agents and managing their state. Tl;dr: it's what you use to run your agents after you create them.
from afk.agents import Agent  # <- Agent is the main class for creating agents in AFK. It encapsulates the model, instructions, and other configurations that define the agent's behavior. Tl;dr: you create an Agent to define what your agent is and how it should behave, and then you use the Runner to execute it.
from afk.tools import tool  # <- The @tool decorator turns a plain Python function into a tool that an agent can call. You give it a name, description, and an args_model so the LLM knows when and how to use it.


# --- Tool argument schema ---

class TwoNumberArgs(BaseModel):  # <- A Pydantic model that defines the arguments for our math tools. Both tools share the same schema: two numbers (a and b). Using a shared model avoids duplication and keeps things consistent.
    a: float = Field(description="The first number")  # <- Field lets you attach metadata like descriptions so the LLM understands what each argument means.
    b: float = Field(description="The second number")


# --- Tool definitions ---

@tool(args_model=TwoNumberArgs, name="add", description="Add two numbers together")  # <- Each @tool call registers a tool the agent can use. The name and description help the LLM decide which tool to call based on the user's message.
def add(args: TwoNumberArgs) -> str:
    result = args.a + args.b
    return f"{args.a} + {args.b} = {result}"


@tool(args_model=TwoNumberArgs, name="subtract", description="Subtract second number from first")
def subtract(args: TwoNumberArgs) -> str:
    result = args.a - args.b
    return f"{args.a} - {args.b} = {result}"


@tool(args_model=TwoNumberArgs, name="multiply", description="Multiply two numbers")
def multiply(args: TwoNumberArgs) -> str:
    result = args.a * args.b
    return f"{args.a} * {args.b} = {result}"


@tool(args_model=TwoNumberArgs, name="divide", description="Divide first number by second")
def divide(args: TwoNumberArgs) -> str:
    if args.b == 0:
        return "Error: Cannot divide by zero!"  # <- Always handle edge cases in your tools. The LLM will relay this message back to the user.
    result = args.a / args.b
    return f"{args.a} / {args.b} = {result}"


# --- Agent setup ---

calculator = Agent(
    name="calculator",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use. See https://afk.arpan.sh/library/agents for more details.
    instructions="""
    You are a friendly math calculator assistant. When a user asks you to perform a calculation, you should use the appropriate tool (add, subtract, multiply, or divide) to compute the result. Always show your work by using the tool and then presenting the result clearly.

    If the user asks something that isn't a math question, politely let them know you're a calculator and can only help with math.

    Be friendly and encouraging!
    """,  # <- Instructions that guide the agent's behavior. The agent will choose the right tool based on these instructions and the user's message.
    tools=[add, subtract, multiply, divide],  # <- Pass all four tools to the agent. The LLM will automatically pick the right one based on what the user asks. This is the key concept: one agent, multiple tools.
)
runner = Runner()

if __name__ == "__main__":
    user_input = input(
        "[] > "
    )  # <- Take user input from the console to interact with the agent.

    response = runner.run_sync(
        calculator, user_message=user_input
    )  # <- Run the agent synchronously using the Runner. We pass the user's input as a message to the agent.

    print(
        f"[calculator] > {response.final_text}"
    )  # <- Print the agent's response to the console. Note: the response is an object that contains various information about the agent's execution, but we are only interested in the final text output for this example.



"""
---
Tl;dr: This example creates a calculator agent with four math tools (add, subtract, multiply, divide). It demonstrates how to define multiple tools using the @tool decorator with a shared Pydantic args model, and how to pass them all to a single agent. The LLM automatically selects the correct tool based on the user's intent. The user interacts through the console, and the agent's response is printed back.
---
---
What's next?
- Try adding more tools like power, square root, or modulo to see how the agent handles a larger toolset.
- Experiment with chaining operations: ask the agent to "add 5 and 3, then multiply the result by 2" and see how it handles multi-step calculations.
- Check out the other examples in the library to see how to create multi-agent systems where agents can delegate tasks to each other!
---
"""
