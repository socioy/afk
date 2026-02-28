"""
---
name: Weather Agent
description: An agent that checks weather for any city using a custom tool.
tags: [agent, runner, tools]
---
---
This example introduces **tools** - how to give an agent the ability to call functions. The agent can check the weather for any city using a custom tool defined with the `@tool` decorator and a Pydantic `args_model`.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic is used to define the schema (shape) of the arguments your tool accepts. The LLM uses this schema to know what parameters to pass when calling the tool.

from afk.core import Runner  # <- Runner is responsible for executing agents and managing their state. Tl;dr: it's what you use to run your agents after you create them.
from afk.agents import Agent  # <- Agent is the main class for creating agents in AFK. It encapsulates the model, instructions, and other configurations that define the agent's behavior. Tl;dr: you create an Agent to define what your agent is and how it should behave, and then you use the Runner to execute it.
from afk.tools import tool  # <- The `tool` decorator turns a plain function into an AFK Tool that agents can call. You define the args schema with a Pydantic model and the decorator handles the rest.


# --- Step 1: Define the tool's argument schema ---

class GetWeatherArgs(BaseModel):
    """Schema for the get_weather tool. The LLM reads this schema (including field descriptions) to know how to call the tool."""
    city: str = Field(description="The city to get weather for")  # <- Each field becomes a parameter the LLM can fill in. The `description` helps the LLM understand what value to provide.


# --- Step 2: Create the tool using the @tool decorator ---

@tool(args_model=GetWeatherArgs, name="get_weather", description="Get the current weather for a city")  # <- `args_model` tells AFK the shape of the arguments. `name` and `description` are what the LLM sees when deciding which tool to call.
def get_weather(args: GetWeatherArgs) -> str:
    """Simulate a weather lookup. In a real app, you'd call a weather API here."""
    weather_data = {
        "new york": "72\u00b0F, Partly Cloudy",
        "london": "58\u00b0F, Rainy",
        "tokyo": "68\u00b0F, Clear",
        "paris": "64\u00b0F, Overcast",
        "sydney": "80\u00b0F, Sunny",
    }  # <- Hardcoded weather data for demonstration. Replace this with a real API call in production.

    city_lower = args.city.lower()
    if city_lower in weather_data:
        return f"Weather in {args.city}: {weather_data[city_lower]}"
    return f"Weather in {args.city}: 70\u00b0F, Clear (simulated default)"  # <- Fallback for cities not in our mock data.


# --- Step 3: Create the agent and give it the tool ---

weather_agent = Agent(
    name="weather_agent",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use. See https://afk.arpan.sh/library/agents for more details.
    instructions="""
    You are a helpful weather assistant. When the user asks about the weather in a city,
    use the get_weather tool to look it up and share the result in a friendly, conversational way.
    If the user doesn't ask about weather, just chat normally and let them know you can check
    the weather for any city if they'd like.

    # example conversations:

    # example 1: user asks about weather
    user: "What's the weather like in Tokyo?"
    agent: *calls get_weather with city="Tokyo"*
    agent: "The weather in Tokyo is 68F and clear - a beautiful day! Is there another city you'd like me to check?"

    # example 2: user doesn't ask about weather
    user: "Hello!"
    agent: "Hey there! I'm your weather assistant. Want me to check the weather for a city? Just name a place!"

    **NOTE**: Always use the get_weather tool when the user asks about weather. Don't make up weather data yourself.

    """,  # <- Instructions that guide the agent's behavior. Notice we tell the agent to use the tool rather than guessing weather data.
    tools=[get_weather],  # <- This is the key part! Pass your tools as a list to the agent. The agent will see the tool's name, description, and parameter schema, and the LLM will decide when to call it based on the user's input.
)
runner = Runner()

if __name__ == "__main__":
    user_input = input(
        "[] > "
    )  # <- Take user input from the console to interact with the agent.

    response = runner.run_sync(
        weather_agent, user_message=user_input
    )  # <- Run the agent synchronously using the Runner. The Runner handles the tool-calling loop: if the LLM decides to call a tool, the Runner executes it and feeds the result back to the LLM automatically.

    print(
        f"[weather_agent] > {response.final_text}"
    )  # <- Print the agent's response to the console. By the time we get final_text, any tool calls have already been resolved and the LLM has composed its final answer.



"""
---
Tl;dr: This example creates a weather agent that uses a custom tool to check the weather for any city. The key concepts are: (1) defining a tool with the `@tool` decorator and a Pydantic `args_model` that describes the tool's parameters, (2) passing the tool to an Agent via `tools=[...]`, and (3) letting the LLM decide when to call the tool based on the user's input. The Runner handles the entire tool-calling loop automatically.
---
---
What's next?
- Try adding more tools to the agent! For example, you could add a "get_forecast" tool that returns a 5-day forecast, and the agent will learn to use both tools as needed.
- Experiment with the tool's description and parameter descriptions to see how they affect when and how the LLM calls the tool.
- Try removing the tool from the agent's tools list and see how the agent responds differently - it will no longer be able to look up weather data.
- Check out the next example to learn about multi-turn conversations and how agents can maintain context across multiple interactions!
---
"""
