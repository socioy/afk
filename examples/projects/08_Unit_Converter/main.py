"""
---
name: Unit Converter
description: An agent that converts between units of length, weight, and temperature using ToolContext.
tags: [agent, runner, tools, tool-context]
---
---
This example demonstrates how to use ToolContext in tool functions. ToolContext provides execution context
(such as request_id, user_id, and metadata) that the framework injects automatically when a tool function
includes it in its signature. The agent converts between length, weight, and temperature units and uses
ToolContext to log metadata about each conversion. A session-level conversion history is also maintained.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic is used to define strongly-typed argument models for each tool. The agent framework validates incoming arguments against these models before calling the tool.

from afk.core import Runner  # <- Runner is responsible for executing agents and managing their state. Tl;dr: it's what you use to run your agents after you create them.
from afk.agents import Agent  # <- Agent is the main class for creating agents in AFK. It encapsulates the model, instructions, and other configurations that define the agent's behavior.
from afk.tools import tool, ToolContext  # <- `tool` is the decorator for turning a function into an AFK tool. `ToolContext` carries per-call execution context (request_id, user_id, metadata) and is injected automatically by the framework when included in a tool function's signature.

# ---------------------------------------------------------------------------
# Conversion history tracked across calls
# ---------------------------------------------------------------------------
conversion_history: list[str] = []  # <- A simple module-level list that accumulates every conversion result string. The show_history tool reads from this list so the user can review past conversions in the current session.

# ===========================================================================
# Length conversion
# ===========================================================================

class LengthConvertArgs(BaseModel):  # <- Every tool needs a Pydantic model that describes its arguments. The framework uses this model to generate the JSON Schema that the LLM sees when deciding how to call the tool, and to validate the arguments the LLM provides.
    value: float = Field(description="The numeric value to convert")
    from_unit: str = Field(description="Source unit (e.g. 'meters', 'feet', 'inches', 'km', 'miles')")
    to_unit: str = Field(description="Target unit (e.g. 'meters', 'feet', 'inches', 'km', 'miles')")

# Length conversion factors to meters (base unit)
LENGTH_TO_METERS = {
    "meters": 1.0, "m": 1.0,
    "kilometers": 1000.0, "km": 1000.0,
    "centimeters": 0.01, "cm": 0.01,
    "millimeters": 0.001, "mm": 0.001,
    "miles": 1609.344,
    "yards": 0.9144,
    "feet": 0.3048, "ft": 0.3048,
    "inches": 0.0254, "in": 0.0254,
}


@tool(args_model=LengthConvertArgs, name="convert_length", description="Convert between length/distance units")  # <- The @tool decorator turns this function into a full AFK Tool object. `args_model` tells the framework which Pydantic model to validate against. `name` and `description` are what the LLM sees when choosing which tool to call.
def convert_length(args: LengthConvertArgs, ctx: ToolContext) -> str:  # <- ToolContext is injected automatically by the framework when included in the function signature. The framework detects `ctx: ToolContext` and passes the current execution context (request_id, user_id, metadata) without the LLM needing to supply it.
    from_factor = LENGTH_TO_METERS.get(args.from_unit.lower())
    to_factor = LENGTH_TO_METERS.get(args.to_unit.lower())
    if from_factor is None:
        return f"Unknown source unit: {args.from_unit}. Supported: {', '.join(LENGTH_TO_METERS.keys())}"
    if to_factor is None:
        return f"Unknown target unit: {args.to_unit}. Supported: {', '.join(LENGTH_TO_METERS.keys())}"

    result = args.value * from_factor / to_factor  # <- Convert to base unit (meters) then to the target unit.
    record = f"{args.value} {args.from_unit} = {result:.4f} {args.to_unit}"
    conversion_history.append(record)  # <- Track every conversion so the user can ask for history later.
    return record


# ===========================================================================
# Weight / mass conversion
# ===========================================================================

class WeightConvertArgs(BaseModel):
    value: float = Field(description="The numeric value to convert")
    from_unit: str = Field(description="Source unit (e.g. 'kg', 'pounds', 'ounces', 'grams')")
    to_unit: str = Field(description="Target unit (e.g. 'kg', 'pounds', 'ounces', 'grams')")

# Weight conversion factors to kilograms (base unit)
WEIGHT_TO_KG = {
    "kilograms": 1.0, "kg": 1.0,
    "grams": 0.001, "g": 0.001,
    "milligrams": 0.000001, "mg": 0.000001,
    "pounds": 0.453592, "lbs": 0.453592, "lb": 0.453592,
    "ounces": 0.0283495, "oz": 0.0283495,
    "tons": 907.185, "tonnes": 1000.0,
}


@tool(args_model=WeightConvertArgs, name="convert_weight", description="Convert between weight/mass units")
def convert_weight(args: WeightConvertArgs, ctx: ToolContext) -> str:  # <- Again, ctx: ToolContext is injected by the framework. You can use it to inspect metadata, request_id, or user_id if needed.
    from_factor = WEIGHT_TO_KG.get(args.from_unit.lower())
    to_factor = WEIGHT_TO_KG.get(args.to_unit.lower())
    if from_factor is None:
        return f"Unknown source unit: {args.from_unit}. Supported: {', '.join(WEIGHT_TO_KG.keys())}"
    if to_factor is None:
        return f"Unknown target unit: {args.to_unit}. Supported: {', '.join(WEIGHT_TO_KG.keys())}"

    result = args.value * from_factor / to_factor  # <- Convert to base unit (kg) then to the target unit.
    record = f"{args.value} {args.from_unit} = {result:.4f} {args.to_unit}"
    conversion_history.append(record)
    return record


# ===========================================================================
# Temperature conversion
# ===========================================================================

class TempConvertArgs(BaseModel):
    value: float = Field(description="The temperature value to convert")
    from_unit: str = Field(description="Source unit: 'celsius', 'fahrenheit', or 'kelvin'")
    to_unit: str = Field(description="Target unit: 'celsius', 'fahrenheit', or 'kelvin'")

TEMP_ALIASES = {  # <- Normalise common aliases so users can type 'c', 'f', 'k' or the full name.
    "celsius": "celsius", "c": "celsius",
    "fahrenheit": "fahrenheit", "f": "fahrenheit",
    "kelvin": "kelvin", "k": "kelvin",
}


def _convert_temp(value: float, from_u: str, to_u: str) -> float:  # <- Helper that contains the actual temperature formulas. Temperature conversion is not a simple ratio like length/weight, so we handle each pair explicitly.
    if from_u == to_u:
        return value
    # Convert to Celsius first as the intermediate representation
    if from_u == "fahrenheit":
        celsius = (value - 32) * 5 / 9
    elif from_u == "kelvin":
        celsius = value - 273.15
    else:
        celsius = value
    # Convert from Celsius to the target unit
    if to_u == "fahrenheit":
        return celsius * 9 / 5 + 32
    elif to_u == "kelvin":
        return celsius + 273.15
    return celsius


@tool(args_model=TempConvertArgs, name="convert_temperature", description="Convert between temperature units (Celsius, Fahrenheit, Kelvin)")
def convert_temperature(args: TempConvertArgs, ctx: ToolContext) -> str:  # <- ToolContext is available here too. Every tool that needs context simply adds ctx: ToolContext to its signature.
    from_u = TEMP_ALIASES.get(args.from_unit.lower())
    to_u = TEMP_ALIASES.get(args.to_unit.lower())
    if from_u is None:
        return f"Unknown source unit: {args.from_unit}. Supported: {', '.join(TEMP_ALIASES.keys())}"
    if to_u is None:
        return f"Unknown target unit: {args.to_unit}. Supported: {', '.join(TEMP_ALIASES.keys())}"

    result = _convert_temp(args.value, from_u, to_u)
    record = f"{args.value} {args.from_unit} = {result:.4f} {args.to_unit}"
    conversion_history.append(record)
    return record


# ===========================================================================
# Conversion history tool
# ===========================================================================

class EmptyArgs(BaseModel):  # <- Some tools don't need any arguments. We still need a Pydantic model (even an empty one) because the framework always validates args through a model.
    pass


@tool(args_model=EmptyArgs, name="conversion_history", description="Show the history of all conversions performed in this session")
def show_history(args: EmptyArgs) -> str:  # <- This tool intentionally omits ToolContext to show that it's optional. If a tool doesn't need execution context, you can leave ctx out and the framework won't inject it.
    if not conversion_history:
        return "No conversions performed yet."
    return "Conversion History:\n" + "\n".join(
        f"  {i + 1}. {h}" for i, h in enumerate(conversion_history)
    )


# ===========================================================================
# Agent and runner setup
# ===========================================================================

converter = Agent(
    name="unit-converter",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use. See https://afk.arpan.sh/library/agents for more details.
    instructions="""
    You are a precise unit converter. You can convert between units of length, weight, and temperature.

    When the user asks for a conversion:
    1. Identify the type of conversion (length, weight, or temperature).
    2. Use the appropriate tool: convert_length, convert_weight, or convert_temperature.
    3. Present the result clearly, always showing both the original value with its unit and the converted value with its unit.

    When the user asks for conversion history, use the conversion_history tool.

    Be helpful and precise. If the user provides a unit you don't recognize, let them know the supported units.

    **NOTE**: Always show units clearly in your responses!
    """,  # <- Instructions that guide the agent's behavior. This is where you can specify the agent's goals, constraints, and any other information that will help it perform its task.
    tools=[convert_length, convert_weight, convert_temperature, show_history],  # <- The list of tools this agent can use. The framework registers these tools and exposes their JSON Schemas to the LLM so it can decide when and how to call them.
)
runner = Runner()

if __name__ == "__main__":
    print(
        "Unit Converter Agent (type 'quit' to exit)"
    )  # <- Welcome banner so the user knows the agent is ready.

    while True:  # <- A conversation loop lets the user perform multiple conversions in a single session. Each iteration collects input, runs the agent, and prints the response.
        user_input = input(
            "[] > "
        )  # <- Take user input from the console to interact with the agent.

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break  # <- Allow the user to exit the loop gracefully.

        response = runner.run_sync(
            converter, user_message=user_input
        )  # <- Run the agent synchronously using the Runner. We pass the user's input as a message to the agent. The runner will handle tool calls automatically.

        print(
            f"[unit-converter] > {response.final_text}"
        )  # <- Print the agent's response to the console. Note: the response is an object that contains various information about the agent's execution, but we are only interested in the final text output for this example.



"""
---
Tl;dr: This example creates a unit converter agent equipped with four tools (convert_length, convert_weight, convert_temperature, conversion_history). It demonstrates how ToolContext works in AFK: when a tool function includes `ctx: ToolContext` in its signature, the framework automatically injects the current execution context (request_id, user_id, metadata) without the LLM needing to supply it. Tools that don't need context can simply omit `ctx` from their signature. The agent runs in a conversation loop so the user can perform multiple conversions and review their history.
---
---
What's next?
- Try adding more unit types (e.g. volume, speed, data size) by following the same pattern: define an args model, a conversion map, and a @tool-decorated function.
- Experiment with ToolContext by reading ctx.request_id or ctx.metadata inside a tool to see what the framework provides.
- Add prehooks or posthooks to your tools (e.g. a posthook that logs every conversion to a file).
- Check out the other examples in the library to see how to use middlewares, create multi-agent systems, and build more complex workflows!
---
"""
