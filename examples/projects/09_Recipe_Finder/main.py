"""
---
name: Recipe Finder
description: A recipe finder agent that uses ToolRegistry to manage tools for searching recipes, viewing details, and substituting ingredients.
tags: [agent, runner, tools, registry]
---
---
This example demonstrates how to use the ToolRegistry to centrally register, organize, and manage tools
instead of passing them directly to an agent. The agent can search recipes by ingredient, get detailed
recipe information, and suggest ingredient substitutions. It also shows how to list registered tools
with registry.list() and how to inspect tool call history via ToolCallRecord.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic is used to define structured argument models for tools. This lets you specify exactly what inputs each tool expects, with types, descriptions, and validation built in.
from afk.core import Runner  # <- Runner is responsible for executing agents and managing their state. Tl;dr: it's what you use to run your agents after you create them.
from afk.agents import Agent  # <- Agent is the main class for creating agents in AFK. It encapsulates the model, instructions, and other configurations that define the agent's behavior.
from afk.tools import tool, ToolRegistry, ToolCallRecord  # <- `tool` is the decorator for turning a function into an AFK tool. `ToolRegistry` is a centralized registry for managing, organizing, and executing tools. `ToolCallRecord` tracks metadata about each tool call (name, timing, success/failure).


# ===========================================================================
# Simulated recipe data
# ===========================================================================

RECIPES: dict[str, dict] = {  # <- A simple in-memory recipe database. In a real application, this would be a database or API call — but for this example, a dictionary keeps things focused on the ToolRegistry concepts.
    "Chicken Stir Fry": {
        "ingredients": ["chicken", "bell pepper", "soy sauce", "garlic", "ginger", "rice"],
        "steps": [
            "1. Slice the chicken into thin strips.",
            "2. Mince the garlic and ginger.",
            "3. Heat oil in a wok over high heat.",
            "4. Cook the chicken until golden, about 5 minutes.",
            "5. Add bell pepper, garlic, and ginger; stir fry for 3 minutes.",
            "6. Add soy sauce and toss to coat.",
            "7. Serve over steamed rice.",
        ],
        "cook_time": "20 minutes",
    },
    "Chicken Curry": {
        "ingredients": ["chicken", "curry powder", "coconut milk", "onion", "garlic", "rice"],
        "steps": [
            "1. Dice the chicken and onion.",
            "2. Saute the onion and garlic until softened.",
            "3. Add chicken and cook until browned.",
            "4. Stir in curry powder and cook for 1 minute.",
            "5. Pour in coconut milk and simmer for 15 minutes.",
            "6. Serve over steamed rice.",
        ],
        "cook_time": "30 minutes",
    },
    "Pasta Carbonara": {
        "ingredients": ["pasta", "eggs", "parmesan", "pancetta", "black pepper", "garlic"],
        "steps": [
            "1. Cook pasta in salted boiling water until al dente.",
            "2. Crisp the pancetta in a skillet.",
            "3. Whisk eggs and parmesan together in a bowl.",
            "4. Drain pasta, reserving some pasta water.",
            "5. Toss hot pasta with pancetta, then stir in the egg mixture.",
            "6. Add pasta water as needed for a creamy sauce.",
            "7. Season with black pepper and serve immediately.",
        ],
        "cook_time": "25 minutes",
    },
    "Vegetable Soup": {
        "ingredients": ["carrot", "celery", "onion", "potato", "tomato", "vegetable broth"],
        "steps": [
            "1. Dice all vegetables into small cubes.",
            "2. Saute onion and celery in a large pot until soft.",
            "3. Add carrot and potato; cook for 3 minutes.",
            "4. Pour in vegetable broth and diced tomato.",
            "5. Bring to a boil, then simmer for 20 minutes.",
            "6. Season with salt and pepper to taste.",
        ],
        "cook_time": "35 minutes",
    },
    "Banana Pancakes": {
        "ingredients": ["banana", "eggs", "flour", "milk", "butter", "maple syrup"],
        "steps": [
            "1. Mash the banana in a bowl.",
            "2. Whisk in eggs and milk until smooth.",
            "3. Stir in flour until just combined.",
            "4. Heat butter in a skillet over medium heat.",
            "5. Pour batter to form pancakes; cook until bubbles form, then flip.",
            "6. Serve with maple syrup.",
        ],
        "cook_time": "15 minutes",
    },
}

SUBSTITUTIONS: dict[str, list[str]] = {  # <- A lookup table of common ingredient substitutions. Each key is an ingredient, and the value is a list of possible alternatives.
    "coconut milk": ["heavy cream", "yogurt", "cashew cream", "almond milk"],
    "butter": ["olive oil", "coconut oil", "margarine", "applesauce"],
    "eggs": ["flax eggs (1 tbsp ground flax + 3 tbsp water per egg)", "chia eggs", "mashed banana", "silken tofu"],
    "milk": ["oat milk", "almond milk", "soy milk", "coconut milk"],
    "flour": ["almond flour", "oat flour", "coconut flour", "whole wheat flour"],
    "soy sauce": ["tamari", "coconut aminos", "liquid aminos", "fish sauce"],
    "parmesan": ["nutritional yeast", "pecorino romano", "aged asiago"],
    "pasta": ["zucchini noodles", "spaghetti squash", "rice noodles", "whole wheat pasta"],
    "rice": ["quinoa", "cauliflower rice", "couscous", "bulgur"],
    "chicken": ["tofu", "tempeh", "seitan", "chickpeas"],
    "pancetta": ["bacon", "prosciutto", "smoked tofu", "mushrooms"],
    "vegetable broth": ["chicken broth", "mushroom broth", "water with bouillon cube"],
    "maple syrup": ["honey", "agave nectar", "date syrup"],
    "curry powder": ["garam masala", "turmeric + cumin + coriander", "ras el hanout"],
}


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class SearchRecipesArgs(BaseModel):  # <- Defines the arguments for the search_recipes tool. The LLM will see this schema and know to pass an ingredient string.
    ingredient: str = Field(description="The ingredient to search for in recipes")


class RecipeDetailsArgs(BaseModel):  # <- Defines the arguments for the get_recipe_details tool. The LLM will see this schema and know to pass a recipe name.
    recipe_name: str = Field(description="The exact name of the recipe to get details for")


class SubstituteArgs(BaseModel):  # <- Defines the arguments for the substitute_ingredient tool. The LLM will see this schema and know to pass an ingredient to find alternatives for.
    ingredient: str = Field(description="The ingredient to find substitutions for")


# ===========================================================================
# Tool definitions
# ===========================================================================

@tool(args_model=SearchRecipesArgs, name="search_recipes", description="Search for recipes that contain a specific ingredient")  # <- Each @tool call creates a Tool object. The name and description help the LLM decide which tool to call based on the user's message.
def search_recipes(args: SearchRecipesArgs) -> str:
    query = args.ingredient.lower()
    matches = []
    for name, data in RECIPES.items():
        if any(query in ing.lower() for ing in data["ingredients"]):  # <- Search through each recipe's ingredient list for a match.
            matches.append(name)
    if not matches:
        return f"No recipes found containing '{args.ingredient}'. Try a different ingredient!"
    return f"Recipes with '{args.ingredient}': {', '.join(matches)}"


@tool(args_model=RecipeDetailsArgs, name="get_recipe_details", description="Get full details for a specific recipe including ingredients, steps, and cook time")
def get_recipe_details(args: RecipeDetailsArgs) -> str:
    recipe = RECIPES.get(args.recipe_name)
    if recipe is None:
        close = [name for name in RECIPES if args.recipe_name.lower() in name.lower()]  # <- Provide helpful suggestions if the exact name doesn't match.
        if close:
            return f"Recipe '{args.recipe_name}' not found. Did you mean: {', '.join(close)}?"
        return f"Recipe '{args.recipe_name}' not found. Available recipes: {', '.join(RECIPES.keys())}"
    ingredients_str = ", ".join(recipe["ingredients"])
    steps_str = "\n".join(recipe["steps"])
    return (
        f"--- {args.recipe_name} ---\n"
        f"Cook time: {recipe['cook_time']}\n"
        f"Ingredients: {ingredients_str}\n"
        f"Steps:\n{steps_str}"
    )


@tool(args_model=SubstituteArgs, name="substitute_ingredient", description="Find alternative ingredients that can replace a given ingredient")
def substitute_ingredient(args: SubstituteArgs) -> str:
    query = args.ingredient.lower()
    subs = SUBSTITUTIONS.get(query)
    if subs is None:
        available = [k for k in SUBSTITUTIONS if query in k]  # <- Partial match fallback so "milk" also finds "coconut milk".
        if available:
            return f"No exact match for '{args.ingredient}'. Related substitutions available for: {', '.join(available)}"
        return f"No substitutions found for '{args.ingredient}'. Try a common ingredient like butter, eggs, or milk."
    return f"Substitutes for '{args.ingredient}': {', '.join(subs)}"


# ===========================================================================
# ToolRegistry setup
# ===========================================================================

registry = ToolRegistry()  # <- Create a ToolRegistry instance. Instead of passing tools directly to the agent, you register them here. The registry provides centralized management: listing, organizing, tracking calls, and more.

registry.register(search_recipes)  # <- Register each tool with the registry. After registration, the registry knows about the tool and can list it, track calls to it, and provide it to agents.
registry.register(get_recipe_details)
registry.register(substitute_ingredient)

# --- Show what's registered ---
print("Registered tools:")  # <- Demonstrates registry.list() — returns all registered Tool objects. Useful for debugging, logging, or building dynamic UIs.
for t in registry.list():
    print(f"  - {t.spec.name}: {t.spec.description}")  # <- Each Tool has a .spec with its name and description, matching what the LLM sees.
print()


# ===========================================================================
# Agent and runner setup
# ===========================================================================

recipe_finder = Agent(
    name="recipe-finder",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use. See https://afk.arpan.sh/library/agents for more details.
    instructions="""
    You are a friendly recipe assistant. You help users find recipes, get cooking instructions, and suggest ingredient substitutions.

    When the user asks what they can cook with a certain ingredient:
    1. Use the search_recipes tool to find matching recipes.
    2. Present the results in a clear, friendly way.

    When the user wants details about a specific recipe:
    1. Use the get_recipe_details tool to fetch the full recipe.
    2. Present the ingredients, steps, and cook time clearly.

    When the user needs to substitute an ingredient:
    1. Use the substitute_ingredient tool to find alternatives.
    2. Suggest the best options and explain briefly when each works well.

    Be warm, encouraging, and helpful — cooking should be fun!

    **NOTE**: Always be specific about ingredient quantities and cooking steps!
    """,  # <- Instructions that guide the agent's behavior. The agent will choose the right tool based on these instructions and the user's message.
    tools=registry.list(),  # <- Instead of listing tools manually like tools=[search_recipes, get_recipe_details, ...], we pull them from the registry with registry.list(). This keeps the agent in sync with whatever tools the registry manages.
)
runner = Runner()

if __name__ == "__main__":
    print(
        "Recipe Finder Agent (type 'quit' to exit)"
    )  # <- Welcome banner so the user knows the agent is ready.
    print(
        "Ask me what you can cook, get recipe details, or find ingredient substitutes!\n"
    )

    while True:  # <- A conversation loop lets the user search for recipes, ask for details, and find substitutes across multiple turns. Each iteration collects input, runs the agent, and prints the response.
        user_input = input(
            "[] > "
        )  # <- Take user input from the console to interact with the agent.

        if user_input.strip().lower() in ("quit", "exit", "q"):
            # --- Show tool call history before exiting ---
            records: list[ToolCallRecord] = registry.recent_calls()  # <- ToolCallRecord tracks every tool call the registry executed: tool name, start/end time, success/failure, and optional error message. Great for debugging and observability.
            if records:
                print("\n--- Tool Call History ---")
                for rec in records:
                    status = "ok" if rec.ok else f"error: {rec.error}"  # <- Each ToolCallRecord has an .ok field (bool) and an .error field (str | None) so you can quickly see which calls succeeded or failed.
                    duration = rec.ended_at_s - rec.started_at_s  # <- Timing information is captured automatically. The difference gives you wall-clock duration for each call.
                    print(f"  {rec.tool_name}: {status} ({duration:.3f}s)")
            print("Goodbye! Happy cooking!")
            break  # <- Allow the user to exit the loop gracefully.

        response = runner.run_sync(
            recipe_finder, user_message=user_input
        )  # <- Run the agent synchronously using the Runner. We pass the user's input as a message to the agent. The runner will handle tool calls automatically.

        print(
            f"[recipe-finder] > {response.final_text}"
        )  # <- Print the agent's response to the console. Note: the response is an object that contains various information about the agent's execution, but we are only interested in the final text output for this example.



"""
---
Tl;dr: This example creates a recipe finder agent with three tools (search_recipes, get_recipe_details, substitute_ingredient) managed through a ToolRegistry. Instead of passing tools directly to the agent, you register them in a centralized registry and then pull them out with registry.list(). The registry tracks every tool invocation as a ToolCallRecord, giving you built-in observability (tool name, timing, success/failure). The agent runs in a conversation loop so the user can search recipes, view details, and find ingredient substitutions interactively.
---
---
What's next?
- Try adding more tools to the registry (e.g. a "rate_recipe" or "save_favorite" tool) and see how registry.list() automatically includes them for the agent.
- Experiment with registry.names() and registry.has() to check what tools are available programmatically before running the agent.
- Use registry.recent_calls() mid-conversation to inspect tool call history without exiting.
- Explore ToolRegistry's concurrency limiting (max_concurrency) and policy hooks for more advanced use cases.
- Check out the other examples in the library to see how to use middlewares, ToolContext, and build multi-agent systems!
---
"""
