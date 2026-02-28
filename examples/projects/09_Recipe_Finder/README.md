
# Recipe Finder

An agent that searches recipes by ingredient, retrieves recipe details, and suggests ingredient substitutions, demonstrating how to use ToolRegistry for centralized tool management.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/09_Recipe_Finder

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/09_Recipe_Finder

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/09_Recipe_Finder

Expected interaction
User: What can I make with chicken?
Agent: I found 2 recipes with chicken: Chicken Stir Fry and Chicken Curry.
User: Tell me more about Chicken Curry
Agent: Chicken Curry - Ingredients: chicken, curry powder, coconut milk, onion, garlic. Steps: ...
User: I don't have coconut milk, what can I use instead?
Agent: You can substitute coconut milk with heavy cream, yogurt, or cashew cream.

The agent uses a ToolRegistry to organize and manage its tools, and can list all registered tools on request.
