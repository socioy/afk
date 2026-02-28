
# Storyteller

An interactive storytelling agent built on afk that uses the async Runner API (`runner.run()`) to craft and continue stories collaboratively based on user prompts.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/06_Storyteller

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/06_Storyteller

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/06_Storyteller

Expected interaction
User: A lonely lighthouse keeper discovers a message in a bottle
Agent: The wind howled against the ancient stone walls of Marrow Point Lighthouse as Elara climbed the spiral staircase...
Agent: What should happen next?
User: She follows the map inside the bottle
Agent: Elara spread the weathered parchment across her desk, tracing the faded ink lines with trembling fingers...

The agent will craft story segments and ask the user to guide what happens next.

