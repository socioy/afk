
# Quiz Master

A trivia quiz game agent built on afk. The agent generates trivia questions, tracks your score, and gives feedback using tools, state, and instructions.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/07_Quiz_Master

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/07_Quiz_Master

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/07_Quiz_Master

Expected interaction
Agent: Welcome to Quiz Master! Ready for some trivia? Here's your first question...
Agent: What is the capital of France? A) London B) Paris C) Berlin D) Madrid
User: B
Agent: Correct! Paris is the capital of France. Score: 1/1

The agent will keep asking questions and tracking your score throughout the session.

