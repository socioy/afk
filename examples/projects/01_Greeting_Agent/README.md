
# Greeting Agent

A minimal example of a agent built on afk. The agent greets a person by name or as a friend if the name is not provided. 

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/01_Greeting_Agent

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/01_Greeting_Agent

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/01_Greeting_Agent

Expected interaction
Agent: Hello! What's your name?
User: Arpan
Agent: Nice to meet you, Arpan!

The agent will remember the name for subsequent greetings.

