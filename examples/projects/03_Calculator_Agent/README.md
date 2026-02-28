
# Calculator Agent

An example agent built on afk that demonstrates multiple tools. The agent is a math calculator that can add, subtract, multiply, and divide using the appropriate tool based on the user's request.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/03_Calculator_Agent

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/03_Calculator_Agent

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/03_Calculator_Agent

Expected interaction
User: What is 12 times 5?
Agent: 12 * 5 = 60

User: Divide 100 by 4
Agent: 100 / 4 = 25.0

The agent will select the correct tool (add, subtract, multiply, or divide) based on your request.

