
# Todo Manager

A stateful todo manager agent built on afk. The agent can add, list, complete, and remove tasks using tools that maintain shared state across calls.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/04_Todo_Manager

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/04_Todo_Manager

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/04_Todo_Manager

Expected interaction
Agent: Hi! I'm your todo manager. I can add, list, complete, and remove tasks for you.
User: Add buy groceries
Agent: Added todo #1: "buy groceries"
User: Add finish homework
Agent: Added todo #2: "finish homework"
User: Show my tasks
Agent: Here are your todos:
  1. [ ] buy groceries
  2. [ ] finish homework
User: Complete 1
Agent: Marked todo #1 "buy groceries" as done!
User: Show my tasks
Agent: Here are your todos:
  1. [x] buy groceries
  2. [ ] finish homework

The agent maintains state across the conversation, so todos persist until you quit.

