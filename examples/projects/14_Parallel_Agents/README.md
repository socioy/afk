
# Parallel Agents

Run multiple agents concurrently using asyncio.gather. Each agent has a different analytical perspective (optimist, realist, critic) and they all process the same input in parallel, returning three viewpoints at once.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/14_Parallel_Agents

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/14_Parallel_Agents

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/14_Parallel_Agents

Expected interaction
User: I'm thinking about quitting my job to start a business.
Optimist: This is a fantastic opportunity to pursue your passion...
Realist: There are both risks and rewards to consider...
Critic: Before you leap, consider the financial runway you need...

All three agents run concurrently so you get all perspectives in roughly the time it takes to run one.

