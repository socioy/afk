
# Weather Agent

An example agent that checks the weather for any city using a custom tool. This example introduces **tools** - how to give an agent the ability to call functions.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/02_Weather_Agent

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/02_Weather_Agent

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/02_Weather_Agent

Expected interaction
User: What's the weather like in Tokyo?
Agent: The weather in Tokyo is 68F and clear - a beautiful day! Is there another city you'd like me to check?

The agent uses the get_weather tool to look up weather data and responds conversationally.

