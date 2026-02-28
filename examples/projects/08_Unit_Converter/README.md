
# Unit Converter

An agent that converts between units of length, weight, and temperature, demonstrating how tools can receive execution context via ToolContext.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/08_Unit_Converter

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/08_Unit_Converter

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/08_Unit_Converter

Expected interaction
User: Convert 5 miles to kilometers
Agent: 5.0 miles = 8.0467 km
User: Now convert 100 pounds to kilograms
Agent: 100.0 pounds = 45.3592 kg
User: What is 72 fahrenheit in celsius?
Agent: 72.0 fahrenheit = 22.2222 celsius

The agent tracks all conversions performed during the session and can show the history on request.

