# Sales Forecast Planner Agent

Run batch forecast workflows with sqlite memory and compaction analytics.

Progressive AFK example **14**.

## Complexity Profile
- Tier: 4
- Progressive pass depth: 15
- Dynamic dataset rows scale with example number via app/dynamic_dataset.py.
- Multi-pass analytics scale with example number via app/progressive_analytics.py.

## AFK Focus
- sqlite memory + resume/compact + batch orchestration

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/14_Sales_Forecast_Planner_Agent
- Then execute:
  cd examples/projects/14_Sales_Forecast_Planner_Agent && python3 main.py
