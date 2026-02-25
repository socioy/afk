# SRE Capacity Planner Agent

Plan SRE capacity under governance controls and aggregate portfolio metrics.

Progressive AFK example **19**.

## Complexity Profile
- Tier: 5
- Progressive pass depth: 20
- Dynamic dataset rows scale with example number via app/dynamic_dataset.py.
- Multi-pass analytics scale with example number via app/progressive_analytics.py.

## AFK Focus
- governance-grade portfolio analytics + policy telemetry fusion

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/19_SRE_Capacity_Planner_Agent
- Then execute:
  cd examples/projects/19_SRE_Capacity_Planner_Agent && python3 main.py
