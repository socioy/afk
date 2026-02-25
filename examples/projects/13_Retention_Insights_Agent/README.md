# Retention Insights Agent

Generate retention action plans with policy-aware analytics and controls.

Progressive AFK example **13**.

## Complexity Profile
- Tier: 3
- Progressive pass depth: 14
- Dynamic dataset rows scale with example number via app/dynamic_dataset.py.
- Multi-pass analytics scale with example number via app/progressive_analytics.py.

## AFK Focus
- policy engine + run_handle events + telemetry projection

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/13_Retention_Insights_Agent
- Then execute:
  cd examples/projects/13_Retention_Insights_Agent && python3 main.py
