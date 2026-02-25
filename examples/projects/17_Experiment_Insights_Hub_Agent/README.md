# Experiment Insights Hub Agent

Synthesize experiment outcomes with scenario simulation and batch analytics.

Progressive AFK example **17**.

## Complexity Profile
- Tier: 4
- Progressive pass depth: 18
- Dynamic dataset rows scale with example number via app/dynamic_dataset.py.
- Multi-pass analytics scale with example number via app/progressive_analytics.py.

## AFK Focus
- sqlite memory + resume/compact + batch orchestration

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/17_Experiment_Insights_Hub_Agent
- Then execute:
  cd examples/projects/17_Experiment_Insights_Hub_Agent && python3 main.py
