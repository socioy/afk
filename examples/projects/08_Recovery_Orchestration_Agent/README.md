# Recovery Orchestration Agent

Drive recovery orchestration with streaming and thread continuity analytics.

Progressive AFK example **8**.

## Complexity Profile
- Tier: 2
- Progressive pass depth: 9
- Dynamic dataset rows scale with example number via app/dynamic_dataset.py.
- Multi-pass analytics scale with example number via app/progressive_analytics.py.

## AFK Focus
- run_stream + thread continuity + multi-turn orchestration

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/08_Recovery_Orchestration_Agent
- Then execute:
  cd examples/projects/08_Recovery_Orchestration_Agent && python3 main.py
