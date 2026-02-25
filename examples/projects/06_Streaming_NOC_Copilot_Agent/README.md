# Streaming NOC Copilot Agent

Stream network operations guidance with real-time event analytics.

Progressive AFK example **6**.

## Complexity Profile
- Tier: 2
- Progressive pass depth: 7
- Dynamic dataset rows scale with example number via app/dynamic_dataset.py.
- Multi-pass analytics scale with example number via app/progressive_analytics.py.

## AFK Focus
- run_stream + thread continuity + multi-turn orchestration

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/06_Streaming_NOC_Copilot_Agent
- Then execute:
  cd examples/projects/06_Streaming_NOC_Copilot_Agent && python3 main.py
