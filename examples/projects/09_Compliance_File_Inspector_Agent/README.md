# Compliance File Inspector Agent

Inspect compliance workflows with richer multi-turn analytics patterns.

Progressive AFK example **9**.

## Complexity Profile
- Tier: 2
- Progressive pass depth: 10
- Dynamic dataset rows scale with example number via app/dynamic_dataset.py.
- Multi-pass analytics scale with example number via app/progressive_analytics.py.

## AFK Focus
- run_stream + thread continuity + multi-turn orchestration

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/09_Compliance_File_Inspector_Agent
- Then execute:
  cd examples/projects/09_Compliance_File_Inspector_Agent && python3 main.py
