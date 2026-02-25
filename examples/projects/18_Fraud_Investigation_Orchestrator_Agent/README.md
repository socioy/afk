# Fraud Investigation Orchestrator Agent

Run governance-grade fraud investigations with portfolio-level execution analytics.

Progressive AFK example **18**.

## Complexity Profile
- Tier: 5
- Progressive pass depth: 19
- Dynamic dataset rows scale with example number via app/dynamic_dataset.py.
- Multi-pass analytics scale with example number via app/progressive_analytics.py.

## AFK Focus
- governance-grade portfolio analytics + policy telemetry fusion

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/18_Fraud_Investigation_Orchestrator_Agent
- Then execute:
  cd examples/projects/18_Fraud_Investigation_Orchestrator_Agent && python3 main.py
