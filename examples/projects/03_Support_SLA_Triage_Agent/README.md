# Support SLA Triage Agent

Prioritize support workload using SLA-aware triage analytics.

Progressive AFK example **3**.

## Complexity Profile
- Tier: 1
- Progressive pass depth: 4
- Dynamic dataset rows scale with example number via app/dynamic_dataset.py.
- Multi-pass analytics scale with example number via app/progressive_analytics.py.

## AFK Focus
- Runner sync execution + typed tools + result analytics

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/03_Support_SLA_Triage_Agent
- Then execute:
  cd examples/projects/03_Support_SLA_Triage_Agent && python3 main.py
