# FinOps Budget Guard Agent

Track spend-risk tradeoffs with policy and telemetry-backed execution analytics.

Progressive AFK example **12**.

## Complexity Profile
- Tier: 3
- Progressive pass depth: 13
- Dynamic dataset rows scale with example number via app/dynamic_dataset.py.
- Multi-pass analytics scale with example number via app/progressive_analytics.py.

## AFK Focus
- policy engine + run_handle events + telemetry projection

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/12_FinOps_Budget_Guard_Agent
- Then execute:
  cd examples/projects/12_FinOps_Budget_Guard_Agent && python3 main.py
