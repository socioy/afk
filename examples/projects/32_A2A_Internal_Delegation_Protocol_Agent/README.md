# A2A Internal Delegation Protocol Agent

Progressive AFK example **32** focused on **InternalA2AProtocol invoke/invoke_stream/dead-letter**.

## Complexity Profile
- Tier level: 11
- Stage-chain depth: 33
- Architecture: multi-module with scenario generation, quality gates, staged analytics, metrics, and report composition.

## What This Example Includes
- Feature workflow for: InternalA2AProtocol invoke/invoke_stream/dead-letter
- Dynamic scenario synthesis (app/scenario.py)
- Structured validation (app/quality.py)
- Progressive stage transforms (app/stages/stage_*.py)
- Derived analytics (app/metrics.py)
- Consolidated report output (app/report_builder.py)

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/32_A2A_Internal_Delegation_Protocol_Agent
- Then execute:
  cd examples/projects/32_A2A_Internal_Delegation_Protocol_Agent && python3 main.py
