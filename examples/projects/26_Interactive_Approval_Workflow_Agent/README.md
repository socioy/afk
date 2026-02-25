# Interactive Approval Workflow Agent

Progressive AFK example **26** focused on **interactive mode + InMemoryInteractiveProvider + policy user input**.

## Complexity Profile
- Tier level: 5
- Stage-chain depth: 27
- Architecture: multi-module with scenario generation, quality gates, staged analytics, metrics, and report composition.

## What This Example Includes
- Feature workflow for: interactive mode + InMemoryInteractiveProvider + policy user input
- Dynamic scenario synthesis (app/scenario.py)
- Structured validation (app/quality.py)
- Progressive stage transforms (app/stages/stage_*.py)
- Derived analytics (app/metrics.py)
- Consolidated report output (app/report_builder.py)

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/26_Interactive_Approval_Workflow_Agent
- Then execute:
  cd examples/projects/26_Interactive_Approval_Workflow_Agent && python3 main.py
