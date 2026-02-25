# Tool Hooks Middleware Agent

Progressive AFK example **23** focused on **tool prehooks/posthooks/middleware + registry middleware**.

## Complexity Profile
- Tier level: 2
- Stage-chain depth: 24
- Architecture: multi-module with scenario generation, quality gates, staged analytics, metrics, and report composition.

## What This Example Includes
- Feature workflow for: tool prehooks/posthooks/middleware + registry middleware
- Dynamic scenario synthesis (app/scenario.py)
- Structured validation (app/quality.py)
- Progressive stage transforms (app/stages/stage_*.py)
- Derived analytics (app/metrics.py)
- Consolidated report output (app/report_builder.py)

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/23_Tool_Hooks_Middleware_Agent
- Then execute:
  cd examples/projects/23_Tool_Hooks_Middleware_Agent && python3 main.py
