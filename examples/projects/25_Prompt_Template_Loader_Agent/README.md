# Prompt Template Loader Agent

Progressive AFK example **25** focused on **instruction_file + prompts_dir + prompt template rendering**.

## Complexity Profile
- Tier level: 4
- Stage-chain depth: 26
- Architecture: multi-module with scenario generation, quality gates, staged analytics, metrics, and report composition.

## What This Example Includes
- Feature workflow for: instruction_file + prompts_dir + prompt template rendering
- Dynamic scenario synthesis (app/scenario.py)
- Structured validation (app/quality.py)
- Progressive stage transforms (app/stages/stage_*.py)
- Derived analytics (app/metrics.py)
- Consolidated report output (app/report_builder.py)

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/25_Prompt_Template_Loader_Agent
- Then execute:
  cd examples/projects/25_Prompt_Template_Loader_Agent && python3 main.py
