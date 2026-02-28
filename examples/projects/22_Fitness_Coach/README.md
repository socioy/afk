
# Fitness Coach

A fitness coaching agent that uses @prehook for input validation and @posthook for output formatting, demonstrating how to add cross-cutting concerns around tool execution.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/22_Fitness_Coach

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/22_Fitness_Coach

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/22_Fitness_Coach

Expected interaction
User: Calculate my calories. I'm 75kg, 175cm, age 30, moderately active, want to lose weight.
Agent: Your estimated daily calorie needs: BMR: 1728 cal, TDEE: 2678 cal, Target: 2178 cal/day...
       Disclaimer: These are estimates based on the Mifflin-St Jeor equation...

The agent uses prehooks to validate input ranges and posthooks to append disclaimers and safety tips.
