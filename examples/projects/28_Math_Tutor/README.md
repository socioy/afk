
# Math Tutor

A math tutoring agent that uses reasoning_enabled and reasoning_effort for extended thinking (chain of thought) when solving problems. The agent generates practice problems, provides progressive hints, and validates answers while showing step-by-step reasoning.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/28_Math_Tutor

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/28_Math_Tutor

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/28_Math_Tutor

Expected interaction
User: Give me an algebra problem
Agent: Here's your algebra problem: Solve for x: 3x + 7 = 22
User: I think x is 6
Agent: Not quite. The correct answer is 5. Here's how to solve it: Step 1: Subtract 7... Step 2: Divide by 3...
User: Give me a hint
Agent: Hint 1/3: Try isolating the variable by moving constants to the other side.

The agent uses reasoning_enabled=True with reasoning_effort="high" and reasoning_max_tokens=2000 for deep step-by-step thinking on math problems.
