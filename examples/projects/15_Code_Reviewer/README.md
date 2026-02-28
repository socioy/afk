
# Code Reviewer

A code review agent that delegates to specialist subagents for style, bugs, and security analysis, then synthesizes a prioritized report.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/15_Code_Reviewer

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/15_Code_Reviewer

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/15_Code_Reviewer

Expected interaction
User: (pastes a code snippet for review)
Agent: Delegates to style-reviewer, bug-detector, and security-auditor subagents, then returns a comprehensive, prioritized code review report.

This example introduces subagents and demonstrates how a coordinator agent can delegate specialized tasks to child agents.

