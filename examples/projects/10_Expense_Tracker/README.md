
# Expense Tracker

An expense tracking agent built on afk that demonstrates **FailSafeConfig** -- runtime safety limits that prevent agents from running too long, making too many LLM calls, or exceeding cost budgets. The agent manages expenses with full CRUD operations and category filtering, all protected by fail-safe limits.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/10_Expense_Tracker

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/10_Expense_Tracker

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/10_Expense_Tracker

Expected interaction
Agent: Hi! I'm your expense tracker. I can add, list, total, filter, and remove expenses.
User: Add lunch for $12.50 under food
Agent: Added expense #1: "lunch" -- $12.50 [food]
User: Add uber ride $8.75 transport
Agent: Added expense #2: "uber ride" -- $8.75 [transport]
User: Show my total
Agent: Your total expenses: $21.25

The agent will remember expenses for subsequent queries within a session. If the agent hits a safety limit (max steps, max LLM calls, or max tool calls), it catches AgentLoopLimitError and reports the limit gracefully.

