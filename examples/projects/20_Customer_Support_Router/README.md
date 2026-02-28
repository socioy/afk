
# Customer Support Router

A customer support system with specialist subagents (billing, technical, account, general) and a SubagentRouter callback that dynamically routes user queries to the right specialist based on keywords.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/20_Customer_Support_Router

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/20_Customer_Support_Router

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/20_Customer_Support_Router

Expected interaction
User: I was charged twice on my credit card
Agent: [Routes to billing-support] Let me check your account. What's your username?
User: The dashboard is really slow today
Agent: [Routes to technical-support] Let me check the service status...
User: I need to update my email address
Agent: [Routes to account-support] I can help with that. What's your username?

The coordinator uses a SubagentRouter callback for deterministic routing instead of relying on the LLM to choose.
