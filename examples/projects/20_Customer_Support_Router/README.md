
# Customer Support Router

A customer support system combining SubagentRouter for deterministic routing with InstructionRole callbacks for dynamic instruction augmentation based on customer tier, system health, and business hours.

## Project Structure

```
20_Customer_Support_Router/
  main.py       # Entry point — coordinator setup and conversation loop
  agents.py     # Specialist subagents, SubagentRouter, coordinator with InstructionRoles
  tools.py      # Tool definitions and simulated data (accounts, services, issues)
  roles.py      # Three InstructionRole callbacks (tier, health, hours)
```

## Key Concepts

- **SubagentRouter**: `(context: dict) -> list[str]` callback for deterministic routing
- **InstructionRole**: `(context: dict, state: str) -> str | list[str] | None` callbacks that APPEND dynamic instructions to the agent's base instructions at runtime
- **Stacked roles**: Multiple InstructionRoles run in order; each adds its own dynamic context

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/20_Customer_Support_Router

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/20_Customer_Support_Router

Expected interaction
Customer username: alice
User: I was charged twice on my credit card
Agent: [Routes to billing-support, VIP handling for premium customer]
User: The dashboard is really slow today
Agent: [Routes to technical-support, proactively mentions degraded database]
