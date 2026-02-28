
# Shopping Assistant

A shopping assistant with subagents that demonstrates context_defaults and inherit_context_keys for flowing configuration (currency, user preferences) from a parent coordinator to child agents. Tools read inherited context via ToolContext.metadata.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/26_Shopping_Assistant

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/26_Shopping_Assistant

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/26_Shopping_Assistant

Expected interaction
User: Find me a laptop
Agent: [Routes to product-finder] Found 1 product(s): ProBook Ultra 15 — $999.99 (USD)
User: Compare the laptop and keyboard prices
Agent: [Routes to price-checker] MechType 75 Wireless: $159.99 | ProBook Ultra 15: $999.99
User: Find electronics deals under $400
Agent: [Routes to price-checker] SoundWave Pro ANC — $349.99, MechType 75 — $159.99

The coordinator's context_defaults (currency, user_preference) flow to subagents via inherit_context_keys. Tools read them from ToolContext.metadata.
