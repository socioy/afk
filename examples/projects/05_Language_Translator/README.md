
# Language Translator

An example agent that translates text between multiple languages using rich instructions and runtime context. This demonstrates that powerful agent behavior can be achieved through instructions alone, without any tools.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/05_Language_Translator

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/05_Language_Translator

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/05_Language_Translator

Expected interaction
User: Translate 'good morning' to French
Agent: **Source** (English): good morning
       **Translation** (French): bonjour

User: How do you say 'thank you' in Japanese?
Agent: **Source** (English): thank you
       **Translation** (Japanese): arigatou gozaimasu

The agent runs in a conversation loop so you can translate multiple phrases in one session. Type 'quit' or 'exit' to stop.

