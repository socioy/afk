
# Chat History Manager

A chat agent that manages multiple conversation threads using AFK's InMemoryMemoryStore. Demonstrates thread lifecycle, MemoryEvent logging, put_state/get_state for metadata, list_state for viewing keys, and thread isolation for separate conversations.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/38_Chat_History_Manager

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/38_Chat_History_Manager

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/38_Chat_History_Manager

Expected interaction
Agent: Welcome! You can manage multiple conversation threads.
User: Create a new thread called "work"
Agent: Created thread "work" and switched to it.
User: Hello, let's discuss the project deadline.
Agent: (responds and logs the message to the "work" thread)
User: Switch to thread "personal"
Agent: Switched to thread "personal". No history yet.
User: Show all threads
Agent: Active threads: work, personal

The agent maintains separate conversation histories per thread, all powered by AFK's memory system.
