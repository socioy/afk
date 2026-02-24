# Persistent CRM Memory Agent

Maintain customer context across turns with SQLite-backed memory and continuity metrics.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/07_Persistent_CRM_Memory_Agent

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/07_Persistent_CRM_Memory_Agent

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=/Users/lucifer/Programming/afk/examples/projects/07_Persistent_CRM_Memory_Agent

What this example focuses on
- Progressive AFK usage with real-world workflows.
- Correct use of runner results and runtime analytics fields.
- A reproducible script that can be adapted for production prototypes.
