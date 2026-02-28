
# Habit Tracker

A habit tracking agent that uses SQLiteMemoryStore for persistent data across sessions, demonstrating how to build agents with long-term state that survives restarts.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/21_Habit_Tracker

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/21_Habit_Tracker

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/21_Habit_Tracker

Expected interaction
User: I want to start tracking meditation
Agent: Habit 'Meditation' added with goal: 10 minutes daily. Start tracking today!
User: I meditated today
Agent: 'Meditation' completed! Streak: 1 day, Total: 1 time. Great start!
(restart the program)
User: Show my habits
Agent: Your Habits: Meditation: streak=1 day, total=1 (data persisted from previous session!)

The agent uses SQLiteMemoryStore so habit data persists in a local database file across restarts.
