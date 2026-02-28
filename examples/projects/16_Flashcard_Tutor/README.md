
# Flashcard Tutor

An agent that quizzes you with flashcards and remembers your study progress using AFK's memory system (InMemoryMemoryStore). Demonstrates thread-scoped state, event recording, and persistent score tracking across a session.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/16_Flashcard_Tutor

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/16_Flashcard_Tutor

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/16_Flashcard_Tutor

Expected interaction
Agent: Welcome! Type 'quiz' to draw a flashcard, answer it, or type 'progress' to see your stats.
User: quiz
Agent: Question: What is the capital of France?
User: Paris
Agent: Correct! The answer is: Paris. Score: 1/1

The agent tracks your score and study history across the session using AFK's MemoryStore.

