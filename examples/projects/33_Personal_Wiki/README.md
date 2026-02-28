
# Personal Wiki

A personal wiki agent that uses long-term memory with text search for storing and retrieving knowledge articles by semantic relevance.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/33_Personal_Wiki

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/33_Personal_Wiki

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/33_Personal_Wiki

Expected interaction
User: Save an article about Python decorators
Agent: Article saved: 'Python Decorators' (ID: article-a1b2c3d4). Tags: python, decorators
User: Search for function wrappers
Agent: Search results for 'function wrappers': [article-a1b2c3d4] Python Decorators (score: 0.85)

The agent uses long-term memory with text search to find semantically relevant articles.
