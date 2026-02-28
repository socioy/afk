
# Bookmark Manager

A bookmark manager agent built on afk that demonstrates RunnerConfig and RunnerDebugConfig. The agent can add, list, search, remove, and export bookmarks using tools that maintain shared state across calls. Debug mode is enabled to show detailed runtime information during agent execution.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/13_Bookmark_Manager

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/13_Bookmark_Manager

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/13_Bookmark_Manager

Expected interaction
Agent: Hi! I'm your bookmark manager. I can save, search, and organize your web bookmarks.
User: Save https://docs.python.org as Python Docs with tags python, reference
Agent: Added bookmark #1: "Python Docs" (https://docs.python.org) [python, reference]
User: Save https://news.ycombinator.com as Hacker News with tags news, tech
Agent: Added bookmark #2: "Hacker News" (https://news.ycombinator.com) [news, tech]
User: Search for python
Agent: Found 1 bookmark(s) matching "python":
  1. Python Docs - https://docs.python.org [python, reference]
User: List my bookmarks
Agent: Here are your bookmarks:
  1. Python Docs - https://docs.python.org [python, reference]
  2. Hacker News - https://news.ycombinator.com [news, tech]
User: Remove 2
Agent: Removed bookmark #2: "Hacker News" (https://news.ycombinator.com)

The agent maintains state across the conversation, so bookmarks persist until you quit. Debug output will appear during execution because RunnerConfig has debug mode enabled.

