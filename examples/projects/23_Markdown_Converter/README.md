
# Markdown Converter

A markdown formatting agent that uses tool-level @middleware for timing and logging, plus @registry_middleware for global call counting across all tools.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/23_Markdown_Converter

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/23_Markdown_Converter

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/23_Markdown_Converter

Expected interaction
User: Convert this to a bullet list: apples, bananas, oranges
  [registry-mw] calling tool: text_to_bullet_list
  [middleware] tool executed in 0.1ms
Agent: Here's your markdown bullet list: - apples - bananas - oranges

The agent demonstrates middleware chaining with timing, logging, and global call counting.
