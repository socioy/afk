
# File Organizer

An async-tools example that organizes files by category in a simulated filesystem. Demonstrates how to define tools with `async def` for I/O-bound operations.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/12_File_Organizer

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/12_File_Organizer

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/12_File_Organizer

Expected interaction
User: Organize all the files in my downloads folder
Agent: (uses list_files, then auto_organize or move_file to sort files into documents, images, music, videos, and other)

The agent will categorize files by their extension and move them to the appropriate folders.

