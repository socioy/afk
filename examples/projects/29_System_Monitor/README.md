
# System Monitor

A system monitoring agent that uses a custom TelemetrySink to track tool calls, spans, counters, and histograms during agent execution. Demonstrates how telemetry gives full visibility into what the agent does and how long each step takes.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/29_System_Monitor

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/29_System_Monitor

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/29_System_Monitor

Expected interaction
User: Check the health of my system
Agent: (calls check_cpu, check_memory, check_disk, check_network tools)
Agent: Here is your system health report...

After the agent responds, the script prints a full telemetry report showing all recorded events, spans, counters, and histograms from the run.

