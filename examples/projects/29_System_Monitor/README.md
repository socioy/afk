
# System Monitor

A system monitoring agent demonstrating two observability layers: custom TelemetrySink for telemetry data collection and the Debugger facade for formatted event output.

## Project Structure

```
29_System_Monitor/
  main.py          # Entry point — streaming with debugger output + telemetry report
  tools.py         # Monitoring tools (CPU, memory, disk, network)
  telemetry.py     # Custom TelemetrySink + Debugger/DebuggerConfig setup
```

## Key Concepts

- **TelemetrySink protocol**: 5 methods (record_event, start_span, end_span, increment_counter, record_histogram) called automatically by the Runner
- **Debugger facade**: `debugger.format_stream_event(event)` for compact formatted output
- **DebuggerConfig**: Controls verbosity ("basic"/"detailed"/"trace"), secret redaction, payload size limits
- **debugger.runner()**: Creates a pre-configured debug Runner

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/29_System_Monitor

Expected output
After the agent responds, you'll see both formatted debugger output (per-event lines) and a full telemetry report (events, spans, counters, histograms).
