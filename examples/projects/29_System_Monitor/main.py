"""
---
name: System Monitor
description: A system monitoring agent demonstrating custom TelemetrySink and the Debugger facade for comprehensive observability.
tags: [agent, runner, tools, telemetry, telemetry-sink, debugger, debugger-config, observability, streaming]
---
---
This example demonstrates AFK's observability stack with two complementary approaches:

1. **Custom TelemetrySink**: Implement the TelemetrySink protocol to collect telemetry data
   (events, spans, counters, histograms) in memory. The Runner calls your sink's methods
   automatically during execution. Great for building custom dashboards, metrics exporters,
   or alerting systems.

2. **Debugger + DebuggerConfig**: AFK's built-in debugger facade that attaches to stream
   handles and outputs formatted debug lines for every agent event. DebuggerConfig controls
   verbosity ("basic", "detailed", "trace"), secret redaction, payload size limits, and
   timestamp display. The debugger can also create pre-configured debug Runners via
   debugger.runner().

The agent uses four monitoring tools (CPU, memory, disk, network) to generate rich telemetry
data. After the run completes, both the custom telemetry report and the debugger output are shown.
---
"""

import asyncio  # <- Async required for streaming and debugger.attach_stream().
from afk.core import Runner  # <- Runner executes agents.
from afk.agents import Agent  # <- Agent defines the monitoring agent.

from tools import check_cpu, check_memory, check_disk, check_network  # <- Import monitoring tools.
from telemetry import MonitorTelemetrySink, debugger, debugger_config  # <- Import custom sink and debugger facade.


# ===========================================================================
# Agent definition
# ===========================================================================

monitor_agent = Agent(
    name="system-monitor",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a system monitoring agent. When the user asks about system health, use ALL
    four monitoring tools to gather a comprehensive picture:
    - check_cpu: CPU utilization and load
    - check_memory: RAM and swap usage
    - check_disk: Disk capacity and I/O
    - check_network: Network traffic and connections

    Always use all four tools to provide a complete system health report.
    Summarize findings with a clear status for each subsystem and highlight any
    areas of concern. End with an overall health assessment.
    """,
    tools=[check_cpu, check_memory, check_disk, check_network],
)


# ===========================================================================
# Runner with telemetry sink
# ===========================================================================

sink = MonitorTelemetrySink()  # <- Instantiate the custom sink.

runner = Runner(
    telemetry=sink,  # <- Attach the custom sink. The Runner calls sink.record_event(), start_span(), etc. automatically.
)


# ===========================================================================
# Main entry point — streaming with debugger attach
# ===========================================================================

async def main():
    print("System Monitor Agent")
    print("=" * 55)
    print()
    print("Observability active:")
    print("  - Custom TelemetrySink: collects events, spans, counters, histograms")
    print(f"  - Debugger: verbosity={debugger_config.verbosity}, redact_secrets={debugger_config.redact_secrets}")
    print()

    user_input = input("[] > Describe what to monitor (or Enter for full check): ").strip()
    if not user_input:
        user_input = "Run a full system health check. Check CPU, memory, disk, and network."

    print(f"\nRunning system health check...\n")

    # --- Use run_stream so the Debugger can attach ---
    handle = await runner.run_stream(  # <- run_stream returns a handle for real-time event streaming. The Debugger attaches to this handle.
        monitor_agent,
        user_message=user_input,
    )

    # --- Debugger attaches to the stream handle ---
    # The debugger.attach_stream() method iterates handle events and formats each one.
    # We supply a custom sink function (print) to output the debug lines.
    # In production, you'd replace print with a logger, file writer, or metrics collector.
    #
    # NOTE: attach_stream consumes the handle's event iterator, so we process the
    # agent's text output through the debugger's formatted output.

    print("[debug output]")
    print("-" * 55)

    final_text = ""
    async for event in handle:  # <- Iterate stream events manually for both debug output and text capture.
        # --- Format and print debug line ---
        debug_line = debugger.format_stream_event(event)  # <- debugger.format_stream_event(event) returns a compact formatted string. Format: "[stream:{type}] step={step} tool={tool_name} text={snippet}..."
        print(f"  {debug_line}")

        # --- Capture text output ---
        if event.type == "text_delta":
            final_text += event.text_delta
        elif event.type == "completed":
            if event.result:
                usage = event.result.usage
                print(f"\n  [tokens: {usage.input_tokens} in / {usage.output_tokens} out]")

    # --- Print agent's full response ---
    print("-" * 55)
    print(f"\n[system-monitor] > {final_text}")

    # --- Print the custom telemetry report ---
    sink.print_report()

    print("\nTwo observability layers demonstrated:")
    print("  1. Custom TelemetrySink: collects raw telemetry data (events, spans, counters, histograms)")
    print("  2. Debugger facade: formats stream events into compact debug lines with configurable verbosity")


if __name__ == "__main__":
    asyncio.run(main())



"""
---
Tl;dr: This example demonstrates AFK's observability with two layers: a custom TelemetrySink that
collects raw telemetry data (events, spans, counters, histograms) in memory, and the Debugger
facade with DebuggerConfig that formats stream events into compact debug lines. The TelemetrySink
is attached via Runner(telemetry=sink) and its methods are called automatically during execution.
The Debugger is configured with DebuggerConfig(verbosity="detailed", redact_secrets=True) and uses
format_stream_event() to produce formatted output for each event. After the run completes, both
the custom telemetry report and formatted debug output are shown.
---
---
What's next?
- Examine telemetry.py for the custom TelemetrySink implementation and DebuggerConfig settings.
- Try changing verbosity to "trace" for maximum detail or "basic" for minimal output.
- Use debugger.runner() to create a pre-configured debug Runner instead of configuring separately.
- Use debugger.attach(handle) with run_handle() instead of run_stream() for non-streaming debugging.
- Forward telemetry to Jaeger, Datadog, or Prometheus by implementing sink methods as API calls.
- Add histogram-based alerting (e.g., warn if tool execution time exceeds 500ms).
- Check out the Document Approval example for AgentRunHandle lifecycle controls!
---
"""
