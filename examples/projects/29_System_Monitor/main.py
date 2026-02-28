"""
---
name: System Monitor
description: A system monitoring agent that uses a custom TelemetrySink to track tool calls, spans, and metrics during agent execution.
tags: [agent, runner, tools, telemetry, observability, telemetry-sink]
---
---
This example demonstrates AFK's telemetry and observability system. A TelemetrySink is a protocol
that receives telemetry data emitted by the Runner during agent execution -- events (point-in-time
occurrences), spans (timed durations), counters (incrementing metrics), and histograms (value
distributions). By implementing the TelemetrySink protocol, you can collect, display, export, or
forward this data to any observability backend. Here we build a custom sink that stores everything
in memory, wire it into the Runner, run a system monitoring agent with four tools, and then print
a full telemetry report showing exactly what happened inside the agent run.
---
"""

import asyncio  # <- Async required for runner.run().
from pydantic import BaseModel  # <- Pydantic for typed tool argument schemas.
from afk.core import Runner, TelemetrySink, TelemetryEvent, TelemetrySpan  # <- Runner executes agents. TelemetrySink is the protocol for telemetry backends. TelemetryEvent and TelemetrySpan are the data types emitted during execution.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.
from afk.tools import tool  # <- The @tool decorator turns a plain Python function into a tool the agent can call.


# ===========================================================================
# Step 1: Implement a custom TelemetrySink
# ===========================================================================
# The TelemetrySink protocol has five methods. Implementing all five gives you
# full visibility into agent execution: what happened (events), how long things
# took (spans), how often things occurred (counters), and value distributions
# (histograms). You can store, print, export, or forward this data anywhere.

class MonitorTelemetrySink:
    """Custom telemetry sink that collects all telemetry data in memory for inspection.

    This class implements the TelemetrySink protocol. The Runner calls these methods
    automatically during agent execution -- you never call them yourself.
    """

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []  # <- Point-in-time events (e.g., "tool_call_started", "agent_run_started").
        self.spans: list[dict] = []  # <- Completed spans with timing information (name, duration, status).
        self.counters: list[dict] = []  # <- Counter increments (e.g., "tool_calls: +1").
        self.histograms: list[dict] = []  # <- Histogram measurements (e.g., "response_time_ms: 142.5").
        self._active_spans: dict[str, TelemetrySpan] = {}  # <- Track open spans by name so we can compute durations on end_span.

    def record_event(self, event: TelemetryEvent) -> None:  # <- Called by the Runner for point-in-time telemetry events. Events capture "something happened" -- like a tool call starting, an agent run beginning, or an error occurring. Each event has a name, timestamp, and optional attributes dict.
        """Record one point-in-time telemetry event."""
        self.events.append(event)

    def start_span(  # <- Called when a timed operation begins (e.g., an LLM call, a tool execution). Returns a TelemetrySpan that the Runner passes back to end_span when the operation completes.
        self,
        name: str,
        *,
        attributes: dict | None = None,
    ) -> TelemetrySpan | None:
        """Start a timed span and return a span handle."""
        import time
        span = TelemetrySpan(
            name=name,
            started_at_ms=int(time.time() * 1000),  # <- Record the start time in milliseconds.
            attributes=dict(attributes or {}),
        )
        self._active_spans[name] = span  # <- Track the span so we can reference it in end_span.
        return span

    def end_span(  # <- Called when a timed operation completes. The Runner passes back the span from start_span, plus the terminal status ("ok" or "error") and optional error details.
        self,
        span: TelemetrySpan | None,
        *,
        status: str,
        error: str | None = None,
        attributes: dict | None = None,
    ) -> None:
        """End a span and record its duration and status."""
        if span is None:
            return
        import time
        ended_at_ms = int(time.time() * 1000)
        self.spans.append({
            "name": span.name,
            "started_at_ms": span.started_at_ms,
            "ended_at_ms": ended_at_ms,
            "duration_ms": ended_at_ms - span.started_at_ms,  # <- Duration = end - start. This tells you exactly how long each operation took.
            "status": status,
            "error": error,
            "attributes": {**span.attributes, **dict(attributes or {})},
        })
        self._active_spans.pop(span.name, None)  # <- Clean up the active span tracker.

    def increment_counter(  # <- Called to record a count of something. The Runner emits counter increments for things like "total tool calls", "total LLM requests", etc. You can aggregate these in your observability backend.
        self,
        name: str,
        value: int = 1,
        *,
        attributes: dict | None = None,
    ) -> None:
        """Record a counter increment."""
        self.counters.append({
            "name": name,
            "value": value,
            "attributes": dict(attributes or {}),
        })

    def record_histogram(  # <- Called to record a numeric measurement for distribution tracking. Examples: response latency in ms, token count per request, tool execution time. Useful for percentile analysis.
        self,
        name: str,
        value: float,
        *,
        attributes: dict | None = None,
    ) -> None:
        """Record a histogram measurement."""
        self.histograms.append({
            "name": name,
            "value": value,
            "attributes": dict(attributes or {}),
        })

    def print_report(self) -> None:  # <- Convenience method to dump all collected telemetry to the console. This is not part of the TelemetrySink protocol -- it's our custom reporting logic.
        """Print a formatted telemetry report to the console."""
        print("\n" + "=" * 60)
        print("  TELEMETRY REPORT")
        print("=" * 60)

        # --- Events section ---
        print(f"\n  Events ({len(self.events)} recorded):")
        print("  " + "-" * 56)
        if self.events:
            for evt in self.events:
                attrs = ", ".join(f"{k}={v}" for k, v in evt.attributes.items()) if evt.attributes else "none"
                print(f"    [{evt.timestamp_ms}] {evt.name} | attrs: {attrs}")
        else:
            print("    (no events recorded)")

        # --- Spans section ---
        print(f"\n  Spans ({len(self.spans)} completed):")
        print("  " + "-" * 56)
        if self.spans:
            for span in self.spans:
                error_info = f" | error: {span['error']}" if span["error"] else ""
                print(f"    {span['name']}: {span['duration_ms']}ms [{span['status']}]{error_info}")
        else:
            print("    (no spans recorded)")

        # --- Counters section ---
        print(f"\n  Counters ({len(self.counters)} increments):")
        print("  " + "-" * 56)
        if self.counters:
            # Aggregate counters by name for a summary view
            totals: dict[str, int] = {}
            for c in self.counters:
                totals[c["name"]] = totals.get(c["name"], 0) + c["value"]
            for name, total in totals.items():
                print(f"    {name}: {total}")
        else:
            print("    (no counters recorded)")

        # --- Histograms section ---
        print(f"\n  Histograms ({len(self.histograms)} measurements):")
        print("  " + "-" * 56)
        if self.histograms:
            for h in self.histograms:
                print(f"    {h['name']}: {h['value']}")
        else:
            print("    (no histograms recorded)")

        print("\n" + "=" * 60)


# ===========================================================================
# Step 2: Define system monitoring tools
# ===========================================================================
# Each tool returns simulated system metrics. In a real application, these
# would call psutil, /proc, or platform APIs. The telemetry sink automatically
# records when the Runner invokes each tool.

class EmptyArgs(BaseModel):  # <- Tools that take no user-facing arguments still need an args_model. This empty model tells the LLM "this tool has no parameters."
    pass


@tool(  # <- CPU monitoring tool. Returns simulated CPU usage data.
    args_model=EmptyArgs,
    name="check_cpu",
    description="Check current CPU usage and load averages. Returns CPU utilization percentage and per-core load.",
)
def check_cpu(args: EmptyArgs) -> str:
    return (
        "CPU Status:\n"
        "  Overall utilization: 34.2%\n"
        "  Core 0: 42.1% | Core 1: 28.7% | Core 2: 31.5% | Core 3: 34.6%\n"
        "  Load average (1m/5m/15m): 2.14 / 1.87 / 1.53\n"
        "  Running processes: 247\n"
        "  Status: HEALTHY"
    )  # <- Simulated data. In production, use psutil.cpu_percent(), os.getloadavg(), etc.


@tool(  # <- Memory monitoring tool. Returns simulated RAM usage data.
    args_model=EmptyArgs,
    name="check_memory",
    description="Check current memory (RAM) usage. Returns total, used, available, and swap information.",
)
def check_memory(args: EmptyArgs) -> str:
    return (
        "Memory Status:\n"
        "  Total: 16.0 GB\n"
        "  Used: 10.4 GB (65.0%)\n"
        "  Available: 5.6 GB\n"
        "  Cached: 3.2 GB\n"
        "  Swap total: 4.0 GB | Swap used: 0.8 GB (20.0%)\n"
        "  Status: HEALTHY"
    )


@tool(  # <- Disk monitoring tool. Returns simulated disk usage data.
    args_model=EmptyArgs,
    name="check_disk",
    description="Check disk usage across mounted volumes. Returns capacity, used space, and I/O stats.",
)
def check_disk(args: EmptyArgs) -> str:
    return (
        "Disk Status:\n"
        "  /dev/sda1 (/):\n"
        "    Total: 512 GB | Used: 287 GB (56.1%) | Free: 225 GB\n"
        "    Read IOPS: 1,240 | Write IOPS: 856\n"
        "  /dev/sdb1 (/data):\n"
        "    Total: 1 TB | Used: 412 GB (40.2%) | Free: 612 GB\n"
        "    Read IOPS: 320 | Write IOPS: 180\n"
        "  Status: HEALTHY"
    )


@tool(  # <- Network monitoring tool. Returns simulated network statistics.
    args_model=EmptyArgs,
    name="check_network",
    description="Check network interface status and traffic. Returns bandwidth, packet stats, and connection counts.",
)
def check_network(args: EmptyArgs) -> str:
    return (
        "Network Status:\n"
        "  Interface: eth0 (UP)\n"
        "    RX: 142.5 MB/s | TX: 38.7 MB/s\n"
        "    Packets RX: 98,432 | Packets TX: 45,210\n"
        "    Errors: 0 | Dropped: 2\n"
        "  Active connections: 1,847\n"
        "    TCP ESTABLISHED: 1,623 | TCP TIME_WAIT: 224\n"
        "  DNS resolution: 2.3ms avg\n"
        "  Status: HEALTHY"
    )


# ===========================================================================
# Step 3: Create the monitoring agent and wire up telemetry
# ===========================================================================

monitor_agent = Agent(
    name="system-monitor",  # <- The agent's display name.
    model="ollama_chat/gpt-oss:20b",  # <- The LLM model the agent will use.
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
    """,  # <- Instructions tell the agent to use all four tools, which generates rich telemetry data.
    tools=[check_cpu, check_memory, check_disk, check_network],  # <- Four monitoring tools. The Runner records telemetry for each tool call.
)

# --- Create the telemetry sink and wire it into the Runner ---
sink = MonitorTelemetrySink()  # <- Instantiate our custom sink. This object will collect all telemetry during the run.

runner = Runner(
    telemetry=sink,  # <- Pass the sink to the Runner. The Runner calls sink.record_event(), sink.start_span(), etc. automatically during execution. You can also pass telemetry="inmemory" to use the built-in InMemoryTelemetrySink from afk.observability.
)


# ===========================================================================
# Step 4: Run the agent and print the telemetry report
# ===========================================================================

async def main():
    print("System Monitor Agent")
    print("=" * 50)
    print("This agent checks system health and generates a telemetry report.")
    print("After the agent responds, you'll see a full telemetry breakdown.\n")

    user_input = input("[] > Describe what you'd like to monitor (or press Enter for default): ").strip()

    if not user_input:
        user_input = "Run a full system health check. Check CPU, memory, disk, and network."  # <- Default prompt that exercises all four tools.

    print(f"\nRunning system health check...\n")

    response = await runner.run(  # <- Run the agent. The Runner emits telemetry events, starts/ends spans, and increments counters for every internal operation (LLM calls, tool executions, etc.). All of this is captured by our sink.
        monitor_agent,
        user_message=user_input,
    )

    # --- Print the agent's response ---
    print(f"[system-monitor] > {response.final_text}")

    # --- Print the telemetry report ---
    sink.print_report()  # <- Our custom method. Dumps all collected events, spans, counters, and histograms to the console. This is the payoff: you can see exactly what the Runner did internally.

    print("\nTelemetry gives you full visibility into agent execution:")
    print("  - Events show WHAT happened (tool calls, LLM requests, state changes)")
    print("  - Spans show HOW LONG each operation took (with start/end times)")
    print("  - Counters show HOW MANY times something occurred (total tool calls, retries)")
    print("  - Histograms show VALUE DISTRIBUTIONS (response latencies, token counts)")


if __name__ == "__main__":
    asyncio.run(main())  # <- asyncio.run() starts the event loop for our async main function.



"""
---
Tl;dr: This example creates a system monitoring agent with four tools (check_cpu, check_memory,
check_disk, check_network) and a custom TelemetrySink that collects all telemetry data emitted
by the Runner during execution. The TelemetrySink protocol has five methods: record_event (for
point-in-time events), start_span/end_span (for timed operations with duration tracking),
increment_counter (for counting occurrences), and record_histogram (for value distribution
measurements). The sink is passed to the Runner via Runner(telemetry=sink), and the Runner
automatically calls the sink's methods during agent execution. After the run completes, the
script prints a full telemetry report showing all recorded events, spans, counters, and
histograms -- giving complete visibility into what the agent did and how long each step took.
---
---
What's next?
- Try passing telemetry="inmemory" to the Runner instead of a custom sink to use the built-in InMemoryTelemetrySink from afk.observability.
- Add real system metrics using the psutil library instead of simulated data.
- Forward telemetry to an external observability backend (Jaeger, Datadog, Prometheus) by implementing the sink methods as API calls.
- Use histogram data to set up alerting thresholds -- e.g., warn if average tool execution time exceeds 500ms.
- Combine telemetry with the eval system (see the Agent Test Harness example) to measure agent performance across test cases.
- Check out the other examples in the library to learn about memory, delegation, and eval systems!
---
"""
