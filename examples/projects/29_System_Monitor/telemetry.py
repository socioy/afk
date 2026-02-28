"""
---
name: System Monitor — Telemetry
description: Custom TelemetrySink implementation and Debugger facade configuration.
tags: [telemetry, telemetry-sink, debugger, debugger-config]
---
---
This module contains two observability components:

1. **MonitorTelemetrySink** — A custom TelemetrySink protocol implementation that collects
   all telemetry data in memory: events (point-in-time occurrences), spans (timed durations),
   counters (incrementing metrics), and histograms (value distributions).

2. **Debugger + DebuggerConfig** — AFK's built-in debugger facade for formatted event output.
   The Debugger can attach to run handles or stream handles and output formatted debug lines
   for every agent event. DebuggerConfig controls verbosity, secret redaction, payload size
   limits, and timestamp display.
---
"""

import time  # <- For computing span durations.
from afk.core import TelemetrySink, TelemetryEvent, TelemetrySpan  # <- TelemetrySink is the protocol. TelemetryEvent and TelemetrySpan are the data types.
from afk.debugger import Debugger, DebuggerConfig  # <- Debugger facade for formatted event output. DebuggerConfig controls verbosity, redaction, and display.


# ===========================================================================
# Custom TelemetrySink implementation
# ===========================================================================
# The TelemetrySink protocol has five methods. Implementing all five gives
# full visibility into agent execution.

class MonitorTelemetrySink:
    """Custom telemetry sink that collects all telemetry data in memory."""

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []  # <- Point-in-time events.
        self.spans: list[dict] = []  # <- Completed spans with timing.
        self.counters: list[dict] = []  # <- Counter increments.
        self.histograms: list[dict] = []  # <- Histogram measurements.
        self._active_spans: dict[str, TelemetrySpan] = {}  # <- Track open spans by name.

    def record_event(self, event: TelemetryEvent) -> None:  # <- Called by the Runner for point-in-time telemetry events.
        """Record one point-in-time telemetry event."""
        self.events.append(event)

    def start_span(self, name: str, *, attributes: dict | None = None) -> TelemetrySpan | None:  # <- Called when a timed operation begins.
        """Start a timed span and return a span handle."""
        span = TelemetrySpan(
            name=name,
            started_at_ms=int(time.time() * 1000),
            attributes=dict(attributes or {}),
        )
        self._active_spans[name] = span
        return span

    def end_span(self, span: TelemetrySpan | None, *, status: str, error: str | None = None, attributes: dict | None = None) -> None:  # <- Called when a timed operation completes.
        """End a span and record its duration and status."""
        if span is None:
            return
        ended_at_ms = int(time.time() * 1000)
        self.spans.append({
            "name": span.name,
            "started_at_ms": span.started_at_ms,
            "ended_at_ms": ended_at_ms,
            "duration_ms": ended_at_ms - span.started_at_ms,
            "status": status,
            "error": error,
            "attributes": {**span.attributes, **dict(attributes or {})},
        })
        self._active_spans.pop(span.name, None)

    def increment_counter(self, name: str, value: int = 1, *, attributes: dict | None = None) -> None:  # <- Called for counting occurrences.
        """Record a counter increment."""
        self.counters.append({"name": name, "value": value, "attributes": dict(attributes or {})})

    def record_histogram(self, name: str, value: float, *, attributes: dict | None = None) -> None:  # <- Called for value distribution tracking.
        """Record a histogram measurement."""
        self.histograms.append({"name": name, "value": value, "attributes": dict(attributes or {})})

    def print_report(self) -> None:  # <- Custom reporting method (not part of the protocol).
        """Print a formatted telemetry report to the console."""
        print("\n" + "=" * 60)
        print("  TELEMETRY REPORT")
        print("=" * 60)

        print(f"\n  Events ({len(self.events)} recorded):")
        print("  " + "-" * 56)
        if self.events:
            for evt in self.events:
                attrs = ", ".join(f"{k}={v}" for k, v in evt.attributes.items()) if evt.attributes else "none"
                print(f"    [{evt.timestamp_ms}] {evt.name} | attrs: {attrs}")
        else:
            print("    (no events recorded)")

        print(f"\n  Spans ({len(self.spans)} completed):")
        print("  " + "-" * 56)
        if self.spans:
            for span in self.spans:
                error_info = f" | error: {span['error']}" if span["error"] else ""
                print(f"    {span['name']}: {span['duration_ms']}ms [{span['status']}]{error_info}")
        else:
            print("    (no spans recorded)")

        print(f"\n  Counters ({len(self.counters)} increments):")
        print("  " + "-" * 56)
        if self.counters:
            totals: dict[str, int] = {}
            for c in self.counters:
                totals[c["name"]] = totals.get(c["name"], 0) + c["value"]
            for name, total in totals.items():
                print(f"    {name}: {total}")
        else:
            print("    (no counters recorded)")

        print(f"\n  Histograms ({len(self.histograms)} measurements):")
        print("  " + "-" * 56)
        if self.histograms:
            for h in self.histograms:
                print(f"    {h['name']}: {h['value']}")
        else:
            print("    (no histograms recorded)")

        print("\n" + "=" * 60)


# ===========================================================================
# Debugger facade configuration
# ===========================================================================
# The Debugger is a convenience facade that creates debug-enabled Runners and
# can attach to run/stream handles to output formatted event lines.
#
# DebuggerConfig controls:
#   enabled: bool          — master toggle
#   verbosity: str         — "basic", "detailed", or "trace"
#   include_content: bool  — include message content in output
#   redact_secrets: bool   — redact sensitive values in payloads
#   max_payload_chars: int — truncate large payloads
#   emit_timestamps: bool  — show timestamps on each line
#   emit_step_snapshots: bool — show state snapshots per step

debugger_config = DebuggerConfig(  # <- Configure the debugger's output behavior.
    enabled=True,  # <- Master enable toggle.
    verbosity="detailed",  # <- "basic" shows minimal info, "detailed" shows attributes, "trace" shows full payloads.
    include_content=True,  # <- Include message/response content in debug output.
    redact_secrets=True,  # <- Automatically redact strings that look like API keys, tokens, etc.
    max_payload_chars=2000,  # <- Truncate payloads longer than this to keep output manageable.
    emit_timestamps=True,  # <- Prefix each debug line with a timestamp.
    emit_step_snapshots=True,  # <- Emit a state snapshot at the start of each step.
)

debugger = Debugger(config=debugger_config)  # <- Create the Debugger instance. This can create debug-enabled Runners via debugger.runner() and attach to handles via debugger.attach().
