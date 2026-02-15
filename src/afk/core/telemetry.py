"""
Telemetry sinks for runner observability.

The default sinks are no-op/in-memory. `OpenTelemetrySink` can be used when
`opentelemetry-api` and `opentelemetry-sdk` are installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..llms.types import JSONValue


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """
    Point-in-time telemetry event.

    Attributes:
        name: Event name.
        timestamp_ms: Epoch milliseconds at emission time.
        attributes: JSON-safe event attributes.
    """

    name: str
    timestamp_ms: int
    attributes: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TelemetrySpan:
    """
    Started telemetry span.

    Attributes:
        name: Span name.
        started_at_ms: Span start timestamp.
        attributes: JSON-safe span attributes.
        native_span: Optional provider-native span object.
    """

    name: str
    started_at_ms: int
    attributes: dict[str, JSONValue] = field(default_factory=dict)
    native_span: Any = None


class TelemetrySink(Protocol):
    """Protocol implemented by telemetry backends."""

    def record_event(self, event: TelemetryEvent) -> None:
        """
        Record a single event.

        Args:
            event: Event payload to emit.
        """
        ...

    def start_span(self, name: str, *, attributes: dict[str, JSONValue] | None = None) -> TelemetrySpan | None:
        """
        Start a span when backend supports spans.

        Args:
            name: Span name.
            attributes: Optional initial span attributes.

        Returns:
            Span wrapper or `None` when unsupported.
        """
        ...

    def end_span(
        self,
        span: TelemetrySpan | None,
        *,
        status: str,
        error: str | None = None,
        attributes: dict[str, JSONValue] | None = None,
    ) -> None:
        """
        End a span with status and optional metadata.

        Args:
            span: Span returned from `start_span`.
            status: Terminal status string (`ok`/`error`/...).
            error: Optional error detail string.
            attributes: Optional final span attributes.
        """
        ...

    def increment_counter(
        self,
        name: str,
        value: int = 1,
        *,
        attributes: dict[str, JSONValue] | None = None,
    ) -> None:
        """
        Increment a named counter.

        Args:
            name: Counter name.
            value: Increment value.
            attributes: Optional counter attributes.
        """
        ...

    def record_histogram(
        self,
        name: str,
        value: float,
        *,
        attributes: dict[str, JSONValue] | None = None,
    ) -> None:
        """
        Record a histogram measurement.

        Args:
            name: Histogram name.
            value: Numeric measurement value.
            attributes: Optional histogram attributes.
        """
        ...


@dataclass(slots=True)
class NullTelemetrySink:
    """No-op telemetry sink used as safe default."""

    def record_event(self, event: TelemetryEvent) -> None:
        """
        Ignore event payload.

        Args:
            event: Event payload.
        """
        _ = event
        return None

    def start_span(self, name: str, *, attributes: dict[str, JSONValue] | None = None) -> TelemetrySpan | None:
        """
        Return `None` because spans are disabled.

        Args:
            name: Span name.
            attributes: Optional span attributes.
        """
        _ = name
        _ = attributes
        return None

    def end_span(
        self,
        span: TelemetrySpan | None,
        *,
        status: str,
        error: str | None = None,
        attributes: dict[str, JSONValue] | None = None,
    ) -> None:
        """
        Ignore span completion call.

        Args:
            span: Span object.
            status: Terminal status.
            error: Optional error string.
            attributes: Optional final attributes.
        """
        _ = span
        _ = status
        _ = error
        _ = attributes
        return None

    def increment_counter(
        self,
        name: str,
        value: int = 1,
        *,
        attributes: dict[str, JSONValue] | None = None,
    ) -> None:
        """
        Ignore counter call.

        Args:
            name: Counter name.
            value: Counter increment value.
            attributes: Optional attributes.
        """
        _ = name
        _ = value
        _ = attributes
        return None

    def record_histogram(
        self,
        name: str,
        value: float,
        *,
        attributes: dict[str, JSONValue] | None = None,
    ) -> None:
        """
        Ignore histogram call.

        Args:
            name: Histogram name.
            value: Measurement value.
            attributes: Optional attributes.
        """
        _ = name
        _ = value
        _ = attributes
        return None


@dataclass(slots=True)
class InMemoryTelemetrySink:
    """Test/debug telemetry sink that stores emitted measurements."""

    _events: list[TelemetryEvent] = field(default_factory=list)
    _spans_open: list[TelemetrySpan] = field(default_factory=list)
    _spans_closed: list[dict[str, Any]] = field(default_factory=list)
    _counters: list[dict[str, Any]] = field(default_factory=list)
    _histograms: list[dict[str, Any]] = field(default_factory=list)

    def record_event(self, event: TelemetryEvent) -> None:
        """
        Store emitted event.

        Args:
            event: Event payload.
        """
        self._events.append(event)

    def start_span(self, name: str, *, attributes: dict[str, JSONValue] | None = None) -> TelemetrySpan:
        """
        Start and store in-memory span.

        Args:
            name: Span name.
            attributes: Optional initial attributes.

        Returns:
            Span wrapper tracked by this sink.
        """
        span = TelemetrySpan(name=name, started_at_ms=now_ms(), attributes=dict(attributes or {}))
        self._spans_open.append(span)
        return span

    def end_span(
        self,
        span: TelemetrySpan | None,
        *,
        status: str,
        error: str | None = None,
        attributes: dict[str, JSONValue] | None = None,
    ) -> None:
        """
        Close span and persist completed record.

        Args:
            span: Span wrapper returned by `start_span`.
            status: Terminal status string.
            error: Optional error detail.
            attributes: Optional final attributes merged into span attributes.
        """
        if span is None:
            return None
        self._spans_closed.append(
            {
                "name": span.name,
                "started_at_ms": span.started_at_ms,
                "ended_at_ms": now_ms(),
                "status": status,
                "error": error,
                "attributes": {
                    **span.attributes,
                    **dict(attributes or {}),
                },
            }
        )
        return None

    def increment_counter(
        self,
        name: str,
        value: int = 1,
        *,
        attributes: dict[str, JSONValue] | None = None,
    ) -> None:
        """
        Record counter data point.

        Args:
            name: Counter name.
            value: Counter increment value.
            attributes: Optional attributes.
        """
        self._counters.append(
            {
                "name": name,
                "value": int(value),
                "attributes": dict(attributes or {}),
                "timestamp_ms": now_ms(),
            }
        )

    def record_histogram(
        self,
        name: str,
        value: float,
        *,
        attributes: dict[str, JSONValue] | None = None,
    ) -> None:
        """
        Record histogram data point.

        Args:
            name: Histogram name.
            value: Numeric measurement.
            attributes: Optional attributes.
        """
        self._histograms.append(
            {
                "name": name,
                "value": float(value),
                "attributes": dict(attributes or {}),
                "timestamp_ms": now_ms(),
            }
        )

    def events(self) -> list[TelemetryEvent]:
        """
        Return captured events.

        Returns:
            Snapshot of captured event list.
        """
        return list(self._events)

    def spans(self) -> list[dict[str, Any]]:
        """
        Return closed span records.

        Returns:
            Snapshot of closed span entries.
        """
        return list(self._spans_closed)

    def counters(self) -> list[dict[str, Any]]:
        """
        Return captured counter records.

        Returns:
            Snapshot of recorded counter data points.
        """
        return list(self._counters)

    def histograms(self) -> list[dict[str, Any]]:
        """
        Return captured histogram records.

        Returns:
            Snapshot of recorded histogram data points.
        """
        return list(self._histograms)


@dataclass(slots=True)
class OpenTelemetrySink:
    """
    OpenTelemetry sink using the global tracer/meter providers.

    This class performs lazy imports so AFK can run without OTel installed.
    """

    service_name: str = "afk-agent-runtime"
    tracer_name: str = "afk.core.runner"
    meter_name: str = "afk.core.runner"

    _tracer: Any = field(default=None, init=False, repr=False)
    _meter: Any = field(default=None, init=False, repr=False)
    _counters: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _histograms: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def _ensure_clients(self) -> None:
        """Lazily initialize OpenTelemetry tracer and meter clients."""
        if self._tracer is not None and self._meter is not None:
            return
        try:
            from opentelemetry import metrics, trace
        except Exception as e:
            raise RuntimeError(
                "OpenTelemetrySink requires 'opentelemetry-api'/'opentelemetry-sdk'"
            ) from e

        self._tracer = trace.get_tracer(self.tracer_name)
        self._meter = metrics.get_meter(self.meter_name)

    def _attr(self, value: dict[str, JSONValue] | None) -> dict[str, Any]:
        """Convert JSON attribute map into OpenTelemetry-compatible attributes."""
        attrs: dict[str, Any] = {}
        for key, item in (value or {}).items():
            attrs[str(key)] = _to_attr(item)
        return attrs

    def record_event(self, event: TelemetryEvent) -> None:
        """
        Record event by incrementing OpenTelemetry counter.

        Args:
            event: Event payload.
        """
        self.increment_counter(
            "agent.events",
            value=1,
            attributes={"event_name": event.name, **event.attributes},
        )

    def start_span(self, name: str, *, attributes: dict[str, JSONValue] | None = None) -> TelemetrySpan | None:
        """
        Start OpenTelemetry span.

        Args:
            name: Span name.
            attributes: Optional initial attributes.

        Returns:
            Span wrapper or `None` when OTel setup fails.
        """
        try:
            self._ensure_clients()
            span = self._tracer.start_span(name=name)
            attr = self._attr(attributes)
            if attr:
                span.set_attributes(attr)
            return TelemetrySpan(
                name=name,
                started_at_ms=now_ms(),
                attributes=dict(attributes or {}),
                native_span=span,
            )
        except Exception:
            return None

    def end_span(
        self,
        span: TelemetrySpan | None,
        *,
        status: str,
        error: str | None = None,
        attributes: dict[str, JSONValue] | None = None,
    ) -> None:
        """
        End OpenTelemetry span with status metadata.

        Args:
            span: Span wrapper from `start_span`.
            status: Terminal status string.
            error: Optional error detail string.
            attributes: Optional final attributes merged onto span.
        """
        if span is None or span.native_span is None:
            return None
        try:
            from opentelemetry.trace import Status, StatusCode

            native = span.native_span
            merged = {**span.attributes, **dict(attributes or {})}
            attr = self._attr(merged)
            if attr:
                native.set_attributes(attr)
            if error:
                native.record_exception(Exception(error))
            if status == "ok":
                native.set_status(Status(StatusCode.OK))
            else:
                native.set_status(Status(StatusCode.ERROR, error or status))
            native.end()
        except Exception:
            return None

    def increment_counter(
        self,
        name: str,
        value: int = 1,
        *,
        attributes: dict[str, JSONValue] | None = None,
    ) -> None:
        """
        Increment or create OpenTelemetry counter.

        Args:
            name: Counter name.
            value: Counter increment value.
            attributes: Optional counter attributes.
        """
        try:
            self._ensure_clients()
            counter = self._counters.get(name)
            if counter is None:
                counter = self._meter.create_counter(name)
                self._counters[name] = counter
            counter.add(int(value), attributes=self._attr(attributes))
        except Exception:
            return None

    def record_histogram(
        self,
        name: str,
        value: float,
        *,
        attributes: dict[str, JSONValue] | None = None,
    ) -> None:
        """
        Record OpenTelemetry histogram measurement.

        Args:
            name: Histogram name.
            value: Numeric value.
            attributes: Optional histogram attributes.
        """
        try:
            self._ensure_clients()
            histogram = self._histograms.get(name)
            if histogram is None:
                histogram = self._meter.create_histogram(name)
                self._histograms[name] = histogram
            histogram.record(float(value), attributes=self._attr(attributes))
        except Exception:
            return None


def now_ms() -> int:
    """
    Return current Unix epoch time in milliseconds.

    Returns:
        Integer epoch timestamp in milliseconds.
    """
    return int(time.time() * 1000)


def _to_attr(value: JSONValue) -> Any:
    """
    Convert JSON-safe values into OpenTelemetry attribute-compatible values.

    Args:
        value: JSON-safe value.

    Returns:
        Attribute-compatible primitive/tuple/string value.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return tuple(_to_attr(item) for item in value)
    if isinstance(value, dict):
        return str(value)
    return str(value)
