"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Debugger facade for runner and stream event inspection.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Callable

from ..agents.types import AgentRunEvent, AgentRunHandle
from ..core import Runner, RunnerConfig, RunnerDebugConfig
from ..core.streaming import AgentStreamHandle, AgentStreamEvent
from .types import DebuggerConfig


_SECRET_KEY_MARKERS = (
    "api_key",
    "token",
    "secret",
    "authorization",
    "password",
)


class Debugger:
    """Convenience debugger facade that can build debug-enabled runners."""

    def __init__(self, config: DebuggerConfig | None = None) -> None:
        self.config = config or DebuggerConfig()

    def runner(self, **runner_kwargs: Any) -> Runner:
        """Create a Runner preconfigured for debug instrumentation."""
        existing_config = runner_kwargs.get("config")
        debug_cfg = RunnerDebugConfig(**asdict(self.config))
        if isinstance(existing_config, RunnerConfig):
            merged = RunnerConfig(
                **{
                    **asdict(existing_config),
                    "debug": True,
                    "debug_config": debug_cfg,
                }
            )
        else:
            merged = RunnerConfig(debug=True, debug_config=debug_cfg)
        runner_kwargs["config"] = merged
        return Runner(**runner_kwargs)

    async def attach(
        self,
        handle: AgentRunHandle,
        *,
        sink: Callable[[str], None] | None = None,
    ) -> None:
        """Attach to a run handle and stream formatted run events."""
        writer = sink or print
        async for event in handle.events:
            writer(self.format_run_event(event))

    async def attach_stream(
        self,
        stream: AgentStreamHandle,
        *,
        sink: Callable[[str], None] | None = None,
    ) -> None:
        """Attach to a stream handle and stream formatted stream events."""
        writer = sink or print
        async for event in stream:
            writer(self.format_stream_event(event))

    def format_run_event(self, event: AgentRunEvent) -> str:
        """Format one AgentRunEvent in a compact debugger-friendly line."""
        payload = self._normalize_payload(event.data)
        return (
            f"[{event.type}] run={event.run_id} thread={event.thread_id} "
            f"step={event.step} state={event.state} data={payload}"
        )

    def format_stream_event(self, event: AgentStreamEvent) -> str:
        """Format one AgentStreamEvent in a compact debugger-friendly line."""
        payload = self._normalize_payload(event.data)
        text = event.text_delta or ""
        if text and len(text) > 200:
            text = text[:200] + "..."
        return (
            f"[stream:{event.type}] step={event.step} state={event.state} "
            f"tool={event.tool_name} ok={event.tool_success} text={text!r} data={payload}"
        )

    def _normalize_payload(self, payload: dict[str, Any]) -> str:
        value: Any = payload
        if self.config.redact_secrets:
            value = self._redact_value(value)
        out = json.dumps(value, ensure_ascii=True)
        if len(out) > self.config.max_payload_chars:
            out = out[: self.config.max_payload_chars] + "..."
        return out

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                key_s = str(key)
                if any(marker in key_s.lower() for marker in _SECRET_KEY_MARKERS):
                    out[key_s] = "***REDACTED***"
                else:
                    out[key_s] = self._redact_value(item)
            return out
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        return value
