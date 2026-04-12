"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Debugger facade for runner and stream event inspection.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from ..agents.types import AgentRunEvent, AgentRunHandle
from ..core import Runner, RunnerConfig, RunnerDebugConfig
from ..core.streaming import AgentStreamEvent, AgentStreamHandle
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


# === Interactive Step Debugger ===


@dataclass
class BreakpointConfig:
    """
    Configuration for interactive debugging breakpoints.

    Attributes:
        pause_on_tool: Glob patterns for tool names to pause on.
        pause_on_llm_error: Pause when LLM returns an error.
        pause_on_tool_error: Pause when tool execution fails.
        pause_on_state: Pause on specific agent states.
        max_steps_before_auto_resume: Auto-resume after N steps.
        auto_resume_timeout_s: Auto-resume after timeout.
        emit_step_snapshots: Emit state snapshots at each step.
    """

    pause_on_tool: list[str] = field(default_factory=list)
    pause_on_llm_error: bool = False
    pause_on_tool_error: bool = True
    pause_on_state: list[str] = field(default_factory=list)
    max_steps_before_auto_resume: int = 0
    auto_resume_timeout_s: float = 0.0
    emit_step_snapshots: bool = True


@dataclass
class StepSnapshot:
    """
    Snapshot of agent state at a specific step.

    Attributes:
        step: Step number.
        state: Agent state.
        tool_name: Tool being executed (if any).
        tool_args: Tool arguments (may be redacted).
        llm_response: LLM response text (truncated).
        message_count: Number of messages in context.
        timestamp_ms: Snapshot timestamp.
    """

    step: int
    state: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    llm_response: str | None = None
    message_count: int = 0
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class DebugSession:
    """
    Active debug session for an agent run.

    Tracks breakpoints, snapshots, and user interactions.
    """

    run_id: str
    thread_id: str
    config: BreakpointConfig
    breakpoints_enabled: bool = True
    paused: bool = False
    pause_reason: str | None = None
    step_snapshots: list[StepSnapshot] = field(default_factory=list)
    breakpoints_hit: list[str] = field(default_factory=list)
    resume_callback: Callable[[], None] | None = None

    async def pause(self, reason: str) -> None:
        """Pause the debug session."""
        self.paused = True
        self.pause_reason = reason

    async def resume(self) -> None:
        """Resume the debug session."""
        self.paused = False
        self.pause_reason = None

    def add_snapshot(self, snapshot: StepSnapshot) -> None:
        """Add a step snapshot to the session."""
        self.step_snapshots.append(snapshot)

    def record_breakpoint_hit(self, pattern: str) -> None:
        """Record that a breakpoint was hit."""
        self.breakpoints_hit.append(pattern)


class InteractiveDebugger:
    """
    Interactive debugger for agent execution.

    Provides breakpoint support, step-through capability,
    and state inspection during agent runs.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, DebugSession] = {}
        self._global_breakpoints: list[str] = []
        self._default_config = BreakpointConfig()

    def enable_debugger(
        self,
        run_id: str,
        thread_id: str,
        config: BreakpointConfig | None = None,
    ) -> DebugSession:
        """
        Enable debugging for a specific run.

        Args:
            run_id: Run identifier.
            thread_id: Thread identifier.
            config: Optional breakpoint configuration.

        Returns:
            DebugSession for this run.
        """
        session = DebugSession(
            run_id=run_id,
            thread_id=thread_id,
            config=config or self._default_config,
        )
        self._sessions[run_id] = session
        return session

    def disable_debugger(self, run_id: str) -> None:
        """Disable debugging for a run."""
        self._sessions.pop(run_id, None)

    def get_session(self, run_id: str) -> DebugSession | None:
        """Get active debug session for a run."""
        return self._sessions.get(run_id)

    def add_global_breakpoint(self, pattern: str) -> None:
        """Add a global breakpoint pattern."""
        self._global_breakpoints.append(pattern)

    def remove_global_breakpoint(self, pattern: str) -> None:
        """Remove a global breakpoint pattern."""
        self._global_breakpoints.remove(pattern)

    def should_break(
        self,
        run_id: str,
        tool_name: str | None = None,
        event_type: str | None = None,
        state: str | None = None,
        has_error: bool = False,
    ) -> tuple[bool, str | None]:
        """
        Check if execution should break for debugging.

        Args:
            run_id: Run identifier.
            tool_name: Current tool name (if any).
            event_type: Current event type.
            state: Current agent state.
            has_error: Whether an error occurred.

        Returns:
            Tuple of (should_break, reason).
        """
        session = self._sessions.get(run_id)
        if not session or not session.breakpoints_enabled:
            return False, None

        config = session.config

        # Check tool breakpoints
        if tool_name:
            for pattern in config.pause_on_tool + self._global_breakpoints:
                if fnmatch.fnmatch(tool_name, pattern):
                    session.record_breakpoint_hit(pattern)
                    return True, f"tool breakpoint: {pattern}"
                # Check negated patterns (starting with !)
                if pattern.startswith("!"):
                    if fnmatch.fnmatch(tool_name, pattern[1:]):
                        return False, None

        # Check error breakpoints
        if has_error:
            if config.pause_on_tool_error:
                return True, "tool error"
            if config.pause_on_llm_error:
                return True, "LLM error"

        # Check state breakpoints
        if state and state in config.pause_on_state:
            return True, f"state: {state}"

        return False, None

    async def on_step_start(
        self,
        run_id: str,
        step: int,
        state: str,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
        llm_response: str | None = None,
        message_count: int = 0,
    ) -> StepSnapshot | None:
        """
        Called at the start of each step to check breakpoints.

        Args:
            run_id: Run identifier.
            step: Step number.
            state: Current agent state.
            tool_name: Tool being executed.
            tool_args: Tool arguments.
            llm_response: LLM response text.
            message_count: Number of messages in context.

        Returns:
            StepSnapshot if breakpoint triggered, None otherwise.
        """
        session = self._sessions.get(run_id)
        if not session:
            return None

        should_break, reason = self.should_break(
            run_id=run_id,
            tool_name=tool_name,
            state=state,
        )

        snapshot = StepSnapshot(
            step=step,
            state=state,
            tool_name=tool_name,
            tool_args=tool_args,
            llm_response=llm_response[:200] if llm_response else None,
            message_count=message_count,
        )

        if session.config.emit_step_snapshots:
            session.add_snapshot(snapshot)

        if should_break:
            await session.pause(reason or "breakpoint")
            return snapshot

        return None

    async def wait_for_resume(self, run_id: str) -> None:
        """
        Wait for user resume signal.

        Args:
            run_id: Run identifier.
        """
        session = self._sessions.get(run_id)
        if not session:
            return

        # If auto-resume is configured, set up timeout
        if session.config.auto_resume_timeout_s > 0:

            async def auto_resume():
                await asyncio.sleep(session.config.auto_resume_timeout_s)
                if session.paused:
                    await session.resume()

            asyncio.create_task(auto_resume())

        # Wait until not paused
        while session.paused:
            await asyncio.sleep(0.1)

    def get_snapshots(self, run_id: str) -> list[StepSnapshot]:
        """Get all step snapshots for a run."""
        session = self._sessions.get(run_id)
        return list(session.step_snapshots) if session else []

    def get_breakpoints_hit(self, run_id: str) -> list[str]:
        """Get list of breakpoints hit during a run."""
        session = self._sessions.get(run_id)
        return list(session.breakpoints_hit) if session else []

    def set_resume_callback(
        self, run_id: str, callback: Callable[[], None]
    ) -> None:
        """Set callback to be called when resumed."""
        session = self._sessions.get(run_id)
        if session:
            session.resume_callback = callback

    async def resume(self, run_id: str) -> None:
        """Resume a paused debug session."""
        session = self._sessions.get(run_id)
        if session:
            await session.resume()
            if session.resume_callback:
                session.resume_callback()


# Global debugger instance
_global_debugger: InteractiveDebugger | None = None


def get_debugger() -> InteractiveDebugger:
    """Get the global debugger instance."""
    global _global_debugger
    if _global_debugger is None:
        _global_debugger = InteractiveDebugger()
    return _global_debugger