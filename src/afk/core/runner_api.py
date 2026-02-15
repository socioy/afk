"""
Public runner API and lifecycle entrypoints.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..agents.base import BaseAgent
from ..agents.errors import AgentCancelledError, AgentCheckpointCorruptionError, AgentConfigurationError
from ..agents.policy import PolicyEngine
from ..agents.runtime import EffectJournal, checkpoint_latest_key
from ..agents.types import AgentResult, AgentRunHandle
from ..memory import (
    MemoryCompactionResult,
    MemoryStore,
    RetentionPolicy,
    StateRetentionPolicy,
    compact_thread_memory,
)
from .interaction import HeadlessInteractionProvider, InteractionProvider
from .telemetry import NullTelemetrySink, TelemetrySink
from .runner_types import RunnerConfig, _RunHandle


class RunnerAPIMixin:
    """
    Public API surface for running, resuming, and compacting agent threads.

    This mixin owns dependency wiring (memory, interaction, policy, telemetry)
    and exposes the stable entrypoints used by `Agent.call(...)` and external
    runtime integrations.
    """

    def __init__(
        self,
        *,
        memory_store: MemoryStore | None = None,
        interaction_provider: InteractionProvider | None = None,
        policy_engine: PolicyEngine | None = None,
        telemetry: TelemetrySink | None = None,
        config: RunnerConfig | None = None,
    ) -> None:
        """
        Initialize a runner API surface with optional runtime dependencies.

        Args:
            memory_store: Memory backend. When `None`, runtime resolves from
                environment on first use and may fallback to in-memory.
            interaction_provider: Human-in-the-loop provider. Required when
                `config.interaction_mode` is not `headless`.
            policy_engine: Optional deterministic policy engine shared across runs.
            telemetry: Telemetry sink for counters/spans/events.
            config: Runner configuration. Defaults to `RunnerConfig()`.

        Raises:
            AgentConfigurationError: If interaction mode requires provider but
                none is supplied.
        """
        self.config = config or RunnerConfig()
        self._memory_store = memory_store
        self._owns_memory_store = memory_store is None
        self._memory_fallback_reason: str | None = None
        if interaction_provider is None:
            if self.config.interaction_mode == "headless":
                self._interaction = HeadlessInteractionProvider(
                    approval_fallback=self.config.approval_fallback,
                    input_fallback=self.config.input_fallback,
                )
            else:
                raise AgentConfigurationError(
                    "interaction_provider is required when interaction_mode is not 'headless'"
                )
        else:
            self._interaction = interaction_provider
        self._policy_engine = policy_engine
        self._telemetry = telemetry or NullTelemetrySink()
        self._effect_journal = EffectJournal()
        self._active_runs = 0

    async def compact_thread(
        self,
        *,
        thread_id: str,
        event_policy: RetentionPolicy | None = None,
        state_policy: StateRetentionPolicy | None = None,
    ) -> MemoryCompactionResult:
        """
        Compact retained memory records for a thread.

        Args:
            thread_id: Thread identifier whose memory should be compacted.
            event_policy: Optional override for event retention compaction.
            state_policy: Optional override for state retention compaction.

        Returns:
            Memory compaction summary for the thread.

        Raises:
            AgentConfigurationError: If `thread_id` is empty.
        """
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise AgentConfigurationError("thread_id must be a non-empty string")
        memory = await self._ensure_memory_store()
        return await compact_thread_memory(
            memory,
            thread_id=thread_id,
            event_policy=event_policy,
            state_policy=state_policy,
        )

    async def run(
        self,
        agent: BaseAgent,
        *,
        user_message: str | None = None,
        context: dict[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> AgentResult:
        """
        Execute an agent run and wait for terminal result.

        Args:
            agent: Agent definition to execute.
            user_message: Optional initial user message.
            context: Optional JSON-like run context.
            thread_id: Optional thread id for memory continuity.

        Returns:
            Terminal agent result.

        Raises:
            AgentCancelledError: If run is cancelled before completion.
        """
        handle = await self.run_handle(
            agent,
            user_message=user_message,
            context=context,
            thread_id=thread_id,
        )
        result = await handle.await_result()
        if result is None:
            raise AgentCancelledError("Run cancelled")
        return result

    async def resume(
        self,
        agent: BaseAgent,
        *,
        run_id: str,
        thread_id: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Resume a previously checkpointed run and wait for completion.

        Args:
            agent: Agent definition to resume.
            run_id: Existing run identifier.
            thread_id: Existing thread identifier.
            context: Optional context overlay for resumed execution.

        Returns:
            Terminal agent result.

        Raises:
            AgentCancelledError: If resumed run is cancelled.
        """
        handle = await self.resume_handle(
            agent,
            run_id=run_id,
            thread_id=thread_id,
            context=context,
        )
        result = await handle.await_result()
        if result is None:
            raise AgentCancelledError("Run cancelled")
        return result

    async def resume_handle(
        self,
        agent: BaseAgent,
        *,
        run_id: str,
        thread_id: str,
        context: dict[str, Any] | None = None,
    ) -> AgentRunHandle:
        """
        Resume a run and return a live handle for lifecycle control.

        If the latest checkpoint already contains a terminal result, this method
        returns a handle pre-populated with that result.

        Args:
            agent: Agent definition used for continued execution.
            run_id: Existing run identifier.
            thread_id: Existing thread identifier.
            context: Optional context overlay for resumed execution.

        Returns:
            Active run handle or pre-resolved terminal handle.

        Raises:
            AgentConfigurationError: If `run_id`/`thread_id` is invalid.
            AgentCheckpointCorruptionError: If checkpoint chain is missing or
                invalid for the given run.
        """
        if not isinstance(run_id, str) or not run_id.strip():
            raise AgentConfigurationError("run_id must be a non-empty string")
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise AgentConfigurationError("thread_id must be a non-empty string")

        memory = await self._ensure_memory_store()
        latest = await memory.get_state(thread_id, checkpoint_latest_key(run_id))
        if not isinstance(latest, dict):
            raise AgentCheckpointCorruptionError(
                f"No checkpoint found for run_id={run_id} thread_id={thread_id}"
            )
        latest = self._normalize_checkpoint_record(latest)

        payload = latest.get("payload")
        phase = latest.get("phase")
        if isinstance(payload, dict):
            terminal = payload.get("terminal_result")
            if phase == "run_terminal" and isinstance(terminal, dict):
                handle = _RunHandle()
                result = self._deserialize_agent_result(terminal)
                await handle.set_result(result)
                return handle

        resume_snapshot = await self._load_latest_runtime_snapshot(
            memory=memory,
            thread_id=thread_id,
            run_id=run_id,
            latest=latest,
        )

        return await self.run_handle(
            agent,
            context=context,
            thread_id=thread_id,
            _resume_run_id=run_id,
            _resume_snapshot=resume_snapshot,
        )

    async def run_handle(
        self,
        agent: BaseAgent,
        *,
        user_message: str | None = None,
        context: dict[str, Any] | None = None,
        thread_id: str | None = None,
        _depth: int = 0,
        _lineage: tuple[int, ...] = (),
        _resume_run_id: str | None = None,
        _resume_snapshot: dict[str, Any] | None = None,
    ) -> AgentRunHandle:
        """
        Start execution and return an async run handle.

        Args:
            agent: Agent definition to execute.
            user_message: Optional initial user message.
            context: Optional JSON-like run context.
            thread_id: Optional thread id for memory continuity.
            _depth: Internal recursion depth for subagent execution.
            _lineage: Internal lineage tuple used for tracing nested runs.
            _resume_run_id: Internal run id for resume continuation.
            _resume_snapshot: Internal restored snapshot payload.

        Returns:
            Handle exposing event stream and run lifecycle controls.
        """
        handle = _RunHandle()
        task = asyncio.create_task(
            self._execute(
                handle,
                agent,
                user_message=user_message,
                context=context,
                thread_id=thread_id,
                depth=_depth,
                lineage=_lineage,
                resume_run_id=_resume_run_id,
                resume_snapshot=_resume_snapshot,
            )
        )
        handle.attach_task(task)
        return handle
