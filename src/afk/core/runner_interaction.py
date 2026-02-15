"""
Subagent and interaction/policy orchestration mixin.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from typing import Any

from ..agents.base import BaseAgent
from ..agents.errors import (
    AgentExecutionError,
    SubagentExecutionError,
    SubagentRoutingError,
)
from ..agents.policy import PolicyDecision as RulePolicyDecision
from ..agents.runtime import to_message_payload
from ..agents.types import (
    AgentRunEvent,
    AgentState,
    ApprovalDecision,
    ApprovalRequest,
    PolicyDecision,
    PolicyEvent,
    RouterDecision,
    RouterInput,
    SubagentExecutionRecord,
    UserInputDecision,
    UserInputRequest,
    json_value_from_tool_result,
)
from ..llms.types import Message
from ..memory import MemoryStore
from .runner_types import _RunHandle


class RunnerInteractionMixin:
    """Implements policy evaluation, subagent dispatch, and HITL requests."""

    async def _run_subagents(
        self,
        *,
        agent: BaseAgent,
        targets: list[str],
        parallel: bool,
        context: dict[str, Any],
        thread_id: str,
        depth: int,
        lineage: tuple[int, ...],
        run_id: str,
        step: int,
        handle: _RunHandle,
        memory: MemoryStore,
        user_id: str | None,
    ) -> tuple[list[SubagentExecutionRecord], str]:
        """
        Execute selected subagents and return records plus bridge text.

        Args:
            agent: Parent agent declaring available subagents.
            targets: Subagent names selected by router/policy.
            parallel: Whether selected subagents run in parallel.
            context: Parent run context snapshot.
            thread_id: Thread id used for child run continuity.
            depth: Current nesting depth.
            lineage: Current lineage tuple for nested tracing.
            run_id: Parent run identifier.
            step: Parent step number.
            handle: Parent run handle for event emission.
            memory: Memory backend for emitted events.
            user_id: Optional user id propagated to memory events.

        Returns:
            Tuple of execution records and bridge text inserted back into parent
            transcript.
        """
        index = {sa.name: sa for sa in agent.subagents}
        selected: list[BaseAgent] = []
        for name in targets:
            sa = index.get(name)
            if sa is None:
                raise SubagentRoutingError(f"Unknown subagent target '{name}'")
            selected.append(sa)
        if not parallel and selected:
            selected = selected[:1]

        async def _one(sub: BaseAgent) -> tuple[SubagentExecutionRecord, str]:
            """Execute one subagent and return execution record plus bridge text."""
            started = time.time()
            await self._emit(
                handle,
                memory,
                AgentRunEvent(
                    type="subagent_started",
                    run_id=run_id,
                    thread_id=thread_id,
                    state="running",
                    step=step,
                    data={"subagent_name": sub.name},
                ),
                user_id=user_id,
            )
            try:
                decision = await self._evaluate_policy(
                    agent=agent,
                    event=PolicyEvent(
                        event_type="subagent_before_execute",
                        run_id=run_id,
                        thread_id=thread_id,
                        step=step,
                        context={
                            str(k): json_value_from_tool_result(v)
                            for k, v in context.items()
                        },
                        subagent_name=sub.name,
                    ),
                    handle=handle,
                    memory=memory,
                    user_id=user_id,
                    state="running",
                )
                if decision.action == "deny":
                    raise SubagentExecutionError(
                        decision.reason or f"Subagent '{sub.name}' denied by policy"
                    )

                inherited = self._build_subagent_context(
                    context=context,
                    inherit_keys=sub.inherit_context_keys,
                )
                sub_handle = await self.run_handle(
                    sub,
                    context=inherited,
                    thread_id=thread_id,
                    _depth=depth,
                    _lineage=lineage,
                )
                out = await sub_handle.await_result()
                if out is None:
                    raise SubagentExecutionError(f"Subagent '{sub.name}' cancelled")
                latency_ms = (time.time() - started) * 1000.0
                rec = SubagentExecutionRecord(
                    subagent_name=sub.name,
                    success=True,
                    output_text=out.final_text,
                    latency_ms=latency_ms,
                )
                bridge = f"Subagent '{sub.name}' result:\n{out.final_text}"
                await self._emit(
                    handle,
                    memory,
                    AgentRunEvent(
                        type="subagent_completed",
                        run_id=run_id,
                        thread_id=thread_id,
                        state="running",
                        step=step,
                        data={"subagent_name": sub.name, "success": True},
                    ),
                    user_id=user_id,
                )
                return rec, bridge
            except Exception as e:
                latency_ms = (time.time() - started) * 1000.0
                rec = SubagentExecutionRecord(
                    subagent_name=sub.name,
                    success=False,
                    error=str(e),
                    latency_ms=latency_ms,
                )
                bridge = f"Subagent '{sub.name}' failed: {e}"
                await self._emit(
                    handle,
                    memory,
                    AgentRunEvent(
                        type="subagent_completed",
                        run_id=run_id,
                        thread_id=thread_id,
                        state="running",
                        step=step,
                        data={
                            "subagent_name": sub.name,
                            "success": False,
                            "error": str(e),
                        },
                    ),
                    user_id=user_id,
                )
                return rec, bridge

        if parallel:
            rows = await asyncio.gather(*[_one(sa) for sa in selected])
        else:
            rows = [await _one(sa) for sa in selected]

        records = [row[0] for row in rows]
        bridges = [row[1] for row in rows]
        return records, "\n\n".join(bridges)

    async def _call_router(
        self,
        agent: BaseAgent,
        *,
        run_id: str,
        thread_id: str,
        step: int,
        context: dict[str, Any],
        messages: list[Message],
    ) -> RouterDecision:
        """
        Invoke configured subagent router and normalize response.

        Args:
            agent: Agent containing optional router callback.
            run_id: Current run identifier.
            thread_id: Current thread identifier.
            step: Current step index.
            context: Current run context.
            messages: Current transcript messages.

        Returns:
            Router decision; empty decision when no router is configured.

        Raises:
            SubagentRoutingError: If router returns invalid payload type.
        """
        router = agent.subagent_router
        if router is None:
            return RouterDecision()
        raw = router(
            RouterInput(
                run_id=run_id,
                thread_id=thread_id,
                step=step,
                context={
                    str(k): json_value_from_tool_result(v) for k, v in context.items()
                },
                messages=[
                    to_message_payload(
                        msg.role,
                        msg.content
                        if isinstance(msg.content, str)
                        else json.dumps(msg.content, ensure_ascii=True),
                    )
                    for msg in messages
                ],
            )
        )
        if inspect.isawaitable(raw):
            raw = await raw
        if not isinstance(raw, RouterDecision):
            raise SubagentRoutingError("subagent_router must return RouterDecision")
        return raw

    async def _evaluate_policy(
        self,
        *,
        agent: BaseAgent,
        event: PolicyEvent,
        handle: _RunHandle | None = None,
        memory: MemoryStore | None = None,
        user_id: str | None = None,
        state: AgentState = "running",
    ) -> PolicyDecision:
        """
        Evaluate policy engine and policy roles for runtime event.

        Args:
            agent: Active agent.
            event: Policy event payload under evaluation.
            handle: Optional run handle for policy audit events.
            memory: Optional memory store for policy audit persistence.
            user_id: Optional user id for emitted audit events.
            state: Current run state for audit events.

        Returns:
            Final policy decision.

        Raises:
            AgentExecutionError: If policy engine evaluation crashes.
        """
        decision = PolicyDecision(action="allow")

        engine = agent.policy_engine or self._policy_engine
        if engine is not None:
            try:
                evaluation = engine.evaluate(event)
                engine_decision = (
                    evaluation.decision
                    if hasattr(evaluation, "decision")
                    else evaluation
                )
            except Exception as e:
                raise AgentExecutionError(f"Policy engine evaluation failed: {e}") from e
            if isinstance(engine_decision, RulePolicyDecision):
                decision = engine_decision
                if decision.action in {
                    "deny",
                    "defer",
                    "request_approval",
                    "request_user_input",
                }:
                    await self._emit_policy_audit(
                        handle=handle,
                        memory=memory,
                        event=event,
                        decision=decision,
                        user_id=user_id,
                        state=state,
                    )
                    return decision

        for role in agent.policy_roles:
            out = role(event)
            if inspect.isawaitable(out):
                out = await out
            if isinstance(out, PolicyDecision):
                decision = out
                if decision.action in {
                    "deny",
                    "defer",
                    "request_approval",
                    "request_user_input",
                }:
                    await self._emit_policy_audit(
                        handle=handle,
                        memory=memory,
                        event=event,
                        decision=decision,
                        user_id=user_id,
                        state=state,
                    )
                    return decision
        await self._emit_policy_audit(
            handle=handle,
            memory=memory,
            event=event,
            decision=decision,
            user_id=user_id,
            state=state,
        )
        return decision

    async def _emit_policy_audit(
        self,
        *,
        handle: _RunHandle | None,
        memory: MemoryStore | None,
        event: PolicyEvent,
        decision: PolicyDecision,
        user_id: str | None,
        state: AgentState,
    ) -> None:
        """
        Emit normalized policy-decision audit event.

        Args:
            handle: Active run handle.
            memory: Memory store used for persistence.
            event: Policy event that was evaluated.
            decision: Decision produced by policy layer.
            user_id: Optional user id for event persistence.
            state: Current run state.
        """
        if handle is None or memory is None:
            return
        await self._emit(
            handle,
            memory,
            AgentRunEvent(
                type="policy_decision",
                run_id=event.run_id,
                thread_id=event.thread_id,
                state=state,
                step=event.step,
                data={
                    "event_type": event.event_type,
                    "action": decision.action,
                    "reason": decision.reason,
                    "policy_id": decision.policy_id,
                    "matched_rules": [str(item) for item in decision.matched_rules],
                },
            ),
            user_id=user_id,
        )

    async def _request_approval(
        self,
        *,
        handle: _RunHandle,
        memory: MemoryStore,
        run_id: str,
        thread_id: str,
        step: int,
        reason: str,
        payload: dict[str, Any],
        user_id: str | None,
    ) -> bool:
        """
        Request approval and handle deferred pause/resume flow.

        Args:
            handle: Active run handle.
            memory: Memory store for pause/resume checkpoints.
            run_id: Current run identifier.
            thread_id: Current thread identifier.
            step: Current step index.
            reason: Human-readable reason for approval.
            payload: JSON-like request payload.
            user_id: Optional user id for event persistence.

        Returns:
            `True` when approved; otherwise `False`.
        """
        req = ApprovalRequest(
            run_id=run_id,
            thread_id=thread_id,
            step=step,
            reason=reason,
            payload={str(k): json_value_from_tool_result(v) for k, v in payload.items()},
        )
        decision = await self._interaction.request_approval(req)
        if isinstance(decision, ApprovalDecision):
            return decision.kind == "allow"

        await self._emit(
            handle,
            memory,
            AgentRunEvent(
                type="run_paused",
                run_id=run_id,
                thread_id=thread_id,
                state="paused",
                step=step,
                message="Waiting for deferred approval",
            ),
            user_id=user_id,
        )
        await self._persist_checkpoint(
            memory=memory,
            thread_id=thread_id,
            run_id=run_id,
            step=step,
            phase="paused",
            payload={"kind": "approval", "reason": reason},
        )
        wait_started_s = time.time()
        wait_span = self._telemetry_start_span(
            "agent.interaction.wait",
            attributes={
                "run_id": run_id,
                "thread_id": thread_id,
                "step": step,
                "kind": "approval",
            },
        )
        deferred = await self._interaction.await_deferred(
            decision.token,
            timeout_s=self.config.approval_timeout_s,
        )
        wait_latency_ms = (time.time() - wait_started_s) * 1000.0
        self._telemetry_histogram(
            "agent.interaction.wait_ms",
            value=wait_latency_ms,
            attributes={"kind": "approval"},
        )
        self._telemetry_counter(
            "agent.interaction.wait.total",
            value=1,
            attributes={
                "kind": "approval",
                "result": "resolved" if deferred is not None else "timeout",
            },
        )
        self._telemetry_end_span(
            wait_span,
            status="ok" if deferred is not None else "error",
            error=None if deferred is not None else "timeout",
            attributes={
                "kind": "approval",
                "wait_ms": wait_latency_ms,
            },
        )
        await self._emit(
            handle,
            memory,
            AgentRunEvent(
                type="run_resumed",
                run_id=run_id,
                thread_id=thread_id,
                state="running",
                step=step,
                message="Deferred approval resolved",
            ),
            user_id=user_id,
        )
        await self._persist_checkpoint(
            memory=memory,
            thread_id=thread_id,
            run_id=run_id,
            step=step,
            phase="resumed",
            payload={"kind": "approval"},
        )
        if isinstance(deferred, ApprovalDecision):
            return deferred.kind == "allow"
        return self.config.approval_fallback == "allow"

    async def _request_user_input(
        self,
        *,
        handle: _RunHandle,
        memory: MemoryStore,
        run_id: str,
        thread_id: str,
        step: int,
        prompt: str,
        payload: dict[str, Any],
        user_id: str | None,
    ) -> UserInputDecision:
        """
        Request user input and handle deferred pause/resume flow.

        Args:
            handle: Active run handle.
            memory: Memory store for pause/resume checkpoints.
            run_id: Current run identifier.
            thread_id: Current thread identifier.
            step: Current step index.
            prompt: Prompt shown to human operator.
            payload: JSON-like request payload.
            user_id: Optional user id for event persistence.

        Returns:
            Resolved user-input decision (or fallback on timeout).
        """
        req = UserInputRequest(
            run_id=run_id,
            thread_id=thread_id,
            step=step,
            prompt=prompt,
            payload={str(k): json_value_from_tool_result(v) for k, v in payload.items()},
        )
        decision = await self._interaction.request_user_input(req)
        if isinstance(decision, UserInputDecision):
            return decision

        await self._emit(
            handle,
            memory,
            AgentRunEvent(
                type="run_paused",
                run_id=run_id,
                thread_id=thread_id,
                state="paused",
                step=step,
                message="Waiting for deferred user input",
            ),
            user_id=user_id,
        )
        await self._persist_checkpoint(
            memory=memory,
            thread_id=thread_id,
            run_id=run_id,
            step=step,
            phase="paused",
            payload={"kind": "user_input", "prompt": prompt},
        )
        wait_started_s = time.time()
        wait_span = self._telemetry_start_span(
            "agent.interaction.wait",
            attributes={
                "run_id": run_id,
                "thread_id": thread_id,
                "step": step,
                "kind": "user_input",
            },
        )
        deferred = await self._interaction.await_deferred(
            decision.token,
            timeout_s=self.config.input_timeout_s,
        )
        wait_latency_ms = (time.time() - wait_started_s) * 1000.0
        self._telemetry_histogram(
            "agent.interaction.wait_ms",
            value=wait_latency_ms,
            attributes={"kind": "user_input"},
        )
        self._telemetry_counter(
            "agent.interaction.wait.total",
            value=1,
            attributes={
                "kind": "user_input",
                "result": "resolved" if deferred is not None else "timeout",
            },
        )
        self._telemetry_end_span(
            wait_span,
            status="ok" if deferred is not None else "error",
            error=None if deferred is not None else "timeout",
            attributes={
                "kind": "user_input",
                "wait_ms": wait_latency_ms,
            },
        )
        await self._emit(
            handle,
            memory,
            AgentRunEvent(
                type="run_resumed",
                run_id=run_id,
                thread_id=thread_id,
                state="running",
                step=step,
                message="Deferred user input resolved",
            ),
            user_id=user_id,
        )
        await self._persist_checkpoint(
            memory=memory,
            thread_id=thread_id,
            run_id=run_id,
            step=step,
            phase="resumed",
            payload={"kind": "user_input"},
        )
        if isinstance(deferred, UserInputDecision):
            return deferred
        return UserInputDecision(kind=self.config.input_fallback, reason="input_timeout")

    def _is_defer_user_input(self, decision: PolicyDecision) -> bool:
        """
        Check whether a defer decision represents user-input interaction.

        Args:
            decision: Policy decision payload.

        Returns:
            `True` when defer marker maps to user input.
        """
        marker = decision.request_payload.get("interaction")
        if not isinstance(marker, str):
            return False
        return marker.strip().lower() in {"user_input", "input"}

    def _resolve_user_input_prompt(
        self,
        *,
        tool_name: str,
        decision: PolicyDecision,
    ) -> str:
        """
        Resolve prompt text used for user-input interaction requests.

        Args:
            tool_name: Tool name requiring user input.
            decision: Policy decision containing optional prompt metadata.

        Returns:
            Prompt string to show to interaction provider.
        """
        payload_prompt = decision.request_payload.get("prompt")
        if isinstance(payload_prompt, str) and payload_prompt.strip():
            return payload_prompt.strip()
        if isinstance(decision.reason, str) and decision.reason.strip():
            return decision.reason.strip()
        return f"Provide input for tool '{tool_name}'"
