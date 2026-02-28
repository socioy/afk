"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Delegation orchestration engine using planner/validator/scheduler/executor stages.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable

from ...agents.contracts import (
    AgentCommunicationProtocol,
    AgentInvocationRequest,
)
from ...agents.delegation import (
    DelegationFinalStatus,
    DelegationNode,
    DelegationNodeResult,
    DelegationPlan,
    DelegationResult,
)
from ...llms.types import JSONValue
from .dispatcher import DelegationPlanner, DelegationScheduler, GraphValidator


class DelegationExecutor:
    """Execute individual delegation nodes with timeout and retry controls."""

    async def execute_node(
        self,
        *,
        node: DelegationNode,
        payload: dict[str, JSONValue],
        protocol: AgentCommunicationProtocol,
        request_factory: Callable[
            [DelegationNode, dict[str, JSONValue], int], AgentInvocationRequest
        ],
    ) -> DelegationNodeResult:
        """Run one node and return terminal status/result."""
        attempts = max(1, node.retry_policy.max_attempts)
        started_ms = int(time.time() * 1000)
        last_error = "unknown"
        terminal_status = "failed"
        last_request: AgentInvocationRequest | None = None
        dead_letter_eligible = True

        for attempt in range(1, attempts + 1):
            request = request_factory(node, payload, attempt)
            last_request = request
            try:
                if node.timeout_s is None:
                    response = await protocol.invoke(request)
                else:
                    async with asyncio.timeout(node.timeout_s):
                        response = await protocol.invoke(request)
            except TimeoutError:
                terminal_status = "timeout"
                last_error = f"Delegation node '{node.node_id}' timed out after {node.timeout_s:.2f}s"
            except asyncio.CancelledError:
                finished_ms = int(time.time() * 1000)
                return DelegationNodeResult(
                    node_id=node.node_id,
                    target_agent=node.target_agent,
                    status="cancelled",
                    success=False,
                    attempts=attempt,
                    error="Cancelled by parent control flow",
                    started_at_ms=started_ms,
                    finished_at_ms=finished_ms,
                )
            except Exception as exc:  # pragma: no cover - defensive branch
                terminal_status = "failed"
                last_error = str(exc)
            else:
                if response.success:
                    finished_ms = int(time.time() * 1000)
                    return DelegationNodeResult(
                        node_id=node.node_id,
                        target_agent=node.target_agent,
                        status="completed",
                        success=True,
                        attempts=attempt,
                        output=response.output,
                        metadata=response.metadata,
                        started_at_ms=started_ms,
                        finished_at_ms=finished_ms,
                    )
                terminal_status = "failed"
                last_error = response.error or "Subagent returned unsuccessful response"
                retryable = response.metadata.get("retryable", True)
                if retryable is False:
                    attempts = attempt
                    dead_letter_eligible = False
                    break

            if attempt < attempts:
                await asyncio.sleep(self._backoff_delay(node=node, attempt=attempt))

        if (
            dead_letter_eligible
            and last_request is not None
            and hasattr(protocol, "record_dead_letter")
            and callable(protocol.record_dead_letter)
        ):
            try:
                await protocol.record_dead_letter(
                    last_request,
                    error=last_error,
                    attempts=attempts,
                )
            except Exception:
                pass  # dead-letter recording must never mask the original failure

        finished_ms = int(time.time() * 1000)
        return DelegationNodeResult(
            node_id=node.node_id,
            target_agent=node.target_agent,
            status="timeout" if terminal_status == "timeout" else "failed",
            success=False,
            attempts=attempts,
            error=last_error,
            started_at_ms=started_ms,
            finished_at_ms=finished_ms,
        )

    def _backoff_delay(self, *, node: DelegationNode, attempt: int) -> float:
        policy = node.retry_policy
        base = max(0.0, policy.backoff_base_s)
        cap = max(base, policy.max_backoff_s)
        delay = min(cap, base * (2 ** max(0, attempt - 1)))
        jitter = random.uniform(0.0, max(0.0, policy.jitter_s))
        return delay + jitter


class DelegationAggregator:
    """Aggregate node results and evaluate join policy to produce final status."""

    def aggregate(
        self,
        *,
        plan: DelegationPlan,
        topological_order: list[str],
        node_results: dict[str, DelegationNodeResult],
    ) -> DelegationResult:
        """Build deterministic fan-in result from per-node execution results."""
        ordered = [
            node_results[node_id]
            for node_id in topological_order
            if node_id in node_results
        ]

        success_count = sum(1 for result in ordered if result.success)
        failure_count = sum(1 for result in ordered if not result.success)

        cancelled_count = sum(1 for result in ordered if result.status == "cancelled")
        cancellation_terminal = (
            bool(ordered)
            and cancelled_count > 0
            and all(result.status in {"cancelled", "skipped"} for result in ordered)
        )
        if cancellation_terminal:
            final_status: DelegationFinalStatus = "cancelled"
        elif plan.join_policy == "first_success":
            final_status = "completed" if success_count > 0 else "failed"
        elif plan.join_policy == "quorum":
            quorum = (
                plan.quorum if isinstance(plan.quorum, int) and plan.quorum > 0 else 1
            )
            final_status = "completed" if success_count >= quorum else "failed"
        elif plan.join_policy == "allow_optional_failures":
            required_failures = any(
                (not node_results[node.node_id].success)
                for node in plan.nodes
                if node.required and node.node_id in node_results
            )
            if required_failures:
                final_status = "failed"
            elif failure_count > 0:
                final_status = "degraded"
            else:
                final_status = "completed"
        else:
            required_failures = any(
                (not node_results[node.node_id].success)
                for node in plan.nodes
                if node.required and node.node_id in node_results
            )
            final_status = "failed" if required_failures else "completed"

        return DelegationResult(
            node_results=node_results,
            ordered_outputs=ordered,
            final_status=final_status,
            success_count=success_count,
            failure_count=failure_count,
        )


class DelegationEngine:
    """Full orchestration pipeline: planner -> validator -> scheduler -> aggregator."""

    def __init__(
        self,
        *,
        max_parallel_subagents_global: int,
        max_parallel_subagents_per_parent: int,
        max_parallel_subagents_per_target_agent: int,
        subagent_queue_backpressure_limit: int,
        planner: DelegationPlanner | None = None,
        validator: GraphValidator | None = None,
        scheduler: DelegationScheduler | None = None,
        executor: DelegationExecutor | None = None,
        aggregator: DelegationAggregator | None = None,
    ) -> None:
        self.planner = planner or DelegationPlanner()
        self.validator = validator or GraphValidator()
        self.scheduler = scheduler or DelegationScheduler(
            max_parallel_subagents_global=max_parallel_subagents_global,
            max_parallel_subagents_per_parent=max_parallel_subagents_per_parent,
            max_parallel_subagents_per_target_agent=max_parallel_subagents_per_target_agent,
            subagent_queue_backpressure_limit=subagent_queue_backpressure_limit,
        )
        self.executor = executor or DelegationExecutor()
        self.aggregator = aggregator or DelegationAggregator()

    async def execute(
        self,
        *,
        plan: DelegationPlan,
        available_targets: set[str],
        protocol: AgentCommunicationProtocol,
        request_factory: Callable[
            [DelegationNode, dict[str, JSONValue], int], AgentInvocationRequest
        ],
        cancel_requested: Callable[[], bool] | None = None,
        interrupt_requested: Callable[[], bool] | None = None,
    ) -> tuple[DelegationResult, list[dict[str, JSONValue]]]:
        """Execute a validated plan and return aggregated result plus audit rows."""
        topological_order = self.validator.validate(
            plan=plan,
            available_targets=available_targets,
        )

        node_results, audit_rows = await self.scheduler.execute(
            plan=plan,
            topological_order=topological_order,
            execute_node=lambda node, payload: self.executor.execute_node(
                node=node,
                payload=payload,
                protocol=protocol,
                request_factory=request_factory,
            ),
            cancel_requested=cancel_requested,
            interrupt_requested=interrupt_requested,
        )
        aggregated = self.aggregator.aggregate(
            plan=plan,
            topological_order=topological_order,
            node_results=node_results,
        )
        return aggregated, audit_rows
