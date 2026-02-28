"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

DAG delegation planner, validator, and parallel scheduler.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace

from ...agents.delegation import (
    DelegationNode,
    DelegationNodeResult,
    DelegationPlan,
    RetryPolicy,
)
from ...llms.types import JSONValue


class DelegationGraphError(ValueError):
    """Raised when a delegation graph is invalid."""


class DelegationBackpressureError(RuntimeError):
    """Raised when ready queue exceeds configured backpressure limits."""


class DelegationPlanner:
    """Build simple DAG plans from selected subagent targets."""

    def create_plan(
        self,
        *,
        targets: list[str],
        parallel: bool,
        default_timeout_s: float | None = 120.0,
        default_retry_policy: RetryPolicy | None = None,
        max_parallelism: int | None = None,
    ) -> DelegationPlan:
        """Create a deterministic fanout plan from target names."""
        counts: dict[str, int] = {}
        nodes: list[DelegationNode] = []
        retry = default_retry_policy or RetryPolicy(max_attempts=1)

        for name in targets:
            normalized = name.strip()
            if not normalized:
                continue
            counts[normalized] = counts.get(normalized, 0) + 1
            suffix = counts[normalized]
            node_id = normalized if suffix == 1 else f"{normalized}#{suffix}"
            nodes.append(
                DelegationNode(
                    node_id=node_id,
                    target_agent=normalized,
                    timeout_s=default_timeout_s,
                    retry_policy=retry,
                    required=True,
                )
            )

        if not nodes:
            return DelegationPlan(
                nodes=[], edges=[], join_policy="all_required", max_parallelism=1
            )

        if max_parallelism is None:
            max_parallel = len(nodes) if parallel else 1
        else:
            max_parallel = max(1, max_parallelism)
        return DelegationPlan(
            nodes=nodes,
            edges=[],
            join_policy="all_required",
            max_parallelism=max_parallel,
        )


class GraphValidator:
    """Validate delegation DAG structure and produce stable topological order."""

    def validate(
        self,
        *,
        plan: DelegationPlan,
        available_targets: set[str],
    ) -> list[str]:
        """Validate plan shape and return stable topological ordering."""
        if plan.max_parallelism < 1:
            raise DelegationGraphError("DelegationPlan.max_parallelism must be >= 1")

        node_ids: set[str] = set()
        node_by_id: dict[str, DelegationNode] = {}
        for node in plan.nodes:
            if node.node_id in node_ids:
                raise DelegationGraphError(
                    f"Duplicate node_id '{node.node_id}' in delegation plan"
                )
            if node.target_agent not in available_targets:
                raise DelegationGraphError(
                    f"Unknown delegation target '{node.target_agent}' for node '{node.node_id}'"
                )
            node_ids.add(node.node_id)
            node_by_id[node.node_id] = node

        indegree: dict[str, int] = {node.node_id: 0 for node in plan.nodes}
        children: dict[str, list[str]] = {node.node_id: [] for node in plan.nodes}

        for edge in plan.edges:
            if edge.from_node not in node_ids:
                raise DelegationGraphError(
                    f"Edge source '{edge.from_node}' is not in delegation nodes"
                )
            if edge.to_node not in node_ids:
                raise DelegationGraphError(
                    f"Edge target '{edge.to_node}' is not in delegation nodes"
                )
            if edge.from_node == edge.to_node:
                raise DelegationGraphError(
                    f"Self-cycle is not allowed for node '{edge.from_node}'"
                )
            indegree[edge.to_node] += 1
            children[edge.from_node].append(edge.to_node)

        ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for child in sorted(children[current]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
            ready.sort()

        if len(order) != len(plan.nodes):
            raise DelegationGraphError("Delegation plan contains a cycle")

        return order


class DelegationScheduler:
    """Execute validated delegation plans with bounded parallelism and deterministic order."""

    def __init__(
        self,
        *,
        max_parallel_subagents_global: int,
        max_parallel_subagents_per_parent: int,
        max_parallel_subagents_per_target_agent: int,
        subagent_queue_backpressure_limit: int,
    ) -> None:
        self._global_semaphore = asyncio.Semaphore(
            max(1, max_parallel_subagents_global)
        )
        self._max_parallel_per_parent = max(1, max_parallel_subagents_per_parent)
        self._max_parallel_per_target = max(1, max_parallel_subagents_per_target_agent)
        self._backpressure_limit = max(1, subagent_queue_backpressure_limit)
        self._target_semaphores: dict[str, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()

    async def _target_semaphore(self, target: str) -> asyncio.Semaphore:
        async with self._lock:
            sem = self._target_semaphores.get(target)
            if sem is None:
                sem = asyncio.Semaphore(self._max_parallel_per_target)
                self._target_semaphores[target] = sem
            return sem

    async def execute(
        self,
        *,
        plan: DelegationPlan,
        topological_order: list[str],
        execute_node: Callable[
            [DelegationNode, dict[str, JSONValue]], Awaitable[DelegationNodeResult]
        ],
        cancel_requested: Callable[[], bool] | None = None,
        interrupt_requested: Callable[[], bool] | None = None,
    ) -> tuple[dict[str, DelegationNodeResult], list[dict[str, JSONValue]]]:
        """Execute nodes in deterministic DAG order with parallel fanout/fanin."""
        if not plan.nodes:
            return {}, []

        node_by_id = {node.node_id: node for node in plan.nodes}
        children: dict[str, list[str]] = {node.node_id: [] for node in plan.nodes}
        parents: dict[str, list[str]] = {node.node_id: [] for node in plan.nodes}
        indegree: dict[str, int] = {node.node_id: 0 for node in plan.nodes}
        edge_by_pair: dict[tuple[str, str], dict[str, str]] = {}

        for edge in plan.edges:
            children[edge.from_node].append(edge.to_node)
            parents[edge.to_node].append(edge.from_node)
            indegree[edge.to_node] += 1
            edge_by_pair[(edge.from_node, edge.to_node)] = dict(edge.output_key_map)

        ready = sorted(
            node_id for node_id in topological_order if indegree[node_id] == 0
        )
        running: dict[str, asyncio.Task[DelegationNodeResult]] = {}
        node_results: dict[str, DelegationNodeResult] = {}
        audit: list[dict[str, JSONValue]] = []
        cancelled = False

        def _is_control_cancelled() -> bool:
            return bool(cancel_requested and cancel_requested()) or bool(
                interrupt_requested and interrupt_requested()
            )

        def _mark_subtree_skipped(start_node: str, reason: str) -> None:
            stack = [start_node]
            while stack:
                current = stack.pop()
                if current in node_results:
                    continue
                node = node_by_id[current]
                node_results[current] = DelegationNodeResult(
                    node_id=current,
                    target_agent=node.target_agent,
                    status="skipped",
                    success=False,
                    attempts=0,
                    error=reason,
                )
                for child in children[current]:
                    stack.append(child)

        def _build_payload(node_id: str) -> dict[str, JSONValue]:
            payload = dict(node_by_id[node_id].input_binding)
            for parent_id in sorted(parents[node_id]):
                parent_result = node_results.get(parent_id)
                if parent_result is None:
                    continue
                if not parent_result.success:
                    continue
                mapping = edge_by_pair.get((parent_id, node_id), {})
                parent_output = parent_result.output
                if mapping and isinstance(parent_output, dict):
                    for source_key, target_key in mapping.items():
                        if source_key in parent_output:
                            payload[target_key] = parent_output[source_key]
                    continue
                if isinstance(parent_output, dict):
                    for key in sorted(parent_output.keys()):
                        if key not in payload:
                            payload[key] = parent_output[key]
                elif parent_output is not None:
                    payload[parent_id] = parent_output
            return payload

        async def _run_with_limits(
            node: DelegationNode, payload: dict[str, JSONValue]
        ) -> DelegationNodeResult:
            per_target = await self._target_semaphore(node.target_agent)
            async with self._global_semaphore, per_target:
                return await execute_node(node, payload)

        parent_parallelism = min(plan.max_parallelism, self._max_parallel_per_parent)
        while ready or running:
            if _is_control_cancelled() and not cancelled:
                cancelled = True
                for task in running.values():
                    task.cancel()

            while not cancelled and ready and len(running) < parent_parallelism:
                if len(ready) + len(running) > self._backpressure_limit:
                    raise DelegationBackpressureError(
                        "Subagent ready queue exceeded subagent_queue_backpressure_limit"
                    )
                node_id = ready.pop(0)
                if node_id in node_results:
                    continue
                node = node_by_id[node_id]

                blocked_parent = next(
                    (
                        parent_id
                        for parent_id in parents[node_id]
                        if parent_id in node_results
                        and not node_results[parent_id].success
                    ),
                    None,
                )
                if blocked_parent is not None:
                    _mark_subtree_skipped(
                        node_id,
                        f"Dependency '{blocked_parent}' did not complete successfully",
                    )
                    continue

                payload = _build_payload(node_id)
                running[node_id] = asyncio.create_task(_run_with_limits(node, payload))

            if not running:
                break

            done: set[asyncio.Task[DelegationNodeResult]] = set()
            while not done:
                if _is_control_cancelled() and not cancelled:
                    cancelled = True
                    for task in running.values():
                        task.cancel()
                done, _ = await asyncio.wait(
                    set(running.values()),
                    timeout=0.05,
                    return_when=asyncio.FIRST_COMPLETED,
                )

            for task in done:
                node_id = next(
                    key for key, candidate in running.items() if candidate is task
                )
                running.pop(node_id, None)

                node = node_by_id[node_id]
                if cancelled:
                    audit.append(
                        {
                            "type": "ignored_late_response",
                            "node_id": node_id,
                            "target_agent": node.target_agent,
                        }
                    )
                    if node_id not in node_results:
                        node_results[node_id] = DelegationNodeResult(
                            node_id=node_id,
                            target_agent=node.target_agent,
                            status="cancelled",
                            success=False,
                            attempts=0,
                            error="Cancelled by parent control flow",
                        )
                    continue

                try:
                    result = task.result()
                except asyncio.CancelledError:
                    result = DelegationNodeResult(
                        node_id=node_id,
                        target_agent=node.target_agent,
                        status="cancelled",
                        success=False,
                        attempts=0,
                        error="Cancelled by parent control flow",
                    )
                except Exception as exc:  # pragma: no cover - defensive branch
                    result = DelegationNodeResult(
                        node_id=node_id,
                        target_agent=node.target_agent,
                        status="failed",
                        success=False,
                        attempts=1,
                        error=str(exc),
                    )

                node_results[node_id] = replace(
                    result,
                    started_at_ms=result.started_at_ms,
                    finished_at_ms=result.finished_at_ms,
                )

                if not result.success:
                    for child in children[node_id]:
                        _mark_subtree_skipped(
                            child,
                            f"Dependency '{node_id}' did not complete successfully",
                        )
                    continue

                for child in children[node_id]:
                    indegree[child] -= 1
                    if (
                        indegree[child] == 0
                        and child not in node_results
                        and child not in running
                    ):
                        ready.append(child)
                ready.sort()

        if cancelled:
            for node in plan.nodes:
                if node.node_id not in node_results:
                    node_results[node.node_id] = DelegationNodeResult(
                        node_id=node.node_id,
                        target_agent=node.target_agent,
                        status="cancelled",
                        success=False,
                        attempts=0,
                        error="Cancelled by parent control flow",
                    )

        for node_id in topological_order:
            if node_id not in node_results:
                node = node_by_id[node_id]
                node_results[node_id] = DelegationNodeResult(
                    node_id=node_id,
                    target_agent=node.target_agent,
                    status="skipped",
                    success=False,
                    attempts=0,
                    error="Node was not scheduled",
                )

        return node_results, audit
