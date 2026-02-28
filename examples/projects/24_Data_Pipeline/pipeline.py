"""
---
name: Data Pipeline — DelegationPlan
description: DAG definition for the data pipeline using DelegationPlan with nodes, edges, retry policies, and join semantics.
tags: [delegation-plan, dag, pipeline, retry]
---
---
The DelegationPlan defines the full pipeline as a directed acyclic graph (DAG). Each
DelegationNode represents one agent invocation with optional input bindings, timeout, and
retry policy. DelegationEdge connects nodes to express dependencies — the delegation engine
handles scheduling, parallelism, and result collection automatically.

Pipeline DAG visualization:

    [extract] ──┐
                ├──> [transform] ──> [report]
    [validate] ─┘

- extract and validate run in parallel (max_parallelism=2)
- transform waits for both extract and validate to complete
- report waits for transform to complete
---
"""

from afk.agents.delegation import (  # <- Delegation system for DAG-based orchestration.
    DelegationPlan,  # <- The plan: a list of nodes, edges, and execution policy.
    DelegationNode,  # <- A node represents one agent invocation in the DAG.
    DelegationEdge,  # <- An edge represents a dependency between nodes (data flow).
    RetryPolicy,  # <- Per-node retry configuration (max attempts, backoff).
)


# ===========================================================================
# DelegationPlan — defines the pipeline DAG
# ===========================================================================

pipeline_plan = DelegationPlan(  # <- The DelegationPlan defines the full pipeline as a DAG. Nodes are agent invocations, edges are dependencies.
    nodes=[
        DelegationNode(  # <- Each node specifies a target agent, optional input bindings, timeout, and retry policy.
            node_id="extract",
            target_agent="data-extractor",
            input_binding={"task": "Extract all employee records from the data source"},  # <- Input bindings are passed as context to the agent.
            timeout_s=30.0,
            retry_policy=RetryPolicy(max_attempts=2, backoff_base_s=1.0),  # <- Retry up to 2 times with 1-second backoff if the agent fails.
        ),
        DelegationNode(
            node_id="validate",
            target_agent="data-validator",
            input_binding={"task": "Validate all extracted records for data quality"},
            timeout_s=30.0,
            retry_policy=RetryPolicy(max_attempts=2),
        ),
        DelegationNode(
            node_id="transform",
            target_agent="data-transformer",
            input_binding={"task": "Aggregate data by department with averages"},
            timeout_s=30.0,
        ),
        DelegationNode(
            node_id="report",
            target_agent="report-generator",
            input_binding={"task": "Generate executive summary report"},
            timeout_s=30.0,
        ),
    ],
    edges=[
        # --- extract and validate can run in parallel (no edges between them) ---
        DelegationEdge(from_node="extract", to_node="transform"),  # <- transform depends on extract completing first.
        DelegationEdge(from_node="validate", to_node="transform"),  # <- transform also depends on validate (both must finish before transform starts).
        DelegationEdge(from_node="transform", to_node="report"),  # <- report depends on transform.
    ],
    join_policy="all_required",  # <- All nodes must succeed for the plan to be considered successful. Other options: "first_success", "quorum", "allow_optional_failures".
    max_parallelism=2,  # <- At most 2 agents run concurrently. extract + validate run in parallel since they have no dependency.
)
