
# Data Pipeline

A data pipeline orchestrator agent that uses DelegationPlan for DAG-based multi-agent execution with parallel stages, dependencies, and retry policies.

## Project Structure

```
24_Data_Pipeline/
  main.py       # Entry point — orchestrator agent and execution loop
  stages.py     # Pipeline stage agents: extractor, validator, transformer, reporter
  pipeline.py   # DelegationPlan DAG: nodes, edges, retry policies, join policy
```

## Key Concepts

- **DelegationPlan**: Defines a DAG of agent invocations with nodes, edges, and execution constraints
- **DelegationNode**: Each node targets an agent with input bindings, timeout, and optional RetryPolicy
- **DelegationEdge**: Expresses dependencies between nodes (scheduling order)
- **join_policy**: "all_required", "first_success", "quorum", or "allow_optional_failures"
- **max_parallelism**: Controls how many agents run concurrently

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/24_Data_Pipeline

Pipeline DAG:
```
    [extract] ──┐
                ├──> [transform] ──> [report]
    [validate] ─┘
```

- extract and validate run in parallel (max_parallelism=2)
- transform waits for both to complete
- report waits for transform
