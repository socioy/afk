# Evals

Enterprise eval case and suite execution with adaptive scheduling, assertions, and budgets.

Source: `docs/library/evals.mdx`

The `afk.evals` package provides a contract-driven eval system for deterministic case execution, suite orchestration, budget enforcement, scoring, and report export.

## TL;DR

- Use `EvalCase` for one runnable scenario.
- Use `run_case(...)` / `arun_case(...)` for single-case runs.
- Use `run_suite(...)` / `arun_suite(...)` for multi-case execution with `adaptive`, `sequential`, or `parallel` scheduling.
- Add assertions/scorers and optional `EvalBudget` to enforce enterprise quality gates.
- Emit machine-readable suite reports with `write_suite_report_json(...)`.

## Quick Start

```python
from afk.agents import Agent
from afk.core import Runner
from afk.evals import (
    EvalCase,
    EvalSuiteConfig,
    FinalTextContainsAssertion,
    EvalBudget,
    run_suite,
)

cases = [
    EvalCase(
        name="smoke-1",
        agent=Agent(model="gpt-4.1-mini", instructions="Say hello"),
        user_message="hello",
    )
]

suite = run_suite(
    runner_factory=Runner,
    cases=cases,
    config=EvalSuiteConfig(
        execution_mode="adaptive",
        assertions=(FinalTextContainsAssertion("hello"),),
        budget=EvalBudget(max_duration_s=5.0),
        fail_fast=True,
    ),
)

print(suite.total, suite.passed, suite.failed)
```

## Execution Modes

- `adaptive`: resolves to `sequential` for small suites and `parallel` for larger workloads.
- `sequential`: one case at a time, deterministic and easiest to debug.
- `parallel`: bounded concurrency with deterministic output ordering.

## Dataset Loader

Load case definitions from JSON using `load_eval_cases_json(...)`, then resolve `agent` names with your resolver.

## Reporting

- `suite_report_payload(...)` returns a schema-versioned report dictionary.
- `write_suite_report_json(...)` writes a JSON report envelope (`eval_suite.v1`).

## Continue Reading

1. [Observability](/library/observability)
2. [Run Events](/library/run-event-contract)
3. [API Reference](/library/api-reference)
