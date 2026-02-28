# Evals and Testing

AFK provides a structured eval framework for testing agent behavior deterministically.
`EvalCase` defines individual test scenarios, `EvalSuiteConfig` controls suite execution,
and pluggable assertions/scorers validate results against expectations.

- Docs: https://afk.arpan.sh/library/evals | https://afk.arpan.sh/library/tested-behaviors
- Source: `src/afk/evals/models.py`, `src/afk/evals/contracts.py`, `src/afk/evals/datasets.py`
- Cross-refs: `agents-and-runner.md`, `tools-system.md`, `cookbook-examples.md`

---

## Overview

The eval framework follows a three-layer model:

| Layer | Role | Key Type |
|-------|------|----------|
| **Case** | One test scenario: agent + input + optional context | `EvalCase` |
| **Suite** | Orchestrates many cases with execution mode, concurrency, and fail-fast | `EvalSuiteConfig` / `arun_suite` |
| **Assertions & Scorers** | Validate case results (pass/fail assertions, numeric scorers) | `EvalAssertion` / `EvalScorer` protocols |

A case passes when the agent state is `"completed"`, all assertions pass, and no budget violations occur.

---

## Quick Start

```python
from afk.agents import Agent
from afk.core.runner import Runner
from afk.evals import (
    EvalCase,
    EvalSuiteConfig,
    FinalTextContainsAssertion,
    StateCompletedAssertion,
    arun_suite,
)

case = EvalCase(
    name="greeting-test",
    agent=Agent(model="gpt-4.1-mini", instructions="Greet the user."),
    user_message="Say hello",
)

result = await arun_suite(
    runner_factory=Runner,
    cases=[case],
    config=EvalSuiteConfig(
        assertions=(StateCompletedAssertion(), FinalTextContainsAssertion("hello")),
    ),
)
print(f"Passed: {result.passed}/{result.total}")
```

---

## EvalCase

`EvalCase` is a frozen dataclass defining one eval scenario.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | **(required)** | Unique test case identifier |
| `agent` | `BaseAgent` | **(required)** | Agent instance to execute |
| `user_message` | `str \| None` | `None` | User message to send to agent |
| `context` | `dict[str, JSONValue]` | `{}` | Optional run context passed to the runner |
| `thread_id` | `str \| None` | `None` | Optional thread for memory continuity |
| `tags` | `tuple[str, ...]` | `()` | Tags for filtering and grouping |
| `budget` | `EvalBudget \| None` | `None` | Per-case budget overrides (takes precedence over suite budget) |

```python
from afk.evals import EvalCase, EvalBudget

case = EvalCase(
    name="bounded-case",
    agent=my_agent,
    user_message="Summarize this document.",
    context={"doc_id": "abc123"},
    tags=("smoke", "summarization"),
    budget=EvalBudget(max_duration_s=30.0, max_total_cost_usd=0.10),
)
```

---

## EvalSuiteConfig

Suite-level configuration controlling how cases are executed.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `execution_mode` | `ExecutionMode` | `"adaptive"` | `"adaptive"`, `"sequential"`, or `"parallel"` |
| `max_concurrency` | `int` | `4` | Max concurrent cases in parallel mode |
| `fail_fast` | `bool` | `False` | Stop after first failure (sequential mode) |
| `assertions` | `tuple[EvalAssertion \| AsyncEvalAssertion, ...]` | `()` | Assertions applied to every case result |
| `scorers` | `tuple[EvalScorer, ...]` | `()` | Scorers applied to every case result |
| `budget` | `EvalBudget \| None` | `None` | Default budget for cases without a per-case budget |

Adaptive mode resolves to `"sequential"` when case count <= 2 or max_concurrency <= 1, otherwise `"parallel"`.

---

## EvalBudget

Hard limits applied to one eval case's run metrics.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_duration_s` | `float \| None` | `None` | Maximum wall-clock duration in seconds |
| `max_total_tokens` | `int \| None` | `None` | Maximum total tokens (input + output) |
| `max_total_cost_usd` | `float \| None` | `None` | Maximum estimated cost in USD |

Budget violations are reported in `EvalCaseResult.budget_violations` and cause the case to fail.

---

## Built-in Assertions

AFK ships four assertion classes that implement the `EvalAssertion` protocol:

| Class | Constructor Fields | Description |
|-------|-------------------|-------------|
| `StateCompletedAssertion` | `name: str = "state_completed"` | Pass only when agent reaches `"completed"` state |
| `FinalTextContainsAssertion` | `needle: str`, `name: str` | Pass only when `final_text` contains the substring |
| `EventTypesExactAssertion` | `expected: tuple[str, ...]`, `name: str` | Pass only when event type sequence matches exactly |

```python
from afk.evals import (
    StateCompletedAssertion,
    FinalTextContainsAssertion,
    EventTypesExactAssertion,
)

assertions = (
    StateCompletedAssertion(),
    FinalTextContainsAssertion("New York"),
    EventTypesExactAssertion(expected=("run_started", "llm_response", "run_completed")),
)
```

### Custom Assertions

Any object that satisfies the `EvalAssertion` or `AsyncEvalAssertion` protocol works:

```python
from afk.evals import EvalAssertion, EvalAssertionResult, EvalCase, EvalCaseResult

class MaxLengthAssertion:
    name: str = "max_length"

    def __init__(self, max_chars: int):
        self.max_chars = max_chars

    def __call__(self, case: EvalCase, result: EvalCaseResult) -> EvalAssertionResult:
        ok = len(result.final_text) <= self.max_chars
        return EvalAssertionResult(
            name=self.name,
            passed=ok,
            details=None if ok else f"length={len(result.final_text)} > {self.max_chars}",
        )
```

---

## Scorers

Scorers produce numeric scores attached to case results. AFK provides `ResultLengthScorer` built-in.

| Class | Description |
|-------|-------------|
| `ResultLengthScorer` | Returns `float(len(result.final_text))` as the score |

Custom scorers satisfy the `EvalScorer` protocol -- a callable with a `name` attribute returning `float`:

```python
from afk.evals import EvalScorer, EvalCase, EvalCaseResult

class KeywordScorer:
    name: str = "keyword_coverage"

    def __init__(self, keywords: list[str]):
        self.keywords = keywords

    def __call__(self, case: EvalCase, result: EvalCaseResult) -> float:
        text = result.final_text.lower()
        hits = sum(1 for kw in self.keywords if kw.lower() in text)
        return hits / max(len(self.keywords), 1)
```

---

## Dataset Loading

Load eval cases from a JSON file using `load_eval_cases_json`. Each row must have `name` and `agent` fields,
with `agent` resolved through a caller-provided `agent_resolver` callback.

```python
from afk.evals import load_eval_cases_json

cases = load_eval_cases_json(
    "tests/evals/cases.json",
    agent_resolver=lambda name: agent_map[name],
)
```

JSON dataset format:

```json
[
  {
    "name": "case-1",
    "agent": "summarizer",
    "user_message": "Summarize this.",
    "context": {"doc_id": "abc"},
    "tags": ["smoke"]
  }
]
```

---

## Running Evals

### Async (preferred)

```python
from afk.core.runner import Runner
from afk.evals import EvalSuiteConfig, arun_suite

result = await arun_suite(
    runner_factory=Runner,
    cases=cases,
    config=EvalSuiteConfig(
        execution_mode="parallel",
        max_concurrency=4,
        assertions=(StateCompletedAssertion(),),
    ),
)

print(f"Passed: {result.passed}/{result.total}, Failed: {result.failed}")
for row in result.results:
    print(f"  {row.case}: {'PASS' if row.passed else 'FAIL'}")
```

### Sync wrapper

```python
from afk.evals import run_suite

result = run_suite(runner_factory=Runner, cases=cases, config=config)
```

### Single case execution

```python
from afk.evals import arun_case, run_case

# Async
case_result = await arun_case(Runner(), case, assertions=assertions, budget=budget)

# Sync
case_result = run_case(Runner(), case, assertions=assertions, budget=budget)
```

---

## EvalSuiteResult

Aggregated result returned by `arun_suite` / `run_suite`.

| Field / Property | Type | Description |
|-----------------|------|-------------|
| `results` | `list[EvalCaseResult]` | Per-case result objects |
| `execution_mode` | `ExecutionMode` | Resolved execution mode used |
| `total` | `int` (property) | Total executed case count |
| `passed` | `int` (property) | Count of cases that passed all checks |
| `failed` | `int` (property) | Count of cases that failed |

## EvalCaseResult

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `case` | `str` | -- | Case name |
| `state` | `AgentState` | -- | Terminal agent state |
| `final_text` | `str` | -- | Final assistant text output |
| `run_id` | `str` | -- | Unique run identifier |
| `thread_id` | `str` | -- | Thread identifier |
| `event_types` | `list[str]` | `[]` | Ordered event type sequence |
| `metrics` | `RunMetrics` | `RunMetrics()` | Projected run metrics |
| `assertions` | `list[EvalAssertionResult]` | `[]` | Assertion/scorer output rows |
| `budget_violations` | `list[str]` | `[]` | Budget violation messages |
| `passed` | `bool` | `True` | Whether the case passed all checks |

---

## Golden Trace Testing

Compare observed event sequences against a saved golden trace file:

```python
from afk.evals import compare_event_types, write_golden_trace

# Save a golden trace after a known-good run
write_golden_trace("tests/evals/golden/basic.json", events)

# Compare in tests
ok, message = compare_event_types(expected=golden_types, observed=result.event_types)
assert ok, message
```

---

## Reporting

Serialize suite results to a JSON report:

```python
from afk.evals import write_suite_report_json, suite_report_payload

# Write to disk
write_suite_report_json("reports/eval-suite.json", suite_result)

# Or get the payload dict directly
payload = suite_report_payload(suite_result)
# payload["schema_version"] == "eval_suite.v1"
# payload["summary"]["total"], payload["summary"]["passed"], etc.
```

---

## Testing Patterns

### Unit Testing Tools

```python
import pytest
from afk.tools import tool, ToolContext, ToolResult

@tool(name="greet", description="Greet a user")
def greet(name: str) -> str:
    return f"Hello, {name}!"

@pytest.mark.asyncio
async def test_greet_tool():
    result: ToolResult = await greet.call({"name": "World"}, ctx=ToolContext())
    assert result.output == "Hello, World!"
```

### Integration Testing Agents

```python
import pytest
from afk.agents import Agent
from afk.core.runner import Runner

@pytest.mark.asyncio
async def test_agent_greeting():
    agent = Agent(model="gpt-4.1-mini", instructions="Greet users warmly.")
    runner = Runner()
    result = await runner.run(agent, user_message="Hi!")
    assert result.state == "completed"
    assert "hello" in result.final_text.lower()
```

### Eval Suite in pytest

```python
import pytest
from afk.agents import Agent
from afk.core.runner import Runner
from afk.evals import (
    EvalCase,
    EvalSuiteConfig,
    FinalTextContainsAssertion,
    StateCompletedAssertion,
    arun_suite,
)

@pytest.mark.asyncio
async def test_eval_suite():
    agent = Agent(model="gpt-4.1-mini", instructions="Answer questions.")
    cases = [
        EvalCase(name="basic", agent=agent, user_message="Say ok"),
    ]
    result = await arun_suite(
        runner_factory=Runner,
        cases=cases,
        config=EvalSuiteConfig(
            assertions=(StateCompletedAssertion(),),
        ),
    )
    assert result.failed == 0
```

---

## CORRECT / WRONG Examples

### CORRECT -- Use pluggable assertions for deterministic checks

```python
config = EvalSuiteConfig(
    assertions=(
        StateCompletedAssertion(),
        FinalTextContainsAssertion("expected phrase"),
    ),
)
```

### WRONG -- Only check final_text manually after the run

```python
result = await arun_suite(runner_factory=Runner, cases=cases)
# Manually scanning results misses structured assertion reporting
for r in result.results:
    assert "expected phrase" in r.final_text  # No assertion metadata captured
```

---

### CORRECT -- Set budget limits on eval cases

```python
from afk.evals import EvalBudget

config = EvalSuiteConfig(
    budget=EvalBudget(max_duration_s=60.0, max_total_cost_usd=0.50),
)
```

### WRONG -- Let evals run without cost limits

```python
# No budget means unbounded cost and duration -- risky in CI
config = EvalSuiteConfig()
```

---

### CORRECT -- Use tags to filter test subsets

```python
cases = [
    EvalCase(name="fast-1", agent=agent, user_message="x", tags=("smoke",)),
    EvalCase(name="slow-1", agent=agent, user_message="y", tags=("full",)),
]
# Filter at the call site
smoke_cases = [c for c in cases if "smoke" in c.tags]
result = await arun_suite(runner_factory=Runner, cases=smoke_cases)
```

### WRONG -- Comment out or delete individual cases

```python
cases = [
    EvalCase(name="fast-1", agent=agent, user_message="x"),
    # EvalCase(name="slow-1", agent=agent, user_message="y"),  # WRONG: disabled by commenting
]
```

---

### CORRECT -- Use public imports from `afk.evals`

```python
from afk.evals import EvalCase, EvalSuiteConfig, arun_suite, run_suite
from afk.evals import StateCompletedAssertion, FinalTextContainsAssertion
from afk.evals import EvalBudget, evaluate_budget
```

### WRONG -- Import from internal submodules

```python
from afk.evals.models import EvalCase          # WRONG: internal path
from afk.evals.assertions import StateCompletedAssertion  # WRONG: internal path
from afk.evals.executor import arun_case       # WRONG: internal path
```

---

## Cross-References

- **Agents & Runner**: See [agents-and-runner.md](./agents-and-runner.md) for `Agent`, `Runner`, `AgentResult`
- **Tools**: See [tools-system.md](./tools-system.md) for `@tool`, `ToolRegistry`, and tool middleware
- **Cookbook**: See [cookbook-examples.md](./cookbook-examples.md) for end-to-end examples

## Source and Docs

- Source: `src/afk/evals/models.py`, `src/afk/evals/contracts.py`, `src/afk/evals/datasets.py`, `src/afk/evals/assertions.py`
- Docs: https://afk.arpan.sh/library/evals, https://afk.arpan.sh/library/tested-behaviors
- Doc files: `docs/library/evals.mdx`, `docs/library/tested-behaviors.mdx`
