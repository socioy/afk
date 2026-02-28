# AFK Coding Principles and Patterns

This is the definitive coding guide for the AFK framework. Every contributor and
agentic coding tool must internalize these principles before writing or reviewing
AFK code.

---

## 1. The Three Pillars -- Respect the Boundaries

AFK's architecture has three non-negotiable pillars. Code that blurs these
boundaries will be rejected.

### Agent (Configuration)
- An `Agent` is a **data object** -- it holds identity, instructions, tools, and policies.
- It never executes anything. It never holds runtime state. It never calls an LLM.
- It is passed **to** a Runner, which does the work.

```python
# CORRECT: Agent is pure configuration
agent = Agent(
    name="researcher",
    model="gpt-4o",
    instructions="You are a research assistant.",
    tools=[search_tool, summarize_tool],
)

# WRONG: Agent should not have execution logic
class MyAgent(Agent):
    async def run(self):  # NO -- execution belongs to Runner
        ...
```

### Runner (Execution)
- The `Runner` is the **stateful execution engine**. It runs the step loop.
- It manages the event loop lifecycle: LLM call -> process response -> execute tools -> loop.
- It owns checkpointing, budget enforcement, streaming, and memory interaction.
- All execution modes (sync, async, streaming, resume) go through Runner.

### Runtime (Infrastructure)
- LLM clients, tool registries, memory stores, telemetry sinks are runtime infrastructure.
- They are **injected** into the Runner, never hardcoded.
- They expose typed contracts (Protocols/ABCs) and can be swapped without changing agent or runner code.

---

## 2. Contract-First Design

Every module boundary in AFK is defined by typed contracts. This is how we
maintain a codebase that 50 contributors can work on without stepping on each other.

### Rules

- **Public APIs use Pydantic models or dataclasses** for structured data.
- **Cross-module boundaries use Protocols** (from `typing.Protocol`) for duck-typing.
- **Enforced contracts use ABCs** (from `abc.ABC`) when subclasses must implement methods.
- **Never pass raw dicts** across module boundaries. Typed objects are always preferred.
- **Type annotations are mandatory** on all public functions and methods.
- **Use `TYPE_CHECKING` imports** to avoid circular dependencies.

```python
# CORRECT: Typed contract at module boundary
from __future__ import annotations
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from afk.llms.types import LLMResponse

class LLMTransport(Protocol):
    async def send(self, request: LLMRequest) -> LLMResponse: ...

# WRONG: Dict-based interface
async def send(self, request: dict) -> dict: ...
```

### Contract Boundaries in AFK

| Boundary | Contract Types |
|----------|---------------|
| Agent <-> Runner | `Agent`, `AgentResult`, `AgentState` |
| Runner <-> LLM | `LLMRequest`, `LLMResponse`, `LLMStreamEvent` |
| Runner <-> Tools | `ToolCall`, `ToolResult`, `ToolSpec` |
| Runner <-> Memory | `MemoryStore` Protocol |
| Tool <-> Handler | `ArgsModel` (Pydantic), `ToolContext`, `ToolResult` |
| A2A boundaries | `AgentInvocationRequest`, `AgentInvocationResponse` |

---

## 3. DX-First API Design

AFK is a library. Our users are engineers building production agent systems.
Every API decision should prioritize their experience.

### Progressive Disclosure

Layer complexity. The simplest use case should be trivial. Advanced features
should be discoverable but not required.

```python
# Level 1: Trivial (5 lines to a working agent)
agent = Agent(name="helper", model="gpt-4o", instructions="Be helpful.")
result = Runner().run_sync(agent, input="Hello!")
print(result.final_text)

# Level 2: Add a tool (still straightforward)
@tool
def search(args: SearchArgs) -> str:
    return fetch_results(args.query)

agent = Agent(name="helper", model="gpt-4o", tools=[search])

# Level 3: Add safety (one config object)
agent = Agent(
    name="helper",
    model="gpt-4o",
    tools=[search],
    fail_safe=FailSafeConfig(max_steps=10, max_cost_usd=0.50),
)

# Level 4: Add policies, memory, streaming (when you need them)
# ...but you never HAVE to use them
```

### Sensible Defaults

Every parameter must have a default that works. Zero-config must produce
reasonable behavior.

```python
# CORRECT: Defaults create a functional agent
class FailSafeConfig:
    max_steps: int = 25           # Prevents runaway loops
    max_cost_usd: float = 1.0     # Prevents bill shock
    max_tool_calls: int = 50      # Bounded tool usage
    tool_timeout_s: float = 30.0  # Tools can't hang forever

# WRONG: Requiring users to set everything
class FailSafeConfig:
    max_steps: int                 # No default -- forces user to guess
```

### Actionable Error Messages

Every error a user can encounter must tell them: what went wrong, why, and how
to fix it.

```python
# CORRECT: Actionable error
raise AgentBudgetExceededError(
    f"Agent '{agent.name}' exceeded max LLM calls ({max_calls}). "
    f"Increase FailSafeConfig.max_llm_calls or reduce agent complexity."
)

# WRONG: Useless error
raise RuntimeError("Budget exceeded")
```

---

## 4. Composition Over Inheritance

AFK uses composition patterns everywhere. Deep inheritance hierarchies make
code hard to understand, test, and extend.

### Middleware Pattern

AFK uses middleware chains at three levels: tool, registry, and LLM.
Middleware is the primary mechanism for cross-cutting concerns.

```python
# Tool-level middleware
@middleware
async def log_tool_call(call_next, args, ctx):
    print(f"Calling tool with: {args}")
    result = await call_next(args)
    print(f"Tool returned: {result}")
    return result

# Registry-level middleware (wraps ALL tools)
@registry_middleware
async def audit_all_tools(call_next, tool_name, args, ctx, registry, spec):
    start = time.monotonic()
    result = await call_next(tool_name, args, ctx)
    elapsed = time.monotonic() - start
    log.info(f"Tool {tool_name} took {elapsed:.2f}s")
    return result
```

### Hook Pattern

Prehooks and posthooks provide targeted interception points on individual tools.

```python
@prehook
def validate_query(args: dict) -> dict:
    if len(args.get("query", "")) > 500:
        args["query"] = args["query"][:500]
    return args  # Prehooks MUST return the (possibly modified) args dict

@posthook
def redact_output(result):
    # Posthooks transform the output after tool execution
    return sanitize(result)
```

### Policy Pattern

The `PolicyEngine` provides declarative, deterministic access control without
subclassing or conditional logic scattered through the codebase.

```python
engine = PolicyEngine(rules=[
    PolicyRule(match="tool:*", action="allow"),
    PolicyRule(match="tool:delete_*", action="request_approval"),
    PolicyRule(match="tool:admin_*", action="deny"),
])
```

### When NOT to Use Inheritance

- Never subclass `Agent` to add execution behavior. Use tools, policies, and runner config.
- Never subclass `Runner` to change step-loop behavior. Use hooks and middleware.
- Never subclass `MemoryStore` to add cross-cutting concerns. Use lifecycle policies.

### When Inheritance IS Acceptable

- `MemoryStore` ABC: concrete adapters (SQLite, Redis, Postgres) implement the store contract.
- `LLM` base class: provider adapters implement the LLM transport contract.
- `BaseTool` -> `Tool`: adds hook/middleware pipeline on top of base execution.

---

## 5. Async-First with Sync Compatibility

AFK is async-native. All core execution paths are `async def`. Sync wrappers
exist for convenience but are never the primary API.

### Rules

- Core logic is always `async def`.
- Sync wrappers (`run_sync`, `chat_sync`) use `asyncio.run()` or thread-safe bridges.
- Never call `asyncio.run()` inside an already-running event loop.
- Use `asyncio.to_thread()` to offload blocking I/O from the event loop.
- Never use `time.sleep()` in async code. Always use `asyncio.sleep()`.
- `asyncio.Semaphore` and `asyncio.Lock` must be created within an async context or be lazy-initialized.

```python
# CORRECT: Async core with sync wrapper
async def run(self, agent, input):
    """Primary async API."""
    ...

def run_sync(self, agent, input):
    """Convenience sync wrapper."""
    return asyncio.run(self.run(agent, input))

# CORRECT: Blocking I/O offloaded
async def read_file_tool(args):
    content = await asyncio.to_thread(Path(args.path).read_text)
    return content

# WRONG: Blocking I/O on event loop
async def read_file_tool(args):
    content = Path(args.path).read_text()  # BLOCKS the event loop
    return content
```

### Event Loop Safety

- Always store references to `asyncio.create_task()` results to prevent GC.
- Use `asyncio.wait_for()` with timeouts on all external calls.
- Handle `asyncio.CancelledError` explicitly in cleanup paths.
- Never use bare `assert` for runtime invariants -- it's stripped with `-O`.

---

## 6. Error Handling Philosophy

Errors in AFK are classified, typed, and actionable. We never swallow
exceptions silently.

### Error Classification

| Category | Behavior | Examples |
|----------|----------|---------|
| **Retryable** | Retry with backoff | 429 rate limit, 503 unavailable, timeout |
| **Fatal** | Fail immediately | 401 auth failure, 400 bad request, invalid config |
| **Degraded** | Continue with reduced capability | Memory store unreachable, telemetry sink down |
| **Budget** | Hard stop | Cost limit, step limit, tool call limit exceeded |

### Rules

- **Typed error hierarchy**: Every error category has its own exception class.
- **Never catch bare `Exception`** in business logic. Catch specific types.
- **Use `raise ... from e`** to preserve exception chains.
- **Log before silencing**: If you must absorb an error (telemetry, non-critical paths), log it first.
- **Status code classification for LLM errors**: Map HTTP codes to retry/fatal categories explicitly.

```python
# CORRECT: Typed, chained, actionable
try:
    response = await transport.send(request)
except httpx.TimeoutException as e:
    raise LLMTimeoutError(
        f"LLM request timed out after {timeout}s for model '{model}'"
    ) from e

# WRONG: Swallowed exception with no trace
try:
    response = await transport.send(request)
except Exception:
    pass  # NO -- silent failure
```

### Never Swallow Errors in Critical Paths

These paths must ALWAYS propagate errors:
- Runner step loop (execution.py)
- Tool execution pipeline (base.py)
- LLM request/response (llm.py, client.py)
- Memory persistence (checkpoint writes)
- Circuit breaker state transitions

---

## 7. Module Design Rules

### One Concern Per Module

Each module has a single, clear responsibility. If you can't describe what a
module does in one sentence, it's too broad.

```
# CORRECT module scope
tools/core/base.py      -> Tool definition and execution pipeline
tools/core/decorator.py  -> Declarative tool creation (@tool, @prehook, etc.)
tools/registry.py        -> Tool storage, lookup, and registry-level middleware
tools/security.py        -> SandboxProfile enforcement and input validation

# WRONG: mixed concerns
tools/utils.py           -> Tool helpers AND security AND registry helpers
```

### Explicit Dependencies

- Pass dependencies via constructor injection or function parameters.
- Never import and use global singletons.
- Never rely on module-level mutable state.

```python
# CORRECT: Injected dependency
class Runner:
    def __init__(self, memory: MemoryStore | None = None):
        self._memory = memory

# WRONG: Global singleton
_global_memory = InMemoryStore()  # Hidden shared state

class Runner:
    def __init__(self):
        self._memory = _global_memory  # Invisible coupling
```

### Provider Isolation

Provider-specific code must be isolated in adapter modules:

```
# CORRECT: Provider logic in adapters
llms/clients/adapters/openai.py     -> OpenAI-specific request/response mapping
llms/clients/adapters/litellm.py    -> LiteLLM-specific transport

# WRONG: Provider logic in shared code
llms/llm.py                         -> if provider == "openai": ...  # NO
```

---

## 8. Testing Standards

### What to Test

| Category | What | Example |
|----------|------|---------|
| **Unit** | Pure logic, transforms, validators | Policy rule matching, error classification |
| **Contract** | Interface compliance | Memory store implements all Protocol methods |
| **Failure** | Error paths, edge cases | Tool timeout, LLM rate limit, memory unavailable |
| **Integration** | Component interaction | Runner + Agent + Tools end-to-end |
| **Eval** | Agent behavior quality | Assertions on AgentResult for eval cases |

### Testing Rules

- Every bug fix includes a regression test.
- Every new feature includes happy-path and failure-path tests.
- Tests use public imports (`from afk...`), never internal paths.
- Test names describe the behavior, not the implementation: `test_budget_exceeded_after_max_steps`, not `test_check_budget`.
- Use `pytest.mark.asyncio` for async tests.
- Mock external dependencies (LLM providers, network), never internal AFK contracts.
- Circuit breaker tests must cover: threshold boundaries, cooldown window, concurrent access, reset behavior.

---

## 9. Documentation Standards

### Every User-Visible Change Requires

1. **Updated docs** that accurately describe the new behavior.
2. **Updated examples** that use public imports and actually run.
3. **A CHANGELOG entry** under the correct heading (Added/Changed/Fixed/Removed).
4. **Migration notes** if behavior changed or API broke.

### Doc Page Structure

Every concept page follows this order:

```
1. Quick Start (minimal working example)
2. When to Use (problem it solves)
3. API Reference (types, methods, parameters)
4. Failure Modes (what can go wrong, how to handle it)
5. Advanced Patterns (composition, customization)
6. Examples (runnable code)
7. Next Steps (links to related pages)
```

### Example Quality

- Examples must be runnable or obviously near-runnable (only omit API keys).
- Use `from afk...` public imports, never internal module paths.
- Keep examples minimal -- one concept per example.
- Include expected output/behavior in comments when helpful.

---

## 10. Anti-Patterns to Reject

These patterns should be caught in review and never merged:

| Anti-Pattern | Why It's Bad | What to Do Instead |
|-------------|-------------|-------------------|
| Silent exception swallowing | Bugs become invisible | Log + re-raise or log + degrade gracefully |
| Hidden mutable global state | Untestable, race-prone | Inject dependencies via constructor |
| Provider logic in shared code | Breaks portability | Isolate in adapter modules |
| Deep inheritance for behavior | Hard to trace, extend, test | Use middleware, hooks, policies |
| Raw dicts at module boundaries | No validation, no IDE support | Use Pydantic models or dataclasses |
| `assert` for runtime checks | Stripped with `-O` flag | Use `if not x: raise ValueError(...)` |
| Blocking I/O in async functions | Freezes the event loop | Use `asyncio.to_thread()` |
| Fire-and-forget `create_task()` | GC can kill the task | Store the task reference |
| Bare `except Exception` in handlers | Masks real bugs | Catch specific exception types |
| `asyncio.sleep(0)` as retry backoff | No actual delay between retries | Use exponential backoff |
| Substring matching for error codes | False positives ("500" matches "25003") | Use structured error classification |
