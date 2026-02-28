# Repository Design and Quality Standards

These standards apply to every file in the AFK repository -- code, docs,
examples, tests, and configuration. They define what "library-grade quality"
means in concrete, actionable terms.

---

## 1. Developer Experience Standards

### API Design Rules

- **Progressive disclosure**: Simple tasks require simple code. Advanced features are opt-in.
- **Sensible defaults**: Every parameter has a working default. Zero-config produces reasonable behavior.
- **Consistency**: Similar operations follow similar patterns across the codebase.
- **Discoverability**: IDE autocompletion and type hints guide users to the right API.
- **Actionable errors**: Every exception message explains what went wrong, why, and how to fix it.

### Import Discipline

```python
# Users ONLY import from public surfaces:
from afk.agents import Agent, ChatAgent
from afk.core import Runner
from afk.tools import tool, Tool, ToolRegistry
from afk.llms import LLM, LLMBuilder
from afk.memory import MemoryStore

# Users NEVER import from internal modules:
from afk.core.runner.internals import ...       # WRONG
from afk.llms.clients.shared.normalization import ...  # WRONG
```

Every `__init__.py` is a curated public API surface. Exports are explicit and
intentional.

---

## 2. Documentation Standards

### Page Quality

- **Concise and actionable**: Get to the point. Show code early.
- **Technically precise**: Every statement is verifiable against the source code.
- **Stable ordering**: Overview -> Usage -> API -> Failure Modes -> Advanced -> Examples -> Next Steps.
- **No duplication**: Link to canonical definitions instead of re-explaining.
- **Aligned to code**: If behavior changes, docs update in the same PR.

### Page Structure Template

Every concept page follows this structure:

```mdx
---
title: Feature Name
description: One-sentence description
---

# Feature Name

{/* 1-2 sentence overview */}

## Quick Start
{/* Minimal working example */}

## When to Use
{/* Problem this solves, when to reach for it vs alternatives */}

## How It Works
{/* Core concepts with diagrams where helpful */}

## API Reference
{/* Types, methods, parameters -- link to full reference for details */}

## Failure Modes
{/* What can go wrong, how errors surface, recovery patterns */}

## Advanced Patterns
{/* Composition, customization, integration with other subsystems */}

## Examples
{/* Additional runnable code samples */}

## Next Steps
{/* Links to related pages */}
```

### Required Coverage

Every major subsystem must have documentation covering:

| Topic | Required |
|-------|----------|
| Quick start example | Yes |
| Configuration options | Yes |
| Error/failure modes | Yes |
| Backend/adapter options | Yes (if pluggable) |
| Integration with other AFK systems | Yes |
| Security considerations | Yes (if applicable) |
| Performance characteristics | Recommended |

---

## 3. Example Quality Standards

### Rules

- **Runnable**: Examples must work or be obviously near-runnable (only omitting API keys).
- **Public imports only**: Always use `from afk...` paths.
- **Minimal scope**: One concept per example. Do not demonstrate memory + streaming + policies in one example.
- **Expected output**: Include comments showing expected behavior when ambiguity is likely.
- **No eval() or exec()**: Never in examples, even with disclaimers.
- **Real patterns**: Examples should show patterns users actually need, not contrived demos.

```python
# GOOD: Minimal, focused, public imports
from afk.agents import Agent
from afk.core import Runner
from afk.tools import tool
from pydantic import BaseModel

class SearchArgs(BaseModel):
    query: str

@tool(args_model=SearchArgs)
async def search(args: SearchArgs) -> str:
    """Search for information."""
    return f"Results for: {args.query}"

agent = Agent(name="researcher", model="gpt-4o", tools=[search])
result = Runner().run_sync(agent, input="Find info about Python 3.13")
print(result.final_text)
# Output: A summary of Python 3.13 features based on search results

# BAD: Too many concepts, internal imports, no expected output
from afk.core.runner.internals import _ensure_memory_store  # WRONG
```

---

## 4. Code Quality Standards

### Architecture Rules

- **Composition over inheritance**: Wire behavior through middleware, hooks, and policies.
- **One concern per module**: Each file has a single, clear responsibility.
- **Explicit interfaces**: Module boundaries defined by Protocols, ABCs, or typed functions.
- **No hidden state**: Pass dependencies and config explicitly. No module-level mutable globals.
- **Provider isolation**: All provider-specific code lives in adapter directories.

### Code Style

- Python 3.13+ features are available.
- `ruff` is the linter and formatter. Line length: 100 characters.
- Type annotations are mandatory on all public functions and methods.
- Use `from __future__ import annotations` for forward references.
- Use `dataclasses` or Pydantic `BaseModel` for data objects.
- Use `slots=True` and `frozen=True` on dataclasses where applicable.
- Use `Enum` or `Literal` for fixed option sets, not magic strings.

### Naming Conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| Classes | PascalCase | `ToolRegistry`, `AgentResult` |
| Functions/methods | snake_case | `run_sync`, `resolve_model` |
| Constants | UPPER_SNAKE | `MAX_RETRIES`, `DEFAULT_TIMEOUT` |
| Private | Leading underscore | `_ensure_setup`, `_normalize_tool` |
| Type variables | Short PascalCase | `ArgsT`, `ReturnT` |
| Protocols | Descriptive PascalCase | `LLMTransport`, `InteractionProvider` |
| Test functions | `test_` + behavior description | `test_budget_exceeded_after_max_steps` |

---

## 5. Extensibility Standards

### Everyone Can Extend, Nobody Needs to Fork

Every major subsystem must provide extension points:

| Subsystem | Extension Mechanism | User Creates |
|-----------|-------------------|-------------|
| Tools | `@tool` decorator, `ToolRegistry` | New tool functions |
| Memory | `MemoryStore` ABC | New storage backends |
| LLM | `LLM` base class, provider registry | New LLM adapters |
| Telemetry | Exporter contracts | New metric exporters |
| Queues | Queue base class | New queue backends |
| Policies | `PolicyRule` data objects | New access control rules |
| Transport | `LLMTransport` Protocol | New HTTP/gRPC clients |

### Extensibility Checklist

Before approving a new subsystem or major component:

- [ ] Does it have a defined Protocol or ABC for the core contract?
- [ ] Can users create new implementations without modifying AFK source?
- [ ] Is the default implementation registered via a factory or registry pattern?
- [ ] Are all configuration options exposed via typed config objects?
- [ ] Is the extension documented with a "how to create your own" example?

---

## 6. Change Management Standards

### Minimal Safe Change

- Fix the root cause. Do not refactor surrounding code in the same PR.
- If a fix reveals a broader issue, file a follow-up issue and link it.
- One concern per PR. If your diff touches 15 files across 5 subsystems, it's probably too broad.

### Release Note Quality

- Every user-visible change has a CHANGELOG entry under the correct heading.
- Release notes describe **what changed for users**, not implementation details.
- Use "Keep a Changelog" format: `Added`, `Changed`, `Fixed`, `Removed`, `Deprecated`, `Security`.

```markdown
# GOOD release note
### Fixed
- `run_sync` no longer fails with `RuntimeError` when called from a CLI script on Python 3.13.

# BAD release note
### Fixed
- Fixed async loop bridge.
```

### Staged Rollout for High-Risk Changes

High-risk changes (runner lifecycle, error taxonomy, public API) should be
rolled out in stages:

1. **Phase 1**: Feature-flagged behind opt-in config or env var.
2. **Phase 2**: Enabled by default with documented opt-out.
3. **Phase 3**: Old behavior removed after one minor version.

---

## 7. Maintainer Review Checklist

Before approving any PR, confirm:

- [ ] Docs are clear, accurate, and updated for any behavior change.
- [ ] Examples are accurate, minimal, and use public imports.
- [ ] Code follows composition patterns (no unnecessary inheritance).
- [ ] Module boundaries are respected (no cross-cutting imports).
- [ ] Error handling is explicit and typed (no swallowed exceptions in critical paths).
- [ ] Quality gates pass (`ruff`, `pytest`, PR template checks).
- [ ] Migration notes exist if behavior or API changed.
- [ ] Resource lifecycle is correct (tasks stored, connections closed, locks released).
- [ ] Performance implications are considered (no unbounded growth, no blocking I/O in async).
