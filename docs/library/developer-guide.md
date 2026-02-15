# AFK Developer Onboarding Guide

This guide is for engineers who want to build production agents quickly using `afk...` public imports only.

## Prerequisites

- Python `>=3.13`
- One configured LLM adapter
- Repo dependencies installed

Install dependencies:

- `uv sync`

or:

- `pip install -e .`

## Environment Setup

Set these before running examples:

- `AFK_LLM_ADAPTER` (for example: `openai`, `litellm`, `anthropic_agent`)
- `AFK_LLM_MODEL` (for example: `gpt-4.1-mini`)
- `AFK_LLM_API_KEY` (or adapter-specific key path used by your runtime)

Example:

```bash
export AFK_LLM_ADAPTER=openai
export AFK_LLM_MODEL=gpt-4.1-mini
export AFK_LLM_API_KEY=your_key_here
```

## First Successful Run

Run this first:

- `uv run python docs/library/examples/01_minimal_chat_agent.py`

If this fails, start with [Troubleshooting](#troubleshooting).

## Build Order (Recommended)

1. Minimal agent:
   - [examples/01_minimal_chat_agent.py](./examples/01_minimal_chat_agent.py)
2. Add policy + approval:
   - [examples/02_policy_with_hitl.py](./examples/02_policy_with_hitl.py)
3. Add subagents:
   - [examples/03_subagents_with_router.py](./examples/03_subagents_with_router.py)
4. Add resume/compaction:
   - [examples/04_resume_and_compact.py](./examples/04_resume_and_compact.py)
5. Add low-level LLM control:
   - [examples/05_direct_llm_structured_output.py](./examples/05_direct_llm_structured_output.py)
6. Add tool sandboxing:
   - [examples/06_tool_registry_security.py](./examples/06_tool_registry_security.py)

## Public Import Patterns

Use these imports in application code:

```python
from afk.agents import Agent
from afk.core import Runner, RunnerConfig
from afk.llms import create_llm
from afk.tools import tool
from afk.memory import create_memory_store_from_env
```

Avoid importing through `src/...` paths.

## Common Implementation Patterns

### Pattern: Agent + Typed Tool

```python
from pydantic import BaseModel, Field
from afk.agents import Agent
from afk.tools import tool

class LookupArgs(BaseModel):
    query: str = Field(min_length=1)

@tool(args_model=LookupArgs, name="lookup")
def lookup(args: LookupArgs) -> dict[str, str]:
    return {"query": args.query}

agent = Agent(
    model="gpt-4.1-mini",
    instructions="Use lookup when external data is needed.",
    tools=[lookup],
)
```

### Pattern: Runner With Explicit Controls

```python
from afk.core import Runner, RunnerConfig

runner = Runner(
    config=RunnerConfig(
        interaction_mode="headless",
        sanitize_tool_output=True,
    )
)
```

## Troubleshooting

### Symptom: "Unknown adapter" error

- Check `AFK_LLM_ADAPTER`
- See available adapters in code via `afk.llms.available_llm_adapters()`

### Symptom: Model runs with wrong provider

- Use explicit model prefix when needed:
  - `openai/gpt-4.1-mini`
  - `anthropic/claude-sonnet-4`

### Symptom: Run pauses forever waiting for approval/input

- Check `interaction_mode`
- For non-headless mode, ensure `interaction_provider` is configured
- Set timeout/fallback in `RunnerConfig`

### Symptom: Tool call denied

- Inspect policy decision and sandbox profile
- Check allowlists and path restrictions

## Next Reads

- Architecture: [architecture.md](./architecture.md)
- API reference: [api-reference.md](./api-reference.md)
- Tool execution details: [tool-call-lifecycle.md](./tool-call-lifecycle.md)
- Tools code walkthrough: [tools-system-walkthrough.md](./tools-system-walkthrough.md)
- Security hardening: [security-model.md](./security-model.md)
