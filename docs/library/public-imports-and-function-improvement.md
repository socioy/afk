# Public Imports and Function Improvement

Use AFK as a package API, not as a source-path import.

## Why This Matters

- Public imports stay stable as internals evolve.
- New engineers can discover APIs from `afk.__init__` exports.
- It keeps docs, tests, and application code aligned.

## Recommended Import Patterns

```python
from afk.agents import Agent, PolicyDecision, PolicyEvent
from afk.core import Runner, RunnerConfig
from afk.llms import create_llm, LLMConfig
from afk.tools import tool
from afk.memory import create_memory_store_from_env
```

Or namespace-style:

```python
from afk import agents, core, llms, tools, memory, evals
```

Avoid importing through internal source-path modules. Use only `afk...` public package paths.

## How To Improve Existing Functions

### 1. Improve instruction functions (context-aware)

```python
from afk.agents import Agent

def instruction_builder(ctx: dict) -> str:
    locale = ctx.get("locale", "en-US")
    user_tier = ctx.get("tier", "free")
    return f"Respond in {locale}. Keep answers within plan limits for tier={user_tier}."

agent = Agent(
    model="openai/gpt-4o-mini",
    instructions=instruction_builder,
)
```

### 2. Improve policy functions (guard unsafe operations)

```python
from afk.agents import Agent, PolicyDecision, PolicyEvent

def policy_role(event: PolicyEvent) -> PolicyDecision:
    if event.event_type == "tool_before_execute" and event.tool_name == "run_shell":
        return PolicyDecision(action="request_approval", reason="Shell command requires approval")
    return PolicyDecision(action="allow")

agent = Agent(
    model="openai/gpt-4o-mini",
    policy_roles=[policy_role],
)
```

### 3. Improve tool functions (typed args + validation)

```python
from pydantic import BaseModel, Field
from afk.tools import tool

class SearchArgs(BaseModel):
    query: str = Field(min_length=2)
    limit: int = Field(default=5, ge=1, le=20)

@tool(args_model=SearchArgs, name="search_docs")
async def search_docs(args: SearchArgs) -> dict:
    return {"query": args.query, "hits": [], "limit": args.limit}
```

### 4. Improve LLM behavior (explicit config)

```python
from afk.llms import LLMConfig, create_llm

llm = create_llm(
    "openai/gpt-4o-mini",
    config=LLMConfig(
        timeout_s=30.0,
        max_retries=2,
        json_max_retries=2,
    ),
)
```

## Practical Rule

If a junior engineer can build features with only `from afk...` imports, docs are aligned with the intended developer experience.

## Runnable `.py` Examples

- [examples/01_minimal_chat_agent.py](./examples/01_minimal_chat_agent.py)
- [examples/02_policy_with_hitl.py](./examples/02_policy_with_hitl.py)
- [examples/05_direct_llm_structured_output.py](./examples/05_direct_llm_structured_output.py)
