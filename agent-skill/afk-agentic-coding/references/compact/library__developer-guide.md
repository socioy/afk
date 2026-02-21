# Developer Guide

Fastest path to build your first AFK agent and harden it for production.

Source: `docs/library/developer-guide.mdx`

This guide is designed for teams shipping AFK agents in real environments, not toy demos.

## TL;DR

- Follow this page to go from zero to production-safe AFK baseline quickly.
- Implement capabilities in maturity order to reduce risk.
- Use public `afk.*` imports only for long-term API stability.

## When to Use

- You are onboarding to AFK for the first time.
- You need a practical path from prototype to production.
- You want copy-ready patterns for tools, policy, prompts, and runtime controls.

## First 20 Minutes

```bash uv
uv pip install afk pydantic
```

```bash pip
pip install afk pydantic
```

```bash

```

```bash

mkdir -p .agents/prompt
```

```bash
python 01_minimal_chat_agent.py
```

  Move to `02_policy_with_hitl.py` before enabling write/delete/action tools.

Create local script files using the embedded examples in Examples, then run them with the filenames shown in each accordion.

## AI Coding Assist (Optional)

If your team uses coding agents, install AFK skills from GitHub:

```bash
npx skills add socioy/afk
```

Then use [Building with AI](/library/building-with-ai) for prompt patterns and recommended skill usage.

## Recommended Build Order

    Start with one agent, one prompt strategy, and one request path.

    Add bounded runtime tools and typed arguments.

    Add approval gates and safety policy before side effects.

    Persist state and verify restart behavior.

    Route work to specialist subagents with explicit boundaries.

    Enforce sandbox, limits, and production safety controls.

Use the full level requirements page: [Agentic System Levels](/library/agentic-levels)

## Required Features by Level

| Level | Required Features |
| --- | --- |
| 1 | `Agent`, prompt strategy (`instructions` or `instruction_file`) |
| 2 | typed tools, tool descriptions, bounded arguments |
| 3 | policy + HITL + sandbox/output controls |
| 4 | checkpoint + resume + compaction strategy |
| 5 | subagents + router + explicit context inheritance |
| 6 | fail-safe budgets, fallback, telemetry, tests/evals |

## Public Import Patterns

```python
from afk.agents import Agent
from afk.core import Runner, RunnerConfig
from afk.llms import create_llm
from afk.tools import tool
from afk.memory import create_memory_store_from_env
```

Do not import from `src/...` in app code. Use public imports (`afk.*`) only.

## Core Workflow

1. Define a minimal `Agent` and validate end-to-end response flow.
2. Add typed tools with strict argument schemas.
3. Add policy decisions for risky tool calls.
4. Add sandbox and output limits for tool safety.
5. Add memory persistence and resume verification.
6. Add tests/evals for regressions (`afk.evals.EvalCase`, `run_suite`, assertions, budgets).

## Common Implementation Patterns

```python
from pydantic import BaseModel, Field
from afk.agents import Agent
from afk.tools import tool

class LookupArgs(BaseModel):
    query: str = Field(min_length=1)

@tool(
    args_model=LookupArgs,
    name="lookup",
    description="Look up canonical product/internal documentation content by query.",
)
def lookup(args: LookupArgs) -> dict[str, str]:
    return {"query": args.query}

agent = Agent(
    model="gpt-4.1-mini",
    instructions="Use lookup when external data is needed.",
    tools=[lookup],
)
```

```python
from afk.core import Runner, RunnerConfig

runner = Runner(
    config=RunnerConfig(
        interaction_mode="headless",
        sanitize_tool_output=True,
    )
)
```

```python
from afk.agents import Agent

agent = Agent(
    name="ChatAgent",
    model="gpt-4.1-mini",
    prompts_dir=".agents/prompt",  # or AFK_AGENT_PROMPTS_DIR
)
```

Expected file: `.agents/prompt/CHAT_AGENT.md`

## Troubleshooting

      Verify AFK_LLM_ADAPTER.
      Check available adapters with afk.llms.available_llm_adapters().

    Use explicit prefixes when needed:

      openai/gpt-4.1-mini
      anthropic/claude-sonnet-4

      Check RunnerConfig.interaction_mode.
      For non-headless mode, ensure interaction_provider is configured.
      Set timeout/fallback behavior in RunnerConfig.

      Inspect policy decision (PolicyEngine rules).
      Verify sandbox profile allowlists.

      Confirm prompt root precedence: prompts_dir, then AFK_AGENT_PROMPTS_DIR, then .agents/prompt.
      For auto mode, verify filename format (UPPER_SNAKE.md).
      For explicit mode, verify instruction_file path remains inside the prompt root.

## Continue Reading

- [Agentic System Levels](/library/agentic-levels)
- [Building with AI](/library/building-with-ai)
- [Architecture](/library/architecture)
- [Tool Call Lifecycle](/library/tool-call-lifecycle)
- [System Prompts](/library/system-prompts)
- [Security Model](/library/security-model)
- [API Reference](/library/api-reference)
