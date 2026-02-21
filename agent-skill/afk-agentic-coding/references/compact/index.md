# AFK

Build production-ready agent systems with policy, tools, memory, and provider-agnostic LLM orchestration.

Source: `docs/index.mdx`

AFK (Agent Forge Kit) is an agent runtime focused on safe orchestration, deterministic behavior, and practical developer experience.

Language roadmap: examples currently use the Python SDK. TypeScript, Go, and Rust SDK docs are planned.

## TL;DR

- AFK provides a production-oriented runtime for tool-using, policy-governed agents.
- The documentation is organized from onboarding to internals to full API contracts.
- Use maturity levels to implement capabilities in the right order.

## When to Use AFK

- engineers building AI features with strict runtime controls
- teams shipping long-running or resumable agent workflows
- platform developers integrating multiple model providers behind one contract

    Go from zero to first working AFK agent with a copy-paste path.

    Choose a maturity level and implement only the required features.

    Learn by running progressively harder examples with diagrams.

    Install AFK skills via `npx skills` and build agents faster.

    Build a clear mental model of execution, tools, memory, and policy.

    Auto-load and template prompts with strict validation and caching.

    Use adapter contracts, streaming/session controls, and integration boundaries.

    Full public APIs, contracts, and edge-case behavior in one place.

## Quick Start

```bash uv
uv pip install afk pydantic
```

```bash pip
pip install afk pydantic
```

```bash

```

```bash
python -c "import afk; print('AFK SDK ready')"
```

## What You Can Build with AFK

- Chat and task-oriented agents with typed tools
- Policy-gated tool execution with human approval flows
- Subagent orchestration and routing
- Resume-safe long-running workflows with checkpoints
- Provider-agnostic LLM integrations

## Build by Level

1. Level 1: Prompted Agent
2. Level 2: Tool Agent
3. Level 3: Governed Agent
4. Level 4: Durable Agent
5. Level 5: Multi-Agent System
6. Level 6: Production Agentic Platform

Use: [Agentic System Levels](/library/agentic-levels)

| Level | Required Features |
| --- | --- |
| 1 | prompt strategy (`instructions` or prompt file), model selection |
| 2 | typed tools + argument validation + tool descriptions |
| 3 | policy/HITL + sandbox/output controls |
| 4 | checkpointing + resume behavior |
| 5 | subagent routing + context boundaries |
| 6 | budgets/fallback/telemetry + regression tests/evals |

## Recommended Reading Order

1. [Library Overview](/library/overview)
2. [Developer Guide](/library/developer-guide)
3. [Building with AI](/library/building-with-ai)
4. [Agentic System Levels](/library/agentic-levels)
5. [Examples](/library/examples/index)
6. [Architecture](/library/architecture)
7. [API Reference](/library/api-reference)
