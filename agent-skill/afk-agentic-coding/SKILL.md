---
name: afk-agentic-coding
description: Use this skill when building, refactoring, or debugging AFK-based agents and workflows with public afk.* APIs, policy-aware tools, checkpoints, and runnable tests.
---

# AFK Agentic Coding

Use this skill for implementation tasks that build or improve AFK agents.

Load local references first before reading full docs:

- `references/agentic-build-playbook.md`
- `references/feature-selection-matrix.md`
- `references/README.md` (generated; points to bundled docs index)

Use local custom tool when you need fast docs lookup:

- `scripts/search_afk_docs.py "your query"`

## Goals

- produce runnable AFK agent code with `from afk...` imports only
- keep behavior deterministic and safe with policy + sandbox controls
- preserve resume safety and traceability in long-running workflows

## Workflow

1. Pick the target maturity level from AFK docs:
- Level 1: prompted agent
- Level 2: tool agent
- Level 3: governed agent
- Level 4: durable agent
- Level 5: multi-agent orchestration
- Level 6: production hardening

2. Build with public imports only:
- `from afk.agents import Agent`
- `from afk.core import Runner, RunnerConfig`
- `from afk.tools import tool`
- `from afk.llms import create_llm`

3. Tool design rules:
- use typed Pydantic args models
- add strong tool descriptions
- set bounded arguments and timeouts

4. Safety rules:
- add policy gates before side-effect tools
- require approvals for risky tool calls
- enforce sandbox/output limits

5. Durability rules:
- verify checkpoint + resume behavior
- keep context boundaries explicit for subagents
- add failure policy defaults for llm/tool/subagent paths

6. Finish with validation:
- run changed tests
- summarize behavior changes and residual risk

## Reference Docs

- https://afk.arpan.sh/library/developer-guide
- https://afk.arpan.sh/library/agentic-levels
- https://afk.arpan.sh/library/api-reference
- https://afk.arpan.sh/library/tested-behaviors

## Indexed References

- `references/index.json`: generated manifest of indexed references for this skill.
- `references/compact/*.md`: compact markdown generated from `docs/**/*.mdx`.
- `scripts/search_afk_docs.py "your query"`: quick lookup across bundled docs index.
