# AFK Python Library Documentation

This documentation is generated from the `afk` codebase and is intended to be a practical, implementation-accurate guide.

## Who This Is For

- Junior engineers building their first production agent runtime.
- Engineers integrating AFK into backend services.
- Engineers extending AFK with custom tools, policies, and memory backends.

## Documentation Map

- [Developer Onboarding Guide](./developer-guide.md)
- [Architecture](./architecture.md)
- [Tools System Walkthrough](./tools-system-walkthrough.md)
- [Full API Reference](./api-reference.md)
- [Public Imports and Function Improvement](./public-imports-and-function-improvement.md)
- [Full Module Reference (All Source Files)](./full-module-reference.md)
- [Tested Behaviors and Edge Cases](./tested-behaviors.md)
- [Examples Runbook](./examples/README.md)
- Deep dives:
  - [Tool Call Lifecycle](./tool-call-lifecycle.md)
  - [Tools System Walkthrough](./tools-system-walkthrough.md)
  - [LLM Interaction Flow](./llm-interaction.md)
  - [Agentic Behavior and Orchestration](./agentic-behavior.md)
  - [Checkpoint and Resume Schema](./checkpoint-schema.md)
  - [Run Event Contract](./run-event-contract.md)
  - [Failure Policy Matrix](./failure-policy-matrix.md)
  - [Security Model](./security-model.md)
- Runnable Python Examples (all code lives in `.py` files):
  - [01_minimal_chat_agent.py](./examples/01_minimal_chat_agent.py)
  - [02_policy_with_hitl.py](./examples/02_policy_with_hitl.py)
  - [03_subagents_with_router.py](./examples/03_subagents_with_router.py)
  - [04_resume_and_compact.py](./examples/04_resume_and_compact.py)
  - [05_direct_llm_structured_output.py](./examples/05_direct_llm_structured_output.py)
  - [06_tool_registry_security.py](./examples/06_tool_registry_security.py)

## Package Overview

AFK has six top-level subsystems:

- `afk.agents`: declarative agent definitions, policy model, run/result types, fail-safe configuration.
- `afk.core`: runtime executor (`Runner`), interaction providers (HITL), telemetry sinks.
- `afk.llms`: provider-agnostic LLM contract, adapter factory, request/response types, streaming/session controls.
- `afk.tools`: tool abstraction, hook/middleware system, registry, sandbox/security policies, built-in runtime/skill tools.
- `afk.memory`: event/state storage, long-term memory retrieval, compaction/retention.
- `afk.evals`: deterministic scenario harness and golden-trace helpers.

## Public Import Rule (Important)

In docs and application code, use package imports only:

- `from afk.agents import Agent`
- `from afk.core import Runner, RunnerConfig`
- `from afk.llms import create_llm`
- `from afk.tools import tool`
- `from afk import agents, core, llms, tools, memory, evals`

Avoid importing from `src/...` paths in user code.

## Quick Setup (10 Minutes)

1. Install dependencies:
   - `uv sync`
2. Choose adapter + model:
   - `export AFK_LLM_ADAPTER=openai`
   - `export AFK_LLM_MODEL=gpt-4.1-mini`
   - `export AFK_LLM_API_KEY=...`
3. Run the first example:
   - `uv run python docs/library/examples/01_minimal_chat_agent.py`

If you prefer `pip`:

- `pip install -e .`
- then run examples with `python docs/library/examples/01_minimal_chat_agent.py`

## Documentation Coverage

This docs set now covers:

- All Python modules under `afk` (module-by-module symbol and method reference).
- Public APIs and runtime internals (execution loop, policy, checkpoint/resume, replay).
- Security model (tool-output trust boundaries, sandbox policies, secret scoping hooks).
- End-to-end deep dives for tool flow, LLM flow, and agentic orchestration.
- Behavior specifications derived from tests under `tests/`.
- Runnable code examples in `.py` files for onboarding and integration.

## Start Here: Build Your First Agent

1. Define an `Agent` with:
   - `model`
   - `instructions`
   - optional `tools`
2. Call `await agent.call(...)` for a one-shot result.
3. If you need lifecycle control/events, use `Runner.run_handle(...)`.
4. If you need policies/HITL, configure:
   - `PolicyEngine` and/or `policy_roles`
   - `Runner(interaction_provider=..., config=RunnerConfig(interaction_mode=...))`
5. If you need persistence/resume:
   - use a non-ephemeral memory backend via environment
   - call `Runner.resume(...)` or `Runner.resume_handle(...)`

Use [01_minimal_chat_agent.py](./examples/01_minimal_chat_agent.py) as the baseline.

## Read By Goal

- "I want a first working agent":
  - [Developer Onboarding Guide](./developer-guide.md)
  - [01_minimal_chat_agent.py](./examples/01_minimal_chat_agent.py)
- "I need approval/HITL and safety gates":
  - [02_policy_with_hitl.py](./examples/02_policy_with_hitl.py)
  - [Tool Call Lifecycle](./tool-call-lifecycle.md)
  - [Tools System Walkthrough](./tools-system-walkthrough.md)
  - [Security Model](./security-model.md)
- "I need subagents and routing":
  - [03_subagents_with_router.py](./examples/03_subagents_with_router.py)
  - [Agentic Behavior and Orchestration](./agentic-behavior.md)
- "I need persistence, resume, and compaction":
  - [04_resume_and_compact.py](./examples/04_resume_and_compact.py)
  - [Checkpoint and Resume Schema](./checkpoint-schema.md)
- "I need low-level LLM control":
  - [05_direct_llm_structured_output.py](./examples/05_direct_llm_structured_output.py)
  - [LLM Interaction Flow](./llm-interaction.md)

## Runtime Lifecycle (High-Level)

For each run, AFK does this:

1. Resolve model into a concrete LLM adapter.
2. Resolve skills and auto-register runtime/skill tools.
3. Build system prompt chunks (trusted header, instructions, skill manifest).
4. Enter step loop with fail-safe budget checks.
5. Optionally run routed subagents.
6. Policy-gate LLM call, then execute LLM.
7. If no tool calls, complete run.
8. If tool calls exist:
   - policy/HITL/sandbox gate each tool call
   - replay tool effect from journal if possible
   - execute remaining tool calls
   - append tool outputs back to transcript
9. Persist checkpoints/snapshots and continue.
10. Emit terminal event and persist terminal result.

See [Architecture](./architecture.md) for sequence/state diagrams.

## Junior Engineer Playbook

When implementing a new production agent, follow this order:

1. Start with [01_minimal_chat_agent.py](./examples/01_minimal_chat_agent.py).
2. Add strongly-typed tools (Pydantic args model + `@tool`).
3. Add policy rules before running tools with side effects.
4. Add sandbox profiles and output limits.
5. Add a concrete memory backend and test resume.
6. Add subagents only when single-agent prompts become hard to maintain.
7. Add eval scenarios with expected event traces.

## Production Defaults You Should Know

- `RunnerConfig.interaction_mode` defaults to `"headless"`.
- Headless interaction fallback decisions default to `deny`.
- Tool output is sanitized and marked untrusted by default.
- Runtime auto-registers filesystem runtime tools (`list_directory`, `read_file`).
- If no memory backend can be created from env, runner falls back to in-memory storage and emits a warning event.

## Common Edge Cases

- **Multiple event consumers**:
  - `AgentRunHandle.events` and `LLMStreamHandle.events` are single-consumer streams.
- **Resume with missing checkpoints**:
  - `Runner.resume_handle(...)` raises `AgentCheckpointCorruptionError` if latest checkpoint is missing/invalid.
- **Tool replay mismatch**:
  - input hash mismatch in effect journal raises checkpoint corruption error (safety against stale/reordered state).
- **Deferred approvals/inputs time out**:
  - fallback behavior comes from `RunnerConfig.approval_fallback` / `RunnerConfig.input_fallback`.
- **LLM retries**:
  - chat retries require idempotency support + `idempotency_key`.
- **Structured outputs**:
  - invalid structured payloads trigger repair attempts up to `LLMConfig.json_max_retries`.

## Where To Go Next

- Developer onboarding: [developer-guide.md](./developer-guide.md)
- API details: [api-reference.md](./api-reference.md)
- Import and extension patterns: [public-imports-and-function-improvement.md](./public-imports-and-function-improvement.md)
- Flow/architecture: [architecture.md](./architecture.md)
- Tool flow deep dive: [tool-call-lifecycle.md](./tool-call-lifecycle.md)
- Tools code walkthrough: [tools-system-walkthrough.md](./tools-system-walkthrough.md)
- LLM flow deep dive: [llm-interaction.md](./llm-interaction.md)
- Agentic orchestration deep dive: [agentic-behavior.md](./agentic-behavior.md)
- Checkpoint contract: [checkpoint-schema.md](./checkpoint-schema.md)
- Event contract: [run-event-contract.md](./run-event-contract.md)
- Failure policy matrix: [failure-policy-matrix.md](./failure-policy-matrix.md)
- Security model: [security-model.md](./security-model.md)
- Source map: [full-module-reference.md](./full-module-reference.md)
- Behavior specs: [tested-behaviors.md](./tested-behaviors.md)
- Example runbook: [examples/README.md](./examples/README.md)
- Existing LLM adapter docs in this repo:
  - [docs/llms/contracts.md](../llms/contracts.md)
  - [docs/llms/adapters.md](../llms/adapters.md)
  - [docs/llms/control-and-session.md](../llms/control-and-session.md)
  - [docs/llms/agent-integration.md](../llms/agent-integration.md)
