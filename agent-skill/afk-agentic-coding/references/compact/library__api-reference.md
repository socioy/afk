# API Reference

Complete public API reference for afk.agents, afk.core, afk.llms, afk.tools, afk.memory, and afk.evals.

Source: `docs/library/api-reference.mdx`

This reference covers the public API exported by `afk/*/__init__.py`, plus key runtime behavior and edge cases required for correct integration.

## TL;DR

- This is the canonical public API contract for AFK packages.
- It covers constructors, runtime behavior, errors, and edge cases.
- Use this page for implementation correctness, not conceptual onboarding.

## When to Use

- You are implementing or reviewing production AFK integrations.
- You need exact field/argument behavior for APIs.
- You are validating compatibility before upgrades.

## How to Use This Document

- Use this file when you already know which package area you need (`afk.agents`, `afk.core`, `afk.tools`, and so on).
- For first-time onboarding, start with [Developer Guide](/library/developer-guide).
- For maturity-based planning, use [Agentic System Levels](/library/agentic-levels).
- For complete symbol-level internal map, use [Full Module Reference](/library/full-module-reference).

## Quick Navigation by Task

- Build first agent: `afk.agents.Agent`
- Control execution lifecycle: `afk.core.Runner`
- Add approval/HITL: `afk.core.InteractionProvider`
- Build tools: `afk.tools.tool`, `afk.tools.ToolRegistry`
- Enforce security: `afk.tools.SandboxProfile`, `RunnerConfig` sandbox fields
- Persist/resume: `afk.memory.*` + `Runner.resume(...)`
- Expose MCP server: `afk.mcp.MCPServer`, `afk.mcp.create_mcp_server`
- Use external MCP tools: `Agent(mcp_servers=[...])`
- Run eval cases/suites: `afk.evals.EvalCase`, `afk.evals.run_case`, `afk.evals.run_suite`
- Configure system prompts: `afk.agents.BaseAgent` (`instruction_file`, `prompts_dir`)

## Public Import Style

Use these import styles in app code:

```python
from afk.agents import Agent, ChatAgent
from afk.core import Runner, RunnerConfig
from afk.llms import create_llm, LLMConfig
from afk.mcp import MCPServer, create_mcp_server
from afk.tools import tool, ToolRegistry
from afk.memory import create_memory_store_from_env
```

You can also import top-level namespaces when useful:

```python
from afk import agents, core, llms, mcp, tools, memory, evals
```

Do not import via `src/...` paths in integration code.

Companion references:

- [Developer Guide](/library/developer-guide)
- [Examples](/library/examples/index)
- [Architecture](/library/architecture)
- [Public Imports and Function Improvement](/library/public-imports-and-function-improvement)
- [Full Module Reference (all source modules)](/library/full-module-reference)
- [Tested Behaviors and Edge Cases](/library/tested-behaviors)
- [Tool Call Lifecycle](/library/tool-call-lifecycle)
- [Agent Skills](/library/agent-skills)
- [System Prompts](/library/system-prompts)
- [Tools Walkthrough](/library/tools-system-walkthrough)
- [LLM Flow](/library/llm-interaction)
- [Agentic Orchestration](/library/agentic-behavior)
- [Checkpoint & Resume](/library/checkpoint-schema)
- [Run Event Contract](/library/run-event-contract)
- [Failure Policies](/library/failure-policy-matrix)
- [Security Model](/library/security-model)

## Top-Level Package

- `afk` exports: `agents`, `core`, `llms`, `mcp`, `memory`, `tools`, `evals`
- Source: [afk/**init**.py](https://github.com/socioy/afk/blob/main/src/afk/__init__.py)

---

## `afk.agents`

Source: [afk/agents/**init**.py](https://github.com/socioy/afk/blob/main/src/afk/agents/__init__.py)

### Core Agent Classes

- `BaseAgent`
- `Agent`
- `ChatAgent`
- `PromptStore`, `get_prompt_store`, `derive_auto_prompt_filename`

Primary source:

- [afk/agents/core/base.py](https://github.com/socioy/afk/blob/main/src/afk/agents/core/base.py)
- [afk/agents/core/chat.py](https://github.com/socioy/afk/blob/main/src/afk/agents/core/chat.py)
- [afk/agents/prompts/store.py](https://github.com/socioy/afk/blob/main/src/afk/agents/prompts/store.py)

#### `BaseAgent(...)` constructor behavior

Key fields:

- `model: str | LLM`
- `tools: list[ToolLike]`
- `subagents: list[BaseAgent]`
- `instructions: str | callable | None`
- `instruction_file: str | Path | None`
- `prompts_dir: str | Path | None`
- `policy_engine`, `policy_roles`, `instruction_roles`
- `subagent_router`
- `fail_safe: FailSafeConfig`
- `skill_tool_policy`, `skills`, `skills_dir`, `enable_skill_tools`
- `mcp_servers`, `enable_mcp_tools`

Validation/edge cases:

- Raises `AgentConfigurationError` when `max_steps  AgentResult`
- `run_handle(...) -> AgentRunHandle`
- `resume(agent, run_id, thread_id, context=None) -> AgentResult`
- `resume_handle(...) -> AgentRunHandle`
- `compact_thread(thread_id, event_policy=None, state_policy=None)`

Critical behavior:

- If `interaction_mode != "headless"` and no interaction provider is supplied, `AgentConfigurationError` is raised.
- `run(...)`/`resume(...)` raise `AgentCancelledError` if handle resolves to `None`.
- `resume_handle(...)` can return immediate terminal result when latest checkpoint already contains terminal payload.
- Missing/invalid checkpoints raise `AgentCheckpointCorruptionError`.

#### Execution loop details

- Budget enforced each step (`max_wall_time_s`, `max_llm_calls`, `max_tool_calls`, `max_steps`, optional `max_total_cost_usd`).
- State transitions validated by `validate_state_transition(...)`.
- Skill resolution occurs at run start; missing skills fail fast.
- Runtime tools (`list_directory`, `read_file`) are always added.
- Skill tools are added when enabled and skills are resolved.
- Tool effects can be replayed from effect journal for idempotency.

Failure policy normalization:

- LLM policy maps to `fail` or `degrade`.
- Tool/subagent/approval policies map to `continue`, `fail`, or `degrade`.

### Interaction Providers

Source: [afk/core/interaction.py](https://github.com/socioy/afk/blob/main/src/afk/core/interaction.py)

Exported:

- `InteractionProvider` (protocol)
- `HeadlessInteractionProvider`
- `InMemoryInteractiveProvider`

Behavior:

- Headless provider returns immediate fallback decisions.
- In-memory provider returns deferred tokens and supports `set_deferred_result(...)`.
- Deferred waits use timeout sleep semantics when unresolved.

### Telemetry

Source: [afk/core/telemetry.py](https://github.com/socioy/afk/blob/main/src/afk/core/telemetry.py)

Exported:

- `TelemetrySink` (protocol)
- `TelemetryEvent`
- `TelemetrySpan`

Behavior:

- Telemetry failures are intentionally non-fatal throughout runner code.
- Concrete telemetry backends (`null` / `inmemory` / `otel`) are exposed from `afk.observability.backends`.

---

## `afk.llms`

Source: [afk/llms/**init**.py](https://github.com/socioy/afk/blob/main/src/afk/llms/__init__.py)

### Core Interfaces

- `LLM`
- `LLMConfig`
- `MiddlewareStack`

Source:

- [afk/llms/llm.py](https://github.com/socioy/afk/blob/main/src/afk/llms/llm.py)
- [afk/llms/config.py](https://github.com/socioy/afk/blob/main/src/afk/llms/config.py)
- [afk/llms/middleware.py](https://github.com/socioy/afk/blob/main/src/afk/llms/middleware.py)

#### `LLM` public methods

- `chat(req, response_model=None)`
- `chat_sync(...)`
- `chat_stream(req, response_model=None)`
- `chat_stream_handle(...)`
- `embed(req)`
- `embed_sync(req)`
- `start_session(session_token=None, checkpoint_token=None)`

Key behavior and edge cases:

- Request IDs are auto-generated when missing.
- `chat` request validation is strict:
  - non-empty model/messages
  - numeric bounds (`top_p`, `temperature`, timeout, max tokens)
  - structured part validation for message content
- `chat_stream` and stream handles require exactly one completion event.
- Structured outputs:
  - validates provider payload when present
  - otherwise parses JSON from text and performs repair retries
- Retry behavior:
  - chat/stream retries require idempotency conditions
  - embedding retries use configured retry count directly
- `LLMStreamHandle.events` is single-consumer.

#### `LLMConfig.from_env()`

Reads:

- `AFK_LLM_MODEL`
- `AFK_EMBED_MODEL`
- `AFK_LLM_API_BASE_URL`
- `AFK_LLM_API_KEY`
- `AFK_LLM_TIMEOUT_S`
- `AFK_LLM_MAX_RETRIES`
- `AFK_LLM_BACKOFF_BASE_S`
- `AFK_LLM_BACKOFF_JITTER_S`
- `AFK_LLM_JSON_MAX_RETRIES`
- `AFK_LLM_MAX_INPUT_CHARS`

### Factory API

Source: [afk/llms/factory.py](https://github.com/socioy/afk/blob/main/src/afk/llms/factory.py)

- `create_llm(adapter, ...)`
- `create_llm_from_env(...)`
- `register_llm_adapter(name, factory, overwrite=False)`
- `available_llm_adapters()`

Built-in adapters:

- `openai`
- `litellm`
- `anthropic_agent`

### Client Adapters

Exported:

- `OpenAIClient`
- `LiteLLMClient`
- `AnthropicAgentClient`
- `ResponsesClientBase`

Source:

- [afk/llms/clients/adapters/openai.py](https://github.com/socioy/afk/blob/main/src/afk/llms/clients/adapters/openai.py)
- [afk/llms/clients/adapters/litellm.py](https://github.com/socioy/afk/blob/main/src/afk/llms/clients/adapters/litellm.py)
- [afk/llms/clients/adapters/anthropic_agent.py](https://github.com/socioy/afk/blob/main/src/afk/llms/clients/adapters/anthropic_agent.py)
- [afk/llms/clients/base/responses.py](https://github.com/socioy/afk/blob/main/src/afk/llms/clients/base/responses.py)

Adapter-specific notes:

- `OpenAIClient` uses Responses API and strict `json_schema` format.
- `LiteLLMClient` uses `aresponses`/`aembedding` and applies API base/key defaults.
- `AnthropicAgentClient`:
  - no embeddings support (`LLMCapabilityError`)
  - supports interrupt/session/checkpoint controls
  - maps transcript into a single prompt + system prompt

### LLM Types

Source: [afk/llms/types.py](https://github.com/socioy/afk/blob/main/src/afk/llms/types.py)

Exported dataclasses/types include:

- `Message`
- `ToolCall`
- `Usage`
- `LLMRequest`
- `LLMResponse`
- `EmbeddingRequest`
- `EmbeddingResponse`
- `LLMCapabilities`
- `ThinkingConfig`
- stream event types:
  - `StreamMessageStartEvent`
  - `StreamTextDeltaEvent`
  - `StreamToolCallDeltaEvent`
  - `StreamMessageStopEvent`
  - `StreamErrorEvent`
  - `StreamCompletedEvent`
- `LLMStreamEvent`
- `LLMStreamHandle`
- `LLMSessionHandle`
- `LLMSessionSnapshot`

### Observability + Errors

Source:

- [afk/llms/observability.py](https://github.com/socioy/afk/blob/main/src/afk/llms/observability.py)
- [afk/llms/errors.py](https://github.com/socioy/afk/blob/main/src/afk/llms/errors.py)

Exported observer types:

- `LLMLifecycleEvent`
- `LLMObserver`

Exported errors:

- `LLMError`
- `LLMTimeoutError`
- `LLMRetryableError`
- `LLMInvalidResponseError`
- `LLMConfigurationError`
- `LLMCapabilityError`
- `LLMCancelledError`
- `LLMInterruptedError`
- `LLMSessionError`
- `LLMSessionPausedError`

---

## `afk.mcp`

Source: [afk/mcp/**init**.py](https://github.com/socioy/afk/blob/main/src/afk/mcp/__init__.py)

Exported:

- server:
  - `MCPServer`
  - `MCPServerConfig`
  - `create_mcp_server(...)`
- external server store:
  - `MCPServerRef`
  - `MCPRemoteTool`
  - `MCPStore`
  - `get_mcp_store()`, `reset_mcp_store()`
- errors:
  - `MCPStoreError`
  - `MCPServerResolutionError`
  - `MCPRemoteProtocolError`
  - `MCPRemoteCallError`

Behavior highlights:

- `MCPServer.from_tools(...)` builds a registry-backed server directly from AFK tools.
- `create_mcp_server(...)` accepts either `registry=` or `tools=`.
- `MCPServerConfig` supports route toggles/paths (`mcp_path`, `sse_path`, `health_path`, `enable_sse`, `enable_health`, `allow_batch_requests`).
- `MCPStore.tools_from_servers(...)` materializes external MCP tools so they can run through standard AFK runner tool orchestration.

---

## `afk.memory`

Source: [afk/memory/**init**.py](https://github.com/socioy/afk/blob/main/src/afk/memory/__init__.py)

Exported:

- models:
  - `JsonValue`, `JsonObject`
  - `MemoryEvent`, `LongTermMemory`
  - `now_ms`, `new_id`
- store API:
  - `MemoryStore`, `MemoryCapabilities`
  - `InMemoryMemoryStore`, `SQLiteMemoryStore`
  - lazy: `RedisMemoryStore`, `PostgresMemoryStore`
- vector:
  - `cosine_similarity`
- factory:
  - `create_memory_store_from_env`
- lifecycle:
  - `RetentionPolicy`, `StateRetentionPolicy`, `MemoryCompactionResult`
  - `apply_event_retention`, `apply_state_retention`, `compact_thread_memory`

Sources:

- [afk/memory/models.py](https://github.com/socioy/afk/blob/main/src/afk/memory/models.py)
- [afk/memory/store/base.py](https://github.com/socioy/afk/blob/main/src/afk/memory/store/base.py)
- [afk/memory/store/in_memory.py](https://github.com/socioy/afk/blob/main/src/afk/memory/store/in_memory.py)
- [afk/memory/store/sqlite.py](https://github.com/socioy/afk/blob/main/src/afk/memory/store/sqlite.py)
- [afk/memory/store/redis.py](https://github.com/socioy/afk/blob/main/src/afk/memory/store/redis.py)
- [afk/memory/store/postgres.py](https://github.com/socioy/afk/blob/main/src/afk/memory/store/postgres.py)
- [afk/memory/factory.py](https://github.com/socioy/afk/blob/main/src/afk/memory/factory.py)
- [afk/memory/lifecycle.py](https://github.com/socioy/afk/blob/main/src/afk/memory/lifecycle.py)

### Memory backend selection (`create_memory_store_from_env`)

`AFK_MEMORY_BACKEND` values:

- `inmemory`
- `sqlite`
- `redis`
- `postgres`

Important edge cases:

- Postgres requires `AFK_VECTOR_DIM`.
- Unknown backend raises `ValueError`.

### Retention/Compaction behavior

- Event retention preserves configured event types plus most recent remainder up to budget.
- State retention keeps latest runs + selected phases + configurable prefixes.
- `compact_thread_memory(...)` is best effort:
  - if backend lacks `replace_thread_events` or `delete_state`, effective removals may be lower than logical removals.

---

## `afk.tools`

Source: [afk/tools/**init**.py](https://github.com/socioy/afk/blob/main/src/afk/tools/__init__.py)

### Core Tool Types

Source: [afk/tools/core/base.py](https://github.com/socioy/afk/blob/main/src/afk/tools/core/base.py)

Exported:

- `Tool`
- `ToolSpec`
- `ToolContext`
- `ToolResult`
- `ToolFn`
- `as_async`
- hook/middleware classes:
  - `PreHook`
  - `PostHook`
  - `Middleware`

Signature support:

- Tool function signatures:
  - `(args)`
  - `(args, ctx)`
  - `(ctx, args)`
- Middleware signatures:
  - `(call_next, args)`
  - `(call_next, args, ctx)`
  - `(args, ctx, call_next)`
  - `(ctx, args, call_next)`

### Decorators

Source: [afk/tools/core/decorator.py](https://github.com/socioy/afk/blob/main/src/afk/tools/core/decorator.py)

Exported:

- `tool(...)`
- `prehook(...)`
- `posthook(...)`
- `middleware(...)`
- `registry_middleware(...)`

### Registry

Source: [afk/tools/registery.py](https://github.com/socioy/afk/blob/main/src/afk/tools/registery.py)

Exported:

- `ToolRegistry`
- `ToolCallRecord`
- `RegistryMiddleware`
- `RegistryMiddlewareFn`

Important behavior:

- Global semaphore for concurrency limiting.
- Timeout precedence:
  - `call(timeout=...)` > tool default timeout > registry default timeout
- Supports plugin loading via entry points.
- Registry-level middleware wraps all tools and can be sync or async.

### Security/Sandbox

Source: [afk/tools/security.py](https://github.com/socioy/afk/blob/main/src/afk/tools/security.py)

Exported:

- `SandboxProfile`
- `SandboxProfileProvider`
- `SecretScopeProvider`
- `validate_tool_args_against_sandbox(...)`
- `build_registry_sandbox_policy(...)`
- `build_registry_output_limit_middleware(...)`
- `resolve_sandbox_profile(...)`
- `apply_tool_output_limits(...)`

Checks include:

- network URL restrictions
- command allowlist / shell operator blocking
- path allowlist/denylist checks
- output truncation

### Prebuilt Tool Factories

Source:

- [afk/tools/prebuilts/runtime.py](https://github.com/socioy/afk/blob/main/src/afk/tools/prebuilts/runtime.py)
- [afk/tools/prebuilts/skills.py](https://github.com/socioy/afk/blob/main/src/afk/tools/prebuilts/skills.py)

Exported:

- `build_runtime_tools(root_dir=...)`
- `build_skill_tools(skills=..., policy=...)`

Built-in runtime tools:

- `list_directory`
- `read_file`

Built-in skill tools:

- `list_skills`
- `read_skill_md`
- `read_skill_file`
- `run_skill_command`

### Tool Export Helpers

Source: [afk/llms/tool_export.py](https://github.com/socioy/afk/blob/main/src/afk/llms/tool_export.py)

Exported:

- `normalize_json_schema`
- `toolspec_to_openai_tool`
- `tool_to_openai_tool`
- `to_openai_tools`
- `to_openai_tools_from_specs`
- `export_tools_for_provider`

### Tool Errors

Source: [afk/tools/core/errors.py](https://github.com/socioy/afk/blob/main/src/afk/tools/core/errors.py)

Exported:

- `ToolAlreadyRegisteredError`
- `ToolExecutionError`
- `ToolNotFoundError`
- `ToolPolicyError`
- `ToolTimeoutError`
- `ToolValidationError`

---

## `afk.evals`

Source: [afk/evals/**init**.py](https://github.com/socioy/afk/blob/main/src/afk/evals/__init__.py)

Exported:

- `EvalCase`
- `EvalCaseResult`
- `EvalSuiteConfig`
- `EvalSuiteResult`
- `run_case(...)`
- `arun_case(...)`
- `run_suite(...)`
- `arun_suite(...)`
- `EvalBudget`
- assertion/scorer contracts and built-ins
- `compare_event_types(...)`
- `write_golden_trace(...)`
- `load_eval_cases_json(...)`
- `write_suite_report_json(...)`

Key modules:

- `afk/evals/executor.py`
- `afk/evals/suite.py`
- `afk/evals/assertions.py`
- `afk/evals/budgets.py`
- `afk/evals/reporting.py`

---

## Environment Variables Reference

### LLM

- `AFK_LLM_ADAPTER`
- `AFK_LLM_MODEL`
- `AFK_EMBED_MODEL`
- `AFK_LLM_API_BASE_URL`
- `AFK_LLM_API_KEY`
- `AFK_LLM_TIMEOUT_S`
- `AFK_LLM_MAX_RETRIES`
- `AFK_LLM_BACKOFF_BASE_S`
- `AFK_LLM_BACKOFF_JITTER_S`
- `AFK_LLM_JSON_MAX_RETRIES`
- `AFK_LLM_MAX_INPUT_CHARS`

### Memory

- `AFK_MEMORY_BACKEND`
- `AFK_SQLITE_PATH`
- `AFK_REDIS_URL`
- `AFK_REDIS_HOST`
- `AFK_REDIS_PORT`
- `AFK_REDIS_DB`
- `AFK_REDIS_PASSWORD`
- `AFK_REDIS_EVENTS_MAX`
- `AFK_PG_DSN`
- `AFK_PG_HOST`
- `AFK_PG_PORT`
- `AFK_PG_USER`
- `AFK_PG_PASSWORD`
- `AFK_PG_DB`
- `AFK_PG_SSL`
- `AFK_PG_POOL_MIN`
- `AFK_PG_POOL_MAX`
- `AFK_VECTOR_DIM`

---

## Example Code Index

All code samples are Python files and can be linked directly from website/MDX build systems:

- [01_minimal_chat_agent.py](/library/examples/index#01-minimal-chat-agent)
- [02_policy_with_hitl.py](/library/examples/index#02-policy-with-hitl)
- [03_subagents_with_router.py](/library/examples/index#03-subagents-with-router)
- [04_resume_and_compact.py](/library/examples/index#04-resume-and-compact)
- [05_direct_llm_structured_output.py](/library/examples/index#05-direct-llm-structured-output)
- [06_tool_registry_security.py](/library/examples/index#06-tool-registry-security)

---

## Edge Case Matrix

### Runner / Agent Runtime

- `run_id` or `thread_id` empty on resume:
  - `AgentConfigurationError`
- `resume_handle` with missing latest checkpoint:
  - `AgentCheckpointCorruptionError`
- `resume_handle` on terminal checkpoint:
  - returns handle pre-resolved with terminal `AgentResult`
- invalid state transition (for example forced terminal -> running):
  - `AgentExecutionError`
- subagent recursion depth exceeded:
  - `SubagentRoutingError`
- subagent cycle detected:
  - `SubagentRoutingError`
- subagent target unknown:
  - `SubagentRoutingError`
- policy engine exception:
  - wrapped in `AgentExecutionError`
- deferred approval/input timeout:
  - fallback from `RunnerConfig.approval_fallback` / `RunnerConfig.input_fallback`
- tool policy deny:
  - tool record appended as failed; run behavior depends on approval/tool failure policy
- sandbox violation:
  - tool record appended as failed; run behavior depends on tool failure policy
- effect replay hash mismatch:
  - `AgentCheckpointCorruptionError`
- memory backend resolution failure from env:
  - runner falls back to in-memory and emits warning event
- cancellation during streaming LLM call:
  - run cancels or interrupts based on handle flags

### LLM Layer

- `LLMRequest.messages` empty:
  - `LLMError`
- `LLMRequest.model` empty:
  - `LLMError`
- invalid `top_p` or `temperature` values:
  - `LLMError`
- `thinking=False` combined with thinking controls:
  - `LLMError`
- tool definitions malformed:
  - `LLMError`
- unsupported adapter capability requested:
  - `LLMCapabilityError`
- stream emits zero or multiple completion events:
  - `LLMInvalidResponseError`
- structured payload invalid or JSON parse fails after retries:
  - `LLMInvalidResponseError`
- session paused then chat/stream attempted:
  - `LLMSessionPausedError`
- stream interruption requested without provider support:
  - `LLMCapabilityError`

### Tools Layer

- tool function signature invalid:
  - `ToolValidationError` at construction/inference time
- prehook returns non-dict:
  - `ToolResult(success=False)` or exception if `raise_on_error=True`
- middleware signature invalid:
  - `ToolValidationError`
- registry call unknown tool:
  - `ToolNotFoundError`
- registry policy throws:
  - wrapped/raised as `ToolPolicyError`
- registry/tool timeout:
  - `ToolTimeoutError` (or failed `ToolResult` depending on layer)
- command tool with disallowed command/operator:
  - policy violation / `SkillCommandDeniedError` depending on call path

### Memory Layer

- store methods called before `setup()`:
  - runtime error from concrete backend
- vector dimension mismatch:
  - `ValueError`
- Postgres backend missing `AFK_VECTOR_DIM`:
  - `ValueError`
- unknown memory backend name:
  - `ValueError`
- `compact_thread_memory` against backend without delete/replace implementations:
  - compaction summary reports logical removals and `state_keys_removed_effective` may be lower

### Policy Engine

- no matching rules:
  - defaults to `allow`
- multiple matching rules:
  - highest priority then lexicographically smallest `rule_id` wins
- `subjects=["any"]`:
  - bypasses subject inference and only condition matching decides

---
