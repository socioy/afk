# Checkpoint & Resume

Checkpoint data model, lifecycle phases, resume semantics, and corruption safeguards.

Source: `docs/library/checkpoint-schema.mdx`

This document defines runtime checkpoint keys, phases, and payload expectations.

## TL;DR

- AFK persists both step checkpoints and runtime snapshots.
- Resume first checks latest pointer, then loads latest valid runtime state.
- Terminal checkpoints can short-circuit resume immediately.
- Effect rows are separate from checkpoints and power idempotent replay.

## When to Use

- You are implementing durable execution and resume.
- You are building tooling around checkpoint inspection.
- You are diagnosing checkpoint corruption or migration failures.

Primary code paths:

- [afk/core/runner/internals.py](https://github.com/socioy/afk/blob/main/src/afk/core/runner/internals.py)
- [afk/core/runner/execution.py](https://github.com/socioy/afk/blob/main/src/afk/core/runner/execution.py)
- [afk/core/runner/interaction.py](https://github.com/socioy/afk/blob/main/src/afk/core/runner/interaction.py)
- [afk/agents/lifecycle/runtime.py](https://github.com/socioy/afk/blob/main/src/afk/agents/lifecycle/runtime.py)
- [afk/agents/lifecycle/versioning.py](https://github.com/socioy/afk/blob/main/src/afk/agents/lifecycle/versioning.py)

## Required Capabilities

- persistent memory backend for non-ephemeral runs
- checkpoint schema/version compatibility strategy
- runtime snapshot retention policy aligned to recovery objectives
- corruption detection and alerting on resume failures

## Key Families

- Step checkpoint: `checkpoint:{run_id}:{step}:{phase}`
- Latest pointer: `checkpoint:{run_id}:latest`
- Effect journal: `effect:{run_id}:{step}:{tool_call_id}`

## Checkpoint Record Envelope

Each checkpoint value uses this envelope:

- `schema_version`
- `run_id`
- `step`
- `phase`
- `timestamp_ms`
- `payload` (JSON-safe map)

## Phase Catalog

Observed checkpoint phases in runner paths:

- `run_started`
- `step_started`
- `pre_subagent_batch`
- `post_subagent_batch`
- `pre_llm`
- `post_llm`
- `pre_tool_batch`
- `post_tool_batch`
- `paused`
- `resumed`
- `runtime_state`
- `run_terminal`

## Payload Keys by Phase

| Phase                 | Typical payload keys                                                                                           |
| --------------------- | -------------------------------------------------------------------------------------------------------------- |
| `run_started`         | `agent_name`, `resumed`                                                                                        |
| `step_started`        | `state`, `message_count`                                                                                       |
| `pre_subagent_batch`  | `targets`, `router_parallel`                                                                                   |
| `post_subagent_batch` | `success_count`, `failure_count`                                                                               |
| `pre_llm`             | `model`, `provider`, `message_count`                                                                           |
| `post_llm`            | `model`, `provider`, `finish_reason`, `tool_call_count`, `session_token`, `checkpoint_token`, `total_cost_usd` |
| `pre_tool_batch`      | `tool_call_count`                                                                                              |
| `post_tool_batch`     | `tool_calls_total`, `tool_failures`                                                                            |
| `paused`              | `kind`, and either `reason` or `prompt`                                                                        |
| `resumed`             | `kind`                                                                                                         |
| `runtime_state`       | full snapshot payload (see below)                                                                              |
| `run_terminal`        | `state`, optional `message`, optional model fields, `terminal_result`                                          |

## `runtime_state` Payload Shape

`runtime_state` payload includes:

- `thread_id`
- `step`, `state`
- `context`
- `messages`
- `llm_calls`, `tool_calls`
- `started_at_s`
- `usage` (`input_tokens`, `output_tokens`, `total_tokens`)
- `total_cost_usd`
- `session_token`, `checkpoint_token`
- `requested_model`, `normalized_model`, `provider_adapter`
- `tool_executions`
- `subagent_executions`
- `skill_reads`
- `skill_command_executions`
- `final_text`
- `final_structured`
- `pending_llm_response`
- `final_response`
- `replayed_effect_count`

## Resume Contract

`Runner.resume_handle(...)` behavior:

1. Load `checkpoint:{run_id}:latest`
2. Normalize/migrate checkpoint schema
3. If latest phase is `run_terminal` with `terminal_result`, return immediate terminal handle
4. Else load latest usable `runtime_state`
5. Re-enter run loop with restored runtime snapshot

## Schema Compatibility

Version helpers:

- `check_checkpoint_schema_version(...)`
- `migrate_checkpoint_record(...)`

Invalid or incompatible checkpoint data raises `AgentCheckpointCorruptionError` in runner path.

## Failure Modes

- missing latest pointer prevents reliable resume
- incompatible schema versions break migration/replay paths
- corrupted effect rows can fail idempotent replay validation
- over-aggressive compaction can remove recovery-critical state

## Compaction Expectations

State retention keeps resume-critical keys by design:

- latest pointer per kept run
- latest-step boundary checkpoint
- `run_terminal`
- selected always-keep phases
- recent `runtime_state` rows per run
- bounded effect rows per run

Reference implementation:

- [afk/memory/lifecycle.py](https://github.com/socioy/afk/blob/main/src/afk/memory/lifecycle.py)

## Implementation Checklist

1. Use a persistent memory backend in non-dev environments.
2. Keep schema-version migrations enabled for compatibility.
3. Treat `run_terminal` payload as source of truth for completed runs.
4. Retain enough `runtime_state` rows to support operational recovery.
5. Alert on checkpoint corruption errors; do not silently continue.

## Examples

- [examples/04_resume_and_compact.py](/library/examples/index#04-resume-and-compact)

## Continue Reading

1. [Run Event Contract](/library/run-event-contract)
2. [Failure Policies](/library/failure-policy-matrix)
