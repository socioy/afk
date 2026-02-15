# Run Event Contract

This document defines `AgentRunEvent` semantics for runtime consumers.

## TL;DR

- Events are emitted in a deterministic order per run stage.
- `AgentRunHandle.events` is a single-consumer stream.
- The same event is also persisted to memory trace storage.
- Event `data` payload varies by event type and should be parsed defensively.

## When To Read This

- You are building real-time UI or operator dashboards.
- You need alerting on run failures/degraded states.
- You are implementing replay/debug tooling from stored trace events.

Primary code paths:

- [afk/agents/types.py](../../src/afk/agents/types.py)
- [afk/core/runner_execution.py](../../src/afk/core/runner_execution.py)
- [afk/core/runner_interaction.py](../../src/afk/core/runner_interaction.py)
- [afk/core/runner_internals.py](../../src/afk/core/runner_internals.py)

## Event Type Set

`AgentEventType` values:

- `run_started`
- `step_started`
- `policy_decision`
- `llm_called`
- `llm_completed`
- `tool_batch_started`
- `tool_completed`
- `subagent_started`
- `subagent_completed`
- `run_paused`
- `run_resumed`
- `run_cancelled`
- `run_interrupted`
- `run_failed`
- `run_completed`
- `warning`

## Common Event Fields

Every event contains:

- `type`
- `run_id`
- `thread_id`
- `state`
- optional `step`
- optional `message`
- optional `data`
- `schema_version`

## Event Payload Contract

| Event type | Typical `data` keys | Notes |
| --- | --- | --- |
| `run_started` | `agent_name`, `resumed` | emitted once at run start |
| `step_started` | none | emitted each loop step |
| `policy_decision` | `event_type`, `action`, `reason`, `policy_id`, `matched_rules` | emitted by policy audit helper |
| `llm_called` | `model`, `provider` | before llm call |
| `llm_completed` | `tool_call_count`, `finish_reason` | after llm call |
| `tool_batch_started` | `tool_call_count` | when tool calls are present |
| `tool_completed` | `tool_name`, `success` | one per processed tool result |
| `subagent_started` | `subagent_name` | subagent dispatch start |
| `subagent_completed` | `subagent_name`, `success`, optional `error` | subagent completion |
| `run_paused` | none | deferred HITL waits |
| `run_resumed` | none | deferred HITL resolved |
| `run_cancelled` | none | cancellation terminal path |
| `run_interrupted` | none | interrupt terminal path |
| `run_failed` | none | failure terminal path |
| `run_completed` | none | completion terminal path |
| `warning` | none | non-fatal warnings |

## Delivery Guarantees

Event is emitted to all four sinks in order:

1. run handle event queue
2. interaction provider `notify(...)`
3. memory trace event append
4. telemetry event/counter

`_emit(...)` handles telemetry failures as non-fatal.

## Consumption Rules

`AgentRunHandle.events`:

- async stream
- single-consumer only
- closes when run ends/cancels/fails

## Memory Trace Record Shape

Each emitted run event is also persisted as a memory `trace` event payload with:

- `schema_version`
- `type`
- `state`
- `step`
- `message`
- `data`

This supports replay/debug/audit workflows independent of live handle consumers.

## Implementation Checklist

1. Consume events with one dedicated reader per run handle.
2. Handle unknown event types as forward-compatible extensions.
3. Use `state`, `type`, and `step` as primary routing fields.
4. Persist/forward terminal events (`run_completed`, `run_failed`, `run_cancelled`).
5. Guard against missing optional `data` keys in consumer logic.

## Related Example

- [examples/02_policy_with_hitl.py](./examples/02_policy_with_hitl.py)
