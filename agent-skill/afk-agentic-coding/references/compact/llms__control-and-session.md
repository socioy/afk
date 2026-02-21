# Streaming & Sessions

Streaming control handle semantics and token-based session continuity behavior.

Source: `docs/llms/control-and-session.mdx`

## TL;DR

- Stream handles add cancellation/interruption controls without changing basic stream consumption.
- Session handles expose token continuity primitives when the adapter supports session control.
- Persistence remains the responsibility of orchestration/storage layers, not `afk.llms`.

## When to Use

- You need to cancel or interrupt long-running stream calls.
- You need adapter-native session continuity tokens.
- You are implementing resumable workflows outside the agent runtime.

## Stream Handles

`chat_stream_handle()` adds control primitives without breaking `chat_stream()`:

- `events: AsyncIterator[LLMStreamEvent]`
- `cancel()`
- `interrupt()`
- `await_result() -> LLMResponse | None`

Behavior:

- Stream contract guarantees exactly one terminal completion event for successful streams.
- `cancel()` ends local stream consumption and surfaces `LLMCancelledError` to the event iterator.
- `interrupt()` calls provider interrupt when supported, otherwise raises `LLMCapabilityError`.
- `await_result()` returns:
  - `LLMResponse` on normal completion
  - `None` on cancellation/interruption

## Session Handles

`start_session()` returns `LLMSessionHandle` when `capabilities.session_control=True`.

Methods:

- `chat(req, response_model=None)`
- `stream(req, response_model=None) -> LLMStreamHandle`
- `pause()`
- `resume(session_token=None)`
- `interrupt()`
- `close()`
- `snapshot() -> LLMSessionSnapshot`

Notes:

- Session handles are token-based continuity wrappers, not orchestration engines.
- `pause()` blocks chat/stream calls with `LLMSessionPausedError`.
- `snapshot()` returns opaque continuity tokens; persist these in agent/storage layers.

## State Ownership

- `afk.llms` does not persist sessions/checkpoints.
- It only returns opaque `session_token` and `checkpoint_token`.
- Agent/storage layers must store and re-inject tokens into future `LLMRequest`s.

## Continue Reading

1. [Contracts](/llms/contracts)
2. [Adapters](/llms/adapters)
3. [Agent Integration](/llms/agent-integration)
