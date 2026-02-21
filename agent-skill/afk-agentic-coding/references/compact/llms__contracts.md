# Contracts

Normalized LLM request, response, streaming, embeddings, capability, and error contracts.

Source: `docs/llms/contracts.mdx`

The `afk.llms` package exposes one normalized interaction contract across all adapters.

## TL;DR

- `LLMRequest`, `LLMResponse`, and `LLMStreamEvent` are the core transport contracts.
- Adapters must declare capabilities and fail fast on unsupported operations.
- Error taxonomy is explicit so caller behavior can be deterministic.

## When to Use

- You are implementing or reviewing adapter-level integrations.
- You need exact field-level behavior for request/response handling.
- You are defining stable interfaces between orchestration and transport layers.

## Core Request/Response Types

- `LLMRequest`
  - Required: `model`, `messages`
  - Correlation/control: `request_id`, `idempotency_key`, `session_token`, `checkpoint_token`
  - Generation controls: `max_tokens`, `temperature`, `top_p`, `stop`
  - Reasoning controls: `thinking`, `thinking_effort`, `max_thinking_tokens`
  - Tool controls: `tools`, `tool_choice`
  - Transport extras: `timeout_s`, `metadata`, `extra`
- `LLMResponse`
  - Core output: `text`, `structured_response`, `tool_calls`, `finish_reason`, `usage`
  - Correlation/control: `request_id`, `provider_request_id`, `session_token`, `checkpoint_token`
  - Debug: `raw`, `model`
- `EmbeddingRequest`
  - `model` is optional and resolves from `LLMConfig.embedding_model` when omitted
  - Batch-only inputs: `inputs: list[str]`
  - Transport extras: `timeout_s`, `metadata`, `extra`
- `EmbeddingResponse`
  - `embeddings`, `model`, `raw`

## Base Client API

`LLM` is the single client type agents should depend on:

- `chat(req, response_model=None) -> LLMResponse`
- `chat_sync(req, response_model=None) -> LLMResponse`
- `chat_stream(req, response_model=None) -> AsyncIterator[LLMStreamEvent]`
- `chat_stream_handle(req, response_model=None) -> LLMStreamHandle`
- `embed(req) -> EmbeddingResponse`
- `embed_sync(req) -> EmbeddingResponse`
- `start_session(...) -> LLMSessionHandle` (capability-gated)

## Capability Flags

`LLMCapabilities` is explicit and adapter-defined:

- `chat`
- `streaming`
- `tool_calling`
- `structured_output`
- `embeddings`
- `interrupt`
- `session_control`
- `checkpoint_resume`
- `idempotency`

Unsupported calls must raise `LLMCapabilityError`.

## Error Contract

- `LLMError` (base)
- `LLMRetryableError`
- `LLMInvalidResponseError`
- `LLMConfigurationError`
- `LLMCapabilityError`
- `LLMCancelledError`
- `LLMInterruptedError`
- `LLMSessionError`
- `LLMSessionPausedError`

## Continue Reading

1. [LLM Overview](/llms/index)
2. [Adapters](/llms/adapters)
3. [Streaming & Sessions](/llms/control-and-session)
