# Adapters

Built-in adapter behavior, transport notes, and custom adapter implementation guidance.

Source: `docs/llms/adapters.mdx`

## TL;DR

- AFK ships multiple adapters behind one normalized interface.
- Capability flags define what each adapter can and cannot do.
- Custom adapters must return normalized AFK types and raise explicit capability errors.

## When to Use

- You are selecting a provider integration for production.
- You are implementing a custom transport adapter.
- You need to compare capability differences across adapters.

## Built-in Adapters

- `OpenAIClient` (`afk.llms.clients.adapters.openai.OpenAIClient`)
- `LiteLLMClient` (`afk.llms.clients.adapters.litellm.LiteLLMClient`)
- `AnthropicAgentClient` (`afk.llms.clients.adapters.anthropic_agent.AnthropicAgentClient`)

Factory entry points:

- `create_llm(adapter=...)`
- `create_llm_from_env()` with `AFK_LLM_ADAPTER`

## Capability Matrix

| Adapter | chat | stream | tools | structured | embeddings | interrupt | session control | checkpoint resume | idempotency |
|---|---|---|---|---|---|---|---|---|---|
| OpenAI | yes | yes | yes | yes | yes | no | no | no | yes |
| LiteLLM | yes | yes | yes | yes | yes | no | no | no | yes |
| Anthropic Agent SDK | yes | yes | yes | yes | no | yes | yes | yes | no |

## Transport Notes

### OpenAI and LiteLLM

- Use Responses-style payload mapping through shared `ResponsesClientBase`.
- Map `LLMRequest.stop` into payload.
- Carry request/session/checkpoint correlation in metadata:
  - `afk_request_id`
  - `afk_session_token`
  - `afk_checkpoint_token`
- Map `idempotency_key` into transport headers.
- Extract provider request id into `LLMResponse.provider_request_id` when available.

### Anthropic Agent SDK

- Uses `claude_agent_sdk` query/client transport and normalizes all outputs into AFK types.
- Session continuity is token-driven (`session_token`/`checkpoint_token`).
- Embeddings are unsupported and raise `LLMCapabilityError`.

## Implementing a Custom Adapter

Subclass `LLM` and implement:

- `_chat_core`
- `_chat_stream_core`
- `_embed_core`
- `provider_id`
- `capabilities`

Optional overrides:

- `_interrupt_request` for provider interrupt support
- thinking-effort policy hooks:
  - `_provider_thinking_effort_aliases`
  - `_provider_supported_thinking_efforts`
  - `_provider_default_thinking_effort`

Rules:

- Return only normalized AFK types (`LLMResponse`, `EmbeddingResponse`, `LLMStreamEvent`).
- Raise `LLMCapabilityError` for unsupported feature paths.
- Do not leak provider-specific raw request types into public API contracts.

## Continue Reading

1. [Contracts](/llms/contracts)
2. [Streaming & Sessions](/llms/control-and-session)
3. [Agent Integration](/llms/agent-integration)
