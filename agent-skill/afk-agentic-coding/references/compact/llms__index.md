# LLM Overview

Provider-agnostic contracts, adapter capabilities, stream/session controls, and agent integration boundaries in afk.llms.

Source: `docs/llms/index.mdx`

`afk.llms` is the transport and normalization layer for all model providers used by AFK.

Use `afk.llms` when you need direct model control. Use `afk.agents` when you need orchestration and tool execution.

## TL;DR

- `afk.llms` provides one stable contract across multiple model providers.
- Adapters expose capability flags so unsupported operations fail explicitly.
- Streaming, sessions, retries, and structured output validation are built into the same interface.

## When to Use

- You need direct request/response control over provider calls.
- You are building custom orchestration outside `afk.agents`.
- You need portable transport behavior across providers.

## What This Layer Owns

- Normalized request/response contracts (`LLMRequest`, `LLMResponse`)
- Adapter abstraction and provider capability gating
- Structured output validation and repair
- Stream control (`cancel`, `interrupt`, `await_result`)
- Session token continuity APIs (adapter-dependent)

## Read by Task

    Understand core request/response types and error guarantees.

    Built-in adapters, capability matrix, and custom adapter rules.

    Handle streaming lifecycle, cancellation, interruption, and session state.

    Clear boundary between orchestration (`afk.agents`) and transport (`afk.llms`).

## Minimal Usage

```python
from afk.llms import LLMRequest, Message, create_llm

llm = create_llm("openai")
req = LLMRequest(
    model="gpt-4.1-mini",
    messages=[Message(role="user", content="Plan the next 3 steps")],
)
resp = await llm.chat(req)
print(resp.text)
```

```python
from afk.llms import LLMRequest, Message, create_llm

llm = create_llm("litellm")
req = LLMRequest(
    model="ollama_chat/gpt-oss:20b",
    messages=[Message(role="user", content="Plan the next 3 steps")],
)
resp = await llm.chat(req)
print(resp.text)
```

## Integration Rule

Keep orchestration in `afk.agents` and `afk.core`.
Keep provider-specific transport concerns in `afk.llms`.

This separation makes retry behavior, observability, and provider migration consistent.

## Continue Reading

1. [Contracts](/llms/contracts)
2. [Adapters](/llms/adapters)
3. [Streaming & Sessions](/llms/control-and-session)
4. [Agent Integration](/llms/agent-integration)
