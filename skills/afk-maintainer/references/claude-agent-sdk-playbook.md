# Claude Agent SDK Maintainer Playbook

Guidelines for changes affecting AFK's integration with `claude-agent-sdk`.

---

## Scope

This playbook covers:
- `src/afk/llms/clients/adapters/anthropic_agent.py` -- adapter implementation
- `src/afk/llms/providers/anthropic_agent.py` -- provider registration
- Any code that imports from or interacts with `claude_agent_sdk`

---

## Architecture Context

The Claude Agent SDK adapter translates between AFK's normalized types and the
Claude Agent SDK's native types:

```
AFK LLMRequest  -->  Claude Agent SDK Request
                     (via adapter transform)
Claude Agent SDK Response  -->  AFK LLMResponse
                               (via adapter normalization)
```

The adapter is one of several provider implementations. It must:
1. Accept AFK-normalized `LLMRequest` objects.
2. Transform them into Claude Agent SDK format.
3. Execute the request via the SDK.
4. Normalize the response back to AFK `LLMResponse`.
5. Map SDK errors to AFK's error hierarchy.

---

## Maintainer Expectations

- **Keep adapter behavior explicit and traceable.** Every transform from AFK types to SDK types should be a clear, testable function.
- **Validate streaming and terminal event behavior** after any adapter change. The streaming contract is the most fragile part.
- **Preserve consistent error mapping** to AFK's error hierarchy (`LLMError`, `LLMRetryableError`, `LLMTimeoutError`).
- **Guard optional SDK symbols** for test environments where the SDK may not be installed.
- **Never leak SDK-specific types** into AFK's public API surface.

---

## Change Review Checklist

When reviewing a PR that touches the Claude Agent SDK adapter:

- [ ] Are request/response transforms still deterministic? (Same AFK input -> same SDK input)
- [ ] Are interrupt/cancel semantics preserved? (AFK cancel -> SDK cancel, with correct state)
- [ ] Are tool call payload assumptions validated? (Schema, argument types, nested objects)
- [ ] Are optional SDK symbols guarded with `try/except ImportError`?
- [ ] Are streaming events normalized correctly to AFK's `LLMStreamEvent` types?
- [ ] Are error codes/exceptions mapped to the correct AFK error type?
- [ ] Is the thinking/reasoning control correctly forwarded (effort levels, budget tokens)?
- [ ] Are new SDK features gated behind capability checks?

---

## Regression Tests to Prioritize

After any Claude Agent SDK adapter change, verify:

1. **Non-streaming chat flow** -- request/response roundtrip produces valid AFK types.
2. **Streaming event normalization** -- all event types (text_delta, tool_use, thinking, done) map correctly.
3. **Structured output parsing** -- JSON schema enforcement and fallback parsing work.
4. **Tool call extraction** -- tool name, arguments, and call IDs are correctly extracted.
5. **Interrupt/cancel behavior** -- interrupting a stream produces correct state and error type.
6. **Timeout behavior** -- request timeouts produce `LLMTimeoutError`, not generic exceptions.
7. **Error mapping** -- SDK-specific errors map to correct AFK error hierarchy.
8. **Thinking/reasoning** -- effort levels and thinking budget tokens are forwarded correctly.

---

## Common Pitfalls

- **SDK version drift**: The Claude Agent SDK is pre-1.0 and may change APIs between versions. Always test against the pinned version after any change.
- **Streaming event shape changes**: SDK may add new event types or change existing ones. The adapter must handle unknown event types gracefully (log + skip, not crash).
- **Token counting differences**: SDK may count tokens differently than AFK expects. Verify `usage` fields in responses.
- **Thinking budget**: The SDK's thinking/extended-thinking feature has its own budget semantics. AFK's `reasoning_max_tokens` must be correctly forwarded.

---

## Documentation

When behavior changes, update:

- API/behavior docs relevant to provider integrations (`docs/llms/adapters.mdx`).
- Release note with provider scope (`Changed` or `Fixed` heading, mention "Claude Agent SDK").
- Configuration reference if new adapter-specific options are added.
- Provider compatibility matrix in `dependency-and-compatibility-rules.md`.
