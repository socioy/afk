# LiteLLM Maintainer Playbook

Guidelines for changes affecting AFK's integration with LiteLLM.

---

## Scope

This playbook covers:
- `src/afk/llms/clients/adapters/litellm.py` -- adapter implementation
- `src/afk/llms/providers/litellm.py` -- provider registration
- Any code that imports from or interacts with `litellm`

---

## Architecture Context

The LiteLLM adapter provides multi-provider routing through a single transport:

```
AFK LLMRequest  -->  LiteLLM acompletion/completion
                     (LiteLLM handles provider routing internally)
LiteLLM Response  -->  AFK LLMResponse
                       (via adapter normalization)
```

LiteLLM's value is provider multiplexing: one adapter supports OpenAI, Anthropic,
Google, Cohere, and many more through LiteLLM's routing layer. The AFK adapter's
job is to:

1. Transform AFK `LLMRequest` into LiteLLM-compatible kwargs.
2. Forward the call to LiteLLM's completion/chat API.
3. Normalize the response back to AFK `LLMResponse`.
4. Map LiteLLM errors to AFK's error hierarchy.
5. Handle the sync/async bridge correctly (LiteLLM has both paths).

---

## Maintainer Expectations

- **Keep provider-specific fields isolated in the adapter layer.** AFK's `LLMRequest` must never contain LiteLLM-specific fields.
- **Preserve AFK-normalized request/response types.** The adapter must output standard AFK types regardless of which backend provider LiteLLM routes to.
- **Validate fallback, retry, and timeout policy interactions.** AFK's `LLMClient` has its own retry/fallback logic that wraps the adapter. These must not conflict with LiteLLM's internal retry behavior.
- **Handle LiteLLM version changes carefully.** LiteLLM releases frequently and may change internal APIs.

---

## Change Review Checklist

When reviewing a PR that touches the LiteLLM adapter:

- [ ] Request payload shape is still valid for target LiteLLM APIs (check against LiteLLM docs).
- [ ] Structured output behavior remains stable across different backend providers.
- [ ] Tool-call extraction and streaming transforms remain compatible.
- [ ] Error wrapping produces actionable AFK error types (not raw LiteLLM exceptions).
- [ ] The sync bridge (`chat_sync` path) uses the correct LiteLLM sync API (not blocking the loop).
- [ ] Provider-specific kwargs are not leaking into AFK's normalized types.
- [ ] Token usage statistics are correctly normalized across different providers.
- [ ] Streaming events from different providers are normalized to AFK's event types.

---

## Regression Tests to Prioritize

After any LiteLLM adapter change, verify:

1. **Basic chat + sync bridge** -- both async and sync paths produce valid AFK responses.
2. **Streaming events** -- stream handling and completion events are correctly normalized.
3. **Tool call extraction** -- tool name, arguments, and call IDs parsed correctly.
4. **Structured output** -- JSON schema enforcement works across providers.
5. **Fallback behavior** -- AFK's runtime client fallback chain works with LiteLLM providers.
6. **Error wrapping** -- LiteLLM exceptions are mapped to the correct AFK error hierarchy.
7. **Timeout behavior** -- AFK-level timeouts correctly cancel LiteLLM requests.

---

## Common Pitfalls

- **Double retry**: Both AFK's `LLMClient` and LiteLLM have retry logic. Ensure AFK's retry wraps LiteLLM's adapter call, and LiteLLM's internal retries are disabled or configured to not conflict.
- **Provider-specific response shapes**: Different providers return different metadata in LiteLLM responses. The adapter must handle missing fields gracefully.
- **Streaming inconsistency**: Different providers have different streaming behaviors through LiteLLM. Some may not support streaming at all. The adapter must detect and handle this.
- **Token counting variance**: Token counts from different providers via LiteLLM may use different counting methods. Normalize to AFK's `Usage` type consistently.
- **Version pinning**: LiteLLM is pinned to an exact version (`1.81.13`). Any update requires full regression testing.

---

## Documentation

When behavior changes, update:

- Provider integration docs/examples (`docs/llms/adapters.mdx`).
- Changelog entries with migration notes when applicable.
- Provider compatibility matrix in `dependency-and-compatibility-rules.md`.
- Any examples that use LiteLLM-routed providers.
