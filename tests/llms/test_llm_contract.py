from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import AsyncIterator

import pytest
from pydantic import BaseModel

from afk.llms.config import LLMConfig
from afk.llms.errors import (
    LLMCapabilityError,
    LLMCancelledError,
    LLMConfigurationError,
    LLMError,
    LLMInvalidResponseError,
    LLMSessionPausedError,
)
from afk.llms.factory import create_llm, create_llm_from_env
from afk.llms.llm import LLM
from afk.llms.middleware import MiddlewareStack
from afk.llms.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
    Message,
    StreamCompletedEvent,
    StreamTextDeltaEvent,
)


def run_async(coro):
    return asyncio.run(coro)


class Out(BaseModel):
    value: int


class DummyLLM(LLM):
    def __init__(
        self,
        *,
        chat_responses: list[LLMResponse] | None = None,
        stream_events=None,
        embed_response: EmbeddingResponse | None = None,
        capabilities: LLMCapabilities | None = None,
        middlewares: MiddlewareStack | None = None,
        thinking_effort_aliases: dict[str, str] | None = None,
        supported_thinking_efforts: set[str] | None = None,
        default_thinking_effort: str | None = None,
        config: LLMConfig | None = None,
    ) -> None:
        super().__init__(
            config=config,
            middlewares=middlewares,
            thinking_effort_aliases=thinking_effort_aliases,
            supported_thinking_efforts=supported_thinking_efforts,
            default_thinking_effort=default_thinking_effort,
        )
        self._chat_responses = chat_responses or [LLMResponse(text="ok")]
        self._stream_events = stream_events or [
            StreamTextDeltaEvent(delta="ok"),
            StreamCompletedEvent(response=LLMResponse(text="ok")),
        ]
        self._embed_response = embed_response or EmbeddingResponse(embeddings=[[0.1, 0.2]])
        self._caps = capabilities or LLMCapabilities(
            chat=True,
            streaming=True,
            tool_calling=True,
            structured_output=True,
            embeddings=True,
            idempotency=True,
        )
        self.chat_calls = 0
        self.embed_calls = 0

    @property
    def provider_id(self) -> str:
        return "dummy"

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._caps

    async def _chat_core(self, req, *, response_model=None) -> LLMResponse:
        _ = req
        _ = response_model
        self.chat_calls += 1
        idx = min(self.chat_calls - 1, len(self._chat_responses) - 1)
        return self._chat_responses[idx]

    async def _chat_stream_core(self, req, *, response_model=None) -> AsyncIterator:
        _ = req
        _ = response_model

        async def _iter():
            for event in self._stream_events:
                yield event

        return _iter()

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        self.embed_calls += 1
        return self._embed_response


def test_llmresponse_does_not_expose_legacy_typo_field():
    resp = LLMResponse(text="x")
    assert not hasattr(resp, "structued_response")
    with pytest.raises(AttributeError):
        _ = resp.structued_response  # type: ignore[attr-defined]


def test_chat_structured_validation_from_text():
    llm = DummyLLM(chat_responses=[LLMResponse(text='{"value": 7}')])
    req = LLMRequest(model="demo", messages=[Message(role="user", content="hi")])

    out = run_async(llm.chat(req, response_model=Out))

    assert out.structured_response == {"value": 7}
    assert isinstance(out.request_id, str) and out.request_id
    assert llm.chat_calls == 1


def test_chat_repair_retry_until_valid():
    llm = DummyLLM(
        chat_responses=[
            LLMResponse(text="not json"),
            LLMResponse(text='{"value": 2}'),
        ]
    )
    req = LLMRequest(model="demo", messages=[Message(role="user", content="hi")])

    out = run_async(llm.chat(req, response_model=Out))

    assert out.structured_response == {"value": 2}
    assert llm.chat_calls == 2


def test_chat_structured_raises_when_exhausted_retries():
    llm = DummyLLM(
        chat_responses=[
            LLMResponse(text="bad"),
            LLMResponse(text="still bad"),
            LLMResponse(text="still bad"),
        ]
    )
    req = LLMRequest(model="demo", messages=[Message(role="user", content="hi")])

    with pytest.raises(LLMInvalidResponseError):
        run_async(llm.chat(req, response_model=Out))


def test_embed_capability_error_when_disabled():
    llm = DummyLLM(
        capabilities=LLMCapabilities(
            chat=True,
            streaming=True,
            tool_calling=True,
            structured_output=True,
            embeddings=False,
        )
    )

    with pytest.raises(LLMCapabilityError):
        run_async(llm.embed(EmbeddingRequest(model="embed", inputs=["a"])))


def test_middleware_order_chat_embed_stream():
    order: list[str] = []

    async def chat_mw(call_next, req):
        order.append("chat_before")
        out = await call_next(req)
        order.append("chat_after")
        return out

    async def embed_mw(call_next, req):
        order.append("embed_before")
        out = await call_next(req)
        order.append("embed_after")
        return out

    def stream_mw(call_next, req):
        async def _iter():
            order.append("stream_before")
            async for event in call_next(req):
                order.append(f"stream_event:{event.type}")
                yield event
            order.append("stream_after")

        return _iter()

    llm = DummyLLM(
        middlewares=MiddlewareStack(
            chat=[chat_mw],
            embed=[embed_mw],
            stream=[stream_mw],
        )
    )

    req = LLMRequest(model="demo", messages=[Message(role="user", content="hi")])
    run_async(llm.chat(req))
    run_async(llm.embed(EmbeddingRequest(model="embed", inputs=["a"])))

    async def consume_stream():
        stream = await llm.chat_stream(req)
        return [event async for event in stream]

    events = run_async(consume_stream())

    assert any(e.type == "completed" for e in events)
    assert order == [
        "chat_before",
        "chat_after",
        "embed_before",
        "embed_after",
        "stream_before",
        "stream_event:text_delta",
        "stream_event:completed",
        "stream_after",
    ]


def test_chat_stream_validates_completion_payload_when_response_model():
    llm = DummyLLM(
        stream_events=[
            StreamTextDeltaEvent(delta="{"),
            StreamCompletedEvent(response=LLMResponse(text='{"value": 11}')),
        ]
    )
    req = LLMRequest(model="demo", messages=[Message(role="user", content="hi")])

    async def scenario():
        stream = await llm.chat_stream(req, response_model=Out)
        return [event async for event in stream]

    events = run_async(scenario())
    completed = [e for e in events if isinstance(e, StreamCompletedEvent)]
    assert len(completed) == 1
    assert completed[0].response.structured_response == {"value": 11}


def test_factory_resolves_adapter_from_env(monkeypatch):
    monkeypatch.setenv("AFK_LLM_ADAPTER", "anthropic_agent")
    llm = create_llm_from_env()
    assert llm.provider_id == "anthropic_agent"


def test_factory_rejects_unknown_adapter():
    with pytest.raises(LLMConfigurationError):
        create_llm("not_real")


def test_factory_supports_openai_adapter():
    llm = create_llm("openai")
    assert llm.provider_id == "openai"


def test_factory_passes_thinking_overrides_to_builtin_adapters():
    llm = create_llm("litellm", default_thinking_effort="balanced")
    req = LLMRequest(
        model="demo",
        messages=[Message(role="user", content="hi")],
        thinking=True,
    )
    resolved = llm.resolve_thinking(req)
    assert resolved.effort == "balanced"


def test_chat_rejects_invalid_thinking_combo():
    llm = DummyLLM()
    req = LLMRequest(
        model="demo",
        messages=[Message(role="user", content="hi")],
        thinking=False,
        thinking_effort="high",
    )
    with pytest.raises(LLMError):
        run_async(llm.chat(req))


def test_chat_stream_handle_cancel():
    class SlowDummy(DummyLLM):
        async def _chat_stream_core(self, req, *, response_model=None) -> AsyncIterator:
            _ = req
            _ = response_model

            async def _iter():
                yield StreamTextDeltaEvent(delta="chunk")
                await asyncio.sleep(10)

            return _iter()

    llm = SlowDummy()
    req = LLMRequest(model="demo", messages=[Message(role="user", content="hi")])

    async def scenario():
        handle = await llm.chat_stream_handle(req)
        stream = handle.events
        first = await anext(stream)
        await handle.cancel()
        with pytest.raises(LLMCancelledError):
            await anext(stream)
        return first, await handle.await_result()

    first, result = run_async(scenario())
    assert first.type == "text_delta"
    assert result is None


def test_chat_stream_handle_interrupt_unsupported():
    llm = DummyLLM(
        capabilities=LLMCapabilities(
            chat=True,
            streaming=True,
            tool_calling=True,
            structured_output=True,
            embeddings=True,
            interrupt=False,
        )
    )
    req = LLMRequest(model="demo", messages=[Message(role="user", content="hi")])

    async def scenario():
        handle = await llm.chat_stream_handle(req)
        with pytest.raises(LLMCapabilityError):
            await handle.interrupt()

    run_async(scenario())


def test_start_session_raises_when_unsupported():
    llm = DummyLLM(
        capabilities=LLMCapabilities(
            chat=True,
            streaming=True,
            tool_calling=True,
            structured_output=True,
            embeddings=True,
            session_control=False,
        )
    )
    with pytest.raises(LLMCapabilityError):
        llm.start_session()


def test_session_pause_resume_behavior():
    llm = DummyLLM(
        capabilities=LLMCapabilities(
            chat=True,
            streaming=True,
            tool_calling=True,
            structured_output=True,
            embeddings=True,
            session_control=True,
        )
    )
    req = LLMRequest(model="demo", messages=[Message(role="user", content="hi")])

    async def scenario():
        session = llm.start_session(session_token="session_1")
        await session.pause()
        with pytest.raises(LLMSessionPausedError):
            await session.chat(req)
        await session.resume()
        out = await session.chat(req)
        snap = await session.snapshot()
        return out, snap

    out, snap = run_async(scenario())
    assert out.text == "ok"
    assert snap.session_token == "session_1"
    assert snap.paused is False


def test_embed_model_falls_back_to_config():
    cfg = replace(LLMConfig.from_env(), embedding_model="embed-default")
    llm = DummyLLM(config=cfg)

    out = run_async(llm.embed(EmbeddingRequest(inputs=["a"])))

    assert out.embeddings == [[0.1, 0.2]]
    assert llm.embed_calls == 1


def test_embed_raises_when_model_unresolved():
    cfg = replace(LLMConfig.from_env(), embedding_model=None)
    llm = DummyLLM(config=cfg)
    with pytest.raises(LLMConfigurationError):
        run_async(llm.embed(EmbeddingRequest(inputs=["a"])))


def test_chat_allows_provider_specific_effort_by_default():
    llm = DummyLLM()
    req = LLMRequest(
        model="demo",
        messages=[Message(role="user", content="hi")],
        thinking=True,
        thinking_effort="balanced",
    )

    out = run_async(llm.chat(req))

    assert out.text == "ok"


def test_thinking_effort_alias_is_applied_from_instance_override():
    llm = DummyLLM(
        thinking_effort_aliases={"balanced": "medium"},
        supported_thinking_efforts={"low", "medium", "high"},
    )
    req = LLMRequest(
        model="demo",
        messages=[Message(role="user", content="hi")],
        thinking=True,
        thinking_effort="balanced",
    )

    resolved = llm.resolve_thinking(req)

    assert resolved.effort == "medium"


def test_thinking_default_effort_is_overrideable_per_instance():
    llm = DummyLLM(default_thinking_effort="balanced")
    req = LLMRequest(
        model="demo",
        messages=[Message(role="user", content="hi")],
        thinking=True,
    )

    resolved = llm.resolve_thinking(req)

    assert resolved.effort == "balanced"


def test_thinking_effort_rejects_unknown_when_supported_set_defined():
    llm = DummyLLM(supported_thinking_efforts={"minimal", "balanced", "deep"})
    req = LLMRequest(
        model="demo",
        messages=[Message(role="user", content="hi")],
        thinking=True,
        thinking_effort="high",
    )

    with pytest.raises(LLMError):
        run_async(llm.chat(req))


class SubclassThinkingLLM(DummyLLM):
    def _provider_supported_thinking_efforts(self) -> set[str] | None:
        return {"minimal", "balanced", "deep"}

    def _provider_default_thinking_effort(self) -> str | None:
        return "balanced"


def test_thinking_effort_can_be_overridden_by_subclass_policy():
    llm = SubclassThinkingLLM()
    req = LLMRequest(
        model="demo",
        messages=[Message(role="user", content="hi")],
        thinking=True,
    )

    resolved = llm.resolve_thinking(req)
    assert resolved.effort == "balanced"

    bad_req = LLMRequest(
        model="demo",
        messages=[Message(role="user", content="hi")],
        thinking=True,
        thinking_effort="high",
    )
    with pytest.raises(LLMError):
        run_async(llm.chat(bad_req))
