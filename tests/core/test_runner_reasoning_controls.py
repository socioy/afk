from __future__ import annotations

import asyncio

import pytest

from afk.agents import Agent
from afk.core import Runner
from afk.llms import LLM
from afk.llms.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
)


class _CaptureReasoningLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[LLMRequest] = []

    @property
    def provider_id(self) -> str:
        return "reasoning"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(chat=True, streaming=False, tool_calling=True)

    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        _ = response_model
        self.requests.append(req)
        return LLMResponse(text="done")

    async def _chat_stream_core(self, req: LLMRequest, *, response_model=None):
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError


def test_agent_reasoning_defaults_propagate_to_llm_request():
    async def _scenario() -> LLMRequest:
        llm = _CaptureReasoningLLM()
        agent = Agent(
            model=llm,
            instructions="x",
            reasoning_enabled=True,
            reasoning_effort="low",
            reasoning_max_tokens=128,
        )
        await Runner().run(agent, user_message="go")
        return llm.requests[0]

    req = asyncio.run(_scenario())
    assert req.thinking is True
    assert req.thinking_effort == "low"
    assert req.max_thinking_tokens == 128


def test_context_reasoning_override_supersedes_agent_defaults():
    async def _scenario() -> LLMRequest:
        llm = _CaptureReasoningLLM()
        agent = Agent(
            model=llm,
            instructions="x",
            reasoning_enabled=True,
            reasoning_effort="low",
            reasoning_max_tokens=128,
        )
        await Runner().run(
            agent,
            user_message="go",
            context={
                "_afk": {
                    "reasoning": {
                        "enabled": True,
                        "effort": "high",
                        "max_tokens": 256,
                    }
                }
            },
        )
        return llm.requests[0]

    req = asyncio.run(_scenario())
    assert req.thinking is True
    assert req.thinking_effort == "high"
    assert req.max_thinking_tokens == 256


def test_invalid_reasoning_combo_is_rejected_by_llm_validation():
    async def _scenario() -> None:
        llm = _CaptureReasoningLLM()
        agent = Agent(
            model=llm,
            instructions="x",
            reasoning_enabled=False,
            reasoning_effort="high",
        )
        await Runner().run(agent, user_message="go")

    with pytest.raises(Exception):
        asyncio.run(_scenario())
