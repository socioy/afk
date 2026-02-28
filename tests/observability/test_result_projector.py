from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

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
from afk.observability import project_run_metrics_from_result


class _StaticLLM(LLM):
    @property
    def provider_id(self) -> str:
        return "test"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            chat=True,
            streaming=False,
            tool_calling=True,
            structured_output=True,
        )

    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        _ = req
        _ = response_model
        return LLMResponse(text="ok")

    async def _chat_stream_core(
        self, req: LLMRequest, *, response_model=None
    ) -> AsyncIterator:
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError


def test_project_run_metrics_from_agent_result():
    result = asyncio.run(
        Runner().run(
            Agent(model=_StaticLLM(), instructions="obs result"),
            user_message="hello",
        )
    )
    metrics = project_run_metrics_from_result(result)

    assert metrics.run_id == result.run_id
    assert metrics.state == result.state
    assert metrics.tool_calls == len(result.tool_executions)
    assert metrics.input_tokens == result.usage_aggregate.input_tokens
