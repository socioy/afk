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
from afk.observability import (
    RuntimeTelemetryCollector,
    project_run_metrics_from_collector,
)


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


def test_project_run_metrics_from_runtime_collector():
    collector = RuntimeTelemetryCollector()
    agent = Agent(model=_StaticLLM(), instructions="obs")
    result = asyncio.run(Runner(telemetry=collector).run(agent, user_message="hello"))

    metrics = project_run_metrics_from_collector(collector)

    assert result.state == "completed"
    assert metrics.run_id == result.run_id
    assert metrics.state == "completed"
    assert metrics.llm_calls >= 1
    assert metrics.tool_calls == 0
